#!/usr/bin/env python3
"""
FastDepth ONNX → RKNN 转换脚本（RK3588）
=============================================
功能：
  1. 加载 ONNX 模型
  2. 生成 FP16（非量化）RKNN 模型
  3. 生成 INT8（PTQ 量化）RKNN 模型
  4. 可选连板精度验证

环境要求：
  - RKNN Toolkit2 (x86_64 Ubuntu 18.04/20.04)
  - pip install rknn_toolkit2-*-cp38-cp38-linux_x86_64.whl

用法：
  # 仅转换（不连板）
  python deploy/rk3588/convert_to_rknn.py \
      --onnx deploy/rk3588/fastdepth_512x768_fp32.onnx \
      --dataset data/usegeo/val/rgb \
      --output-dir deploy/rk3588

  # 转换 + 连板推理验证
  python deploy/rk3588/convert_to_rknn.py \
      --onnx deploy/rk3588/fastdepth_512x768_fp32.onnx \
      --dataset data/usegeo/val/rgb \
      --output-dir deploy/rk3588 \
      --target rk3588 \
      --device-id <your_device_id>
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

# ============================================================
# 配置区域（根据你的模型训练配置调整）
# ============================================================
# 模型输入：RGB [0, 1]，shape=(1, 3, 512, 768)
# 训练时 Data.py: rgb_tensor = transforms.ToTensor()(rgb) → [0,1]
# 模型输出：单通道深度图 (1, 1, 512, 768)，经过 softplus 后为正值
INPUT_HEIGHT = 512
INPUT_WIDTH = 768
INPUT_CHANNELS = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="FastDepth ONNX → RKNN 转换（RK3588 NPU）"
    )
    parser.add_argument("--onnx", required=True, type=str,
                        help="输入 ONNX 模型路径")
    parser.add_argument("--output-dir", default="deploy/rk3588", type=str,
                        help="输出目录")
    parser.add_argument("--dataset", default="data/usegeo/val/rgb", type=str,
                        help="Calibration 数据集目录（RGB 图像）")
    parser.add_argument("--num-calib", default=50, type=int,
                        help="Calibration 图像数量")
    parser.add_argument("--height", default=INPUT_HEIGHT, type=int,
                        help="模型输入高度")
    parser.add_argument("--width", default=INPUT_WIDTH, type=int,
                        help="模型输入宽度")
    parser.add_argument("--target", default=None, type=str,
                        help="连板推理目标平台 (rk3588 / None=模拟器)")
    parser.add_argument("--device-id", default=None, type=str,
                        help="设备编号（多设备连接时指定）")
    parser.add_argument("--skip-fp16", action="store_true",
                        help="跳过 FP16 转换")
    parser.add_argument("--skip-int8", action="store_true",
                        help="跳过 INT8 转换")
    parser.add_argument("--name-prefix", default="fastdepth", type=str,
                        help="输出文件名前缀 (默认 fastdepth)")
    parser.add_argument("--verbose", action="store_true",
                        help="输出详细日志")
    return parser.parse_args()


def prepare_calibration_images(dataset_dir, num_images, height, width):
    """
    从 RGB 图像目录中选取前 num_images 张作为 calibration 数据集，
    返回 numpy array，shape=(N, H, W, C)，dtype=float32，值域 [0, 1]
    """
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([
        f for f in os.listdir(dataset_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if len(image_files) == 0:
        raise FileNotFoundError(f"在 {dataset_dir} 中未找到任何图像文件")

    selected = image_files[:num_images]
    print(f"[Calibration] 从 {dataset_dir} 选取 {len(selected)} 张图像")

    images = []
    for fname in selected:
        img_path = os.path.join(dataset_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((width, height), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0  # → [0, 1]
        images.append(img_np)

    return np.stack(images, axis=0), selected


def create_calibration_txt(image_files, dataset_dir, output_path):
    """创建 calibration 文件列表（供 RKNN Toolkit2 dataset 参数使用）"""
    with open(output_path, 'w') as f:
        for fname in image_files:
            f.write(os.path.join(dataset_dir, fname) + '\n')
    print(f"[Calibration] 列表文件已保存到: {output_path}")
    return output_path


def convert_fp16(rknn, onnx_path, output_dir, height, width, name_prefix="fastdepth"):
    """FP16（非量化）模型转换"""
    output_path = os.path.join(output_dir, f"{name_prefix}_{height}x{width}_fp16.rknn")
    print("\n" + "=" * 60)
    print("[FP16] 开始转换...")
    print("=" * 60)

    # 配置
    rknn.config(
        mean_values=[[0, 0, 0]],        # 输入已归一化到 [0,1]，不需减均值
        std_values=[[255, 255, 255]],   # 反归一化乘回 255，适配 NPU 内部处理
        target_platform='rk3588',
    )

    # 加载 ONNX
    print(f"[FP16] 加载 ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError(f"load_onnx 失败, 返回码: {ret}")

    # 构建（不量化）
    print("[FP16] 构建 RKNN 模型（非量化）...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"build 失败, 返回码: {ret}")

    # 导出
    ret = rknn.export_rknn(export_path=output_path)
    if ret != 0:
        raise RuntimeError(f"export_rknn 失败, 返回码: {ret}")

    print(f"[FP16] ✅ 模型已导出: {output_path}")
    return output_path


def convert_int8(rknn, onnx_path, calib_txt, output_dir, height, width, name_prefix="fastdepth"):
    """INT8 PTQ 量化模型转换（calib_txt 为标定图片列表文件路径）"""
    output_path = os.path.join(output_dir, f"{name_prefix}_{height}x{width}_int8.rknn")
    print("\n" + "=" * 60)
    print("[INT8 PTQ] 开始转换...")
    print("=" * 60)

    # 配置量化参数
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform='rk3588',
        quantized_dtype='asymmetric_quantized-8',  # INT8 非对称量化
        # 可选：混合量化参数
        # quantized_algorithm='normal',  # 'normal' | 'mmse' | 'kl_divergence'
    )

    # 加载 ONNX
    print(f"[INT8] 加载 ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError(f"load_onnx 失败, 返回码: {ret}")

    # 构建（量化）
    # 统计标定图片数量
    with open(calib_txt, 'r') as f:
        num_lines = sum(1 for _ in f)
    print(f"[INT8] 构建量化 RKNN 模型（使用 {num_lines} 张标定图像）...")
    ret = rknn.build(do_quantization=True, dataset=calib_txt)
    if ret != 0:
        raise RuntimeError(f"build 量化失败, 返回码: {ret}")

    # 导出
    ret = rknn.export_rknn(export_path=output_path)
    if ret != 0:
        raise RuntimeError(f"export_rknn 失败, 返回码: {ret}")

    print(f"[INT8] ✅ 模型已导出: {output_path}")
    return output_path


def verify_inference(rknn, output_dir, target, device_id, height, width):
    """使用模拟器/连板进行推理验证"""
    print("\n" + "=" * 60)
    print(f"[验证] 初始化运行时环境 (target={target or 'simulator'})")
    print("=" * 60)

    ret = rknn.init_runtime(
        target=target,
        device_id=device_id,
        perf_debug=False,
    )
    if ret != 0:
        print(f"[警告] init_runtime 失败 (ret={ret})，跳过推理验证")
        return

    # 构造随机输入进行推理测试
    dummy_input = np.random.randn(1, height, width, INPUT_CHANNELS).astype(np.float32)

    # NHWC → 适配 RKNN 的 layout
    print("[验证] 执行推理...")
    outputs = rknn.inference(inputs=[dummy_input])

    if outputs is not None:
        for i, out in enumerate(outputs):
            print(f"  输出 {i}: shape={out.shape}, dtype={out.dtype}, "
                  f"min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
        print("[验证] ✅ 推理成功")
    else:
        print("[验证] ⚠️ 推理返回 None")


def main():
    args = parse_args()

    # 延迟导入：rknn 只在安装了 RKNN Toolkit2 的环境中才可用
    try:
        from rknn.api import RKNN
    except ImportError:
        print("=" * 60)
        print("错误：未检测到 RKNN Toolkit2！")
        print("请先安装 RKNN Toolkit2：")
        print("  pip install rknn_toolkit2-*-cp38-cp38-linux_x86_64.whl")
        print("=" * 60)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 准备 Calibration 数据 ---
    if not args.skip_int8:
        calib_images, calib_files = prepare_calibration_images(
            args.dataset, args.num_calib, args.height, args.width
        )
        calib_txt = create_calibration_txt(
            calib_files, os.path.abspath(args.dataset),
            os.path.join(args.output_dir, "calibration_list.txt")
        )
    else:
        calib_images = None

    # --- FP16 转换 ---
    if not args.skip_fp16:
        rknn_fp16 = RKNN(verbose=args.verbose, verbose_file='./fp16_convert.log')
        try:
            fp16_path = convert_fp16(rknn_fp16, args.onnx, args.output_dir, args.height, args.width, args.name_prefix)
            # 模拟器推理验证
            verify_inference(rknn_fp16, args.output_dir, None, None, args.height, args.width)
        finally:
            rknn_fp16.release()

    # --- INT8 转换 ---
    if not args.skip_int8:
        rknn_int8 = RKNN(verbose=args.verbose, verbose_file='./int8_convert.log')
        try:
            int8_path = convert_int8(rknn_int8, args.onnx, calib_txt, args.output_dir, args.height, args.width, args.name_prefix)
            # 模拟器推理验证
            verify_inference(rknn_int8, args.output_dir, None, None, args.height, args.width)
        finally:
            rknn_int8.release()

    # --- 连板验证（如果指定了 target） ---
    if args.target == 'rk3588':
        print("\n" + "=" * 60)
        print("[连板验证] 在 RK3588 开发板上执行推理...")
        print("=" * 60)

        h, w = args.height, args.width
        for model_type, model_path in [
            ("FP16", os.path.join(args.output_dir, f"{args.name_prefix}_{h}x{w}_fp16.rknn")),
            ("INT8", os.path.join(args.output_dir, f"{args.name_prefix}_{h}x{w}_int8.rknn")),
        ]:
            if not os.path.exists(model_path):
                print(f"  跳过 {model_type}: 文件不存在")
                continue

            rknn_dev = RKNN(verbose=args.verbose)
            try:
                ret = rknn_dev.load_rknn(path=model_path)
                if ret != 0:
                    print(f"  {model_type}: 加载失败")
                    continue
                verify_inference(rknn_dev, args.output_dir, args.target,
                                 args.device_id, args.height, args.width)
            finally:
                rknn_dev.release()

    print("\n" + "=" * 60)
    print("全部转换完成！")
    print("=" * 60)
    print(f"输出目录: {os.path.abspath(args.output_dir)}")
    for f in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
