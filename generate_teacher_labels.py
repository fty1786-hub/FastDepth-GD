import argparse
import os
import torch
from PIL import Image
import numpy as np
import tifffile as tiff
from tqdm import tqdm


def align_scale_only(teacher, gt):
    """
    Scale-only alignment: teacher * scale → best match to GT.

    Uses RATIO OF MEDIANS on valid pixels (robust, preserves relative structure).
    NO shift term — shift destroys the depth gradient relationships that
    ZoeDepth learned, especially in near-field regions where a constant
    offset can invert local depth ordering.

    Why scale-only:
      - teacher * scale preserves: relative ordering, gradient direction,
        and gradient ratio (grad(aligned) = scale * grad(teacher))
      - teacher * scale + shift destroys: zero-crossing structure,
        near-field gradient sign, and inter-pixel relative differences

    This is critical for distillation: we want the student to learn
    ZoeDepth's structural knowledge, not just match absolute values.
    """
    mask = gt > 0
    if mask.sum() < 10:
        # 有效像素太少，退化为全局均值比
        t_mean = teacher[mask].mean() if mask.sum() > 0 else teacher.mean()
        g_mean = gt[mask].mean() if mask.sum() > 0 else gt.mean()
        scale = g_mean / (t_mean + 1e-8)
        return teacher * scale

    t = teacher[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)

    # Robust scale: ratio of medians (less sensitive to outliers than mean)
    t_median = np.median(t)
    g_median = np.median(g)
    if t_median > 1e-6:
        scale_median = g_median / t_median
    else:
        scale_median = 0.0

    # Also compute ratio of means as a sanity check
    t_mean = t.mean()
    g_mean = g.mean()
    scale_mean = g_mean / (t_mean + 1e-8) if t_mean > 1e-6 else 0.0

    # Use median-based scale (more robust), fall back to mean if absurd
    if 0.01 < scale_median < 100.0:
        scale = scale_median
    elif 0.01 < scale_mean < 100.0:
        scale = scale_mean
    else:
        # Both unreasonable — clamp to safe range
        scale = np.clip(scale_mean, 0.01, 100.0)

    # Ensure scale is positive (ZoeDepth should always be same direction as GT)
    if scale <= 0:
        scale = g_mean / (t_mean + 1e-8)

    return teacher * scale


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-image scale-calibrated ZoeDepth-NK labels for training images."
    )
    parser.add_argument("--rgb-dir", default="data/usegeo/train/rgb")
    parser.add_argument("--depth-dir", default="data/usegeo/train/depth")
    parser.add_argument("--output-dir", default="data/teacher_depth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    model.eval().to(device)

    rgb_dir = args.rgb_dir
    gt_dir = args.depth_dir
    teacher_depth_dir = args.output_dir
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"Training RGB directory does not exist: {rgb_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"Training ground-truth directory does not exist: {gt_dir}")
    os.makedirs(teacher_depth_dir, exist_ok=True)

    image_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f"共 {len(image_files)} 张图，使用 scale-only 对齐保留相对深度结构...")

    with torch.no_grad():
        for img_name in tqdm(image_files):
            img_path = os.path.join(rgb_dir, img_name)
            img = Image.open(img_path).convert("RGB")

            # ZoeDepth 推理，输出单位为米
            depth_pred = model.infer_pil(img)

            # 读取对应 GT（单位与保存时一致，这里是原始 float32 tiff）
            gt_name = img_name.split('.')[0] + '_depth_res.tiff'
            gt_path = os.path.join(gt_dir, gt_name)
            if not os.path.exists(gt_path):
                # 尝试其他命名规则
                gt_name = img_name.rsplit('_res', 1)[0] + '_depth_res.tiff'
                gt_path = os.path.join(gt_dir, gt_name)

            if os.path.exists(gt_path):
                gt = tiff.imread(gt_path).astype(np.float32)
                # GT 和 teacher 分辨率可能不同，先 resize teacher 到 GT 尺寸做对齐
                if gt.shape != depth_pred.shape:
                    from PIL import Image as PILImage
                    pred_pil = PILImage.fromarray(depth_pred)
                    pred_resized = np.array(pred_pil.resize(
                        (gt.shape[1], gt.shape[0]), PILImage.BILINEAR
                    ))
                else:
                    pred_resized = depth_pred

                aligned = align_scale_only(pred_resized, gt)
            else:
                raise FileNotFoundError(
                    f"Missing training ground-truth pair for {img_name}: {gt_path}"
                )

            aligned = np.clip(aligned, 0.1, 550.0).astype(np.float32)

            out_name = img_name.split('.')[0] + '.tiff'
            out_path = os.path.join(teacher_depth_dir, out_name)
            tiff.imwrite(out_path, aligned)


if __name__ == '__main__':
    main()
