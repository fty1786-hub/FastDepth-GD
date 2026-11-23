import csv
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
from Data import createDataLoaders, UseGeoDataset # 确保 Data.py 中有这两个定义

import criteria
import models
from metrics import AverageMeter, Result
from utils import *

# 基础设置
cudnn.benchmark = True

# 全局变量定义
fieldnames = ['rmse', 'mae', 'edge_mae', 'delta1', 'absrel', 'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()


def _load_training_checkpoint(path, map_location=None):
    """Full checkpoint contains argparse.Namespace; PyTorch>=2.6 needs weights_only=False."""
    kwargs = {}
    if map_location is not None:
        kwargs['map_location'] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _tensor_stats(name, tensor):
    finite_mask = torch.isfinite(tensor)
    valid_mask = finite_mask & (tensor > 0)
    finite_ratio = finite_mask.float().mean().item()
    valid_ratio = valid_mask.float().mean().item()

    if finite_mask.any():
        finite_vals = tensor[finite_mask]
        t_min = finite_vals.min().item()
        t_max = finite_vals.max().item()
        t_mean = finite_vals.mean().item()
        t_std = finite_vals.std().item()
    else:
        t_min, t_max, t_mean, t_std = float('nan'), float('nan'), float('nan'), float('nan')

    print(
        f"      {name}: min={t_min:.6f}, max={t_max:.6f}, mean={t_mean:.6f}, std={t_std:.6f}, "
        f"finite_ratio={finite_ratio:.4f}, valid_ratio(>0)={valid_ratio:.4f}"
    )


def _edge_mae(pred, target_gt, quantile=0.85):
    valid_mask = (target_gt > 0) & torch.isfinite(target_gt) & torch.isfinite(pred)
    if valid_mask.sum() == 0:
        return 0.0

    grad_x = target_gt[:, :, :, 1:] - target_gt[:, :, :, :-1]
    grad_y = target_gt[:, :, 1:, :] - target_gt[:, :, :-1, :]
    grad_x = torch.nn.functional.pad(grad_x.abs(), (0, 1, 0, 0))
    grad_y = torch.nn.functional.pad(grad_y.abs(), (0, 0, 0, 1))
    grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12)

    valid_grad = grad_mag[valid_mask]
    if valid_grad.numel() == 0:
        return 0.0
    th = torch.quantile(valid_grad, quantile)
    edge_mask = valid_mask & (grad_mag >= th)
    if edge_mask.sum() == 0:
        return 0.0
    return float((pred[edge_mask] - target_gt[edge_mask]).abs().mean())


def _distill_scale(epoch, args):
    if args.distill_warmup_epochs <= 0:
        return 1.0
    if epoch < args.distill_warmup_epochs:
        return float(epoch + 1) / float(args.distill_warmup_epochs)
    return 1.0

def debug_depth_stats(train_loader, val_loader, model, num_batches=2):
    print("\n=> [DEBUG] Depth stats check started...")
    model.eval()

    with torch.no_grad():
        for split_name, loader in [("train", train_loader), ("val", val_loader)]:
            print(f"=> [DEBUG] split={split_name}, max_batches={num_batches}")
            for i, data in enumerate(loader):
                if i >= num_batches:
                    break

                if len(data) == 3:
                    input_img, target_gt, target_teacher = data
                else:
                    input_img, target_gt = data
                    target_teacher = None

                input_img = input_img.cuda(non_blocking=True)
                target_gt = target_gt.cuda(non_blocking=True)
                pred = model(input_img)

                print(f"   - batch {i + 1}:")
                _tensor_stats("target_gt", target_gt)
                if target_teacher is not None:
                    _tensor_stats("target_teacher", target_teacher.cuda(non_blocking=True))
                _tensor_stats("pred_raw", pred)

    print("=> [DEBUG] Depth stats check finished.\n")

def main():
    global args, best_result, output_directory, train_csv, test_csv

    # 1. 解析参数
    args = parse_command()
    print(f"==> 启动参数: {args}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 2. 准备输出目录
    output_directory = get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    # 3. 创建数据加载器
    print("=> 正在初始化 UseGeo 数据加载器...")
    train_loader, val_loader = createDataLoaders(args)

    # 4. 创建模型
    print(f"=> 正在创建模型: {args.arch} ...")
    # UseGeoDataset resize 由命令行参数控制
    model = models.MobileNet(
        decoder=args.decoder,
        output_size=(args.input_height, args.input_width),
        pretrained=args.pretrained,
        use_ca=not args.disable_ca,
        use_guided_upsample=args.use_guided_upsample
    )
    model = model.cuda()
    print("=> 模型已加载至 GPU.")

    if args.debug_depth_stats:
        debug_depth_stats(train_loader, val_loader, model, num_batches=args.debug_depth_batches)

    # 5. 损失函数与优化器 (修改为蒸馏损失)
    print("=> 正在初始化蒸馏损失函数 (DistillationLoss)...")
    # 注意：需确保你的 criteria.py 中已经写好了 DistillationLoss 类
    criterion = criteria.DistillationLoss(
        alpha=1.0,
        gamma=args.distill_gamma,
        edge_lambda=args.edge_lambda,
        edge_weight=args.edge_weight,
        silog_lambda=args.silog_lambda,
        aux_lambda=args.aux_lambda,
        distill_multiscale_lambda=args.distill_multiscale_lambda,
        grad_gt_lambda=args.grad_gt_lambda,
        ssim_lambda=args.ssim_lambda,
        teacher_edge_lambda=args.teacher_edge_lambda,
        distill_zscore_lambda=args.distill_zscore_lambda,
        distill_ssim_lambda=args.distill_ssim_lambda,
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 6. 处理断点续训 (Resume)
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> 加载断点: {args.resume}")
            checkpoint = _load_training_checkpoint(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_result = checkpoint['best_result']
            model.load_state_dict(checkpoint['model'].state_dict(), strict=False)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            except ValueError:
                print("=> optimizer 参数组不匹配（模型结构已变更），跳过 optimizer 状态恢复")
            print(f"=> 已加载至 Epoch {checkpoint['epoch']}")
        else:
            print(f"=> 找不到断点文件: {args.resume}")

    # 7. 初始化 CSV 文件
    if not args.resume:
        for csv_path in [train_csv, test_csv]:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    # 8. 训练/评估循环
    if args.evaluate:
        print("=> 进入评估模式...")
        validate(val_loader, model, start_epoch, write_to_file=False)
        return

    no_improve_epochs = 0
    print("=> 开始训练循环...")
    for epoch in range(start_epoch, args.epochs):
        if (
            args.stage2_start_epoch > 0
            and epoch == args.stage2_start_epoch
            and args.finetune_height > 0
            and args.finetune_width > 0
        ):
            pixel_ratio = (args.finetune_height * args.finetune_width) / (
                args.input_height * args.input_width
            )
            old_bs = args.batch_size
            new_bs = max(1, int(round(old_bs / pixel_ratio)))
            print(
                f"=> 进入二阶段高分辨率微调: "
                f"{args.input_height}x{args.input_width} -> {args.finetune_height}x{args.finetune_width}"
            )
            print(f"=> batch_size 自动缩放: {old_bs} -> {new_bs} (像素比={pixel_ratio:.2f}x)")
            args.input_height = args.finetune_height
            args.input_width = args.finetune_width
            args.batch_size = new_bs
            train_loader, val_loader = createDataLoaders(args)

        adjust_learning_rate(optimizer, epoch, args)
        print(f"=> epoch {epoch} lr={optimizer.param_groups[0]['lr']:.2e}")

        # 训练一个 Epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # 验证并获取指标
        result, img_merge = validate(val_loader, model, epoch)

        # 保存最优模型
        improvement = best_result.rmse - result.rmse
        is_best = improvement > args.early_stop_min_delta
        if is_best:
            best_result = result
            no_improve_epochs = 0
            with open(best_txt, 'w') as txtfile:
                txtfile.write(f"epoch={epoch}\nmse={result.mse:.3f}\nrmse={result.rmse:.3f}\n"
                              f"absrel={result.absrel:.3f}\nmae={result.mae:.3f}\n"
                              f"delta1={result.delta1:.3f}\nt_gpu={result.gpu_time:.4f}\n")
            if img_merge is not None:
                save_image(img_merge, os.path.join(output_directory, 'comparison_best.png'))
        else:
            no_improve_epochs += 1

        # 保存 Checkpoint
        save_checkpoint({
            'args': args,
            'epoch': epoch,
            'model': model,
            'best_result': best_result,
            'optimizer': optimizer,
        }, is_best, epoch, output_directory)

        if epoch >= args.early_stop_warmup and no_improve_epochs >= args.early_stop_patience:
            print(
                f"=> 早停触发: 连续 {no_improve_epochs} 个 epoch 未达到 "
                f"RMSE 改善阈值 {args.early_stop_min_delta:.6f}。"
            )
            print(f"=> 提前结束训练于 epoch {epoch}，当前 best_rmse={best_result.rmse:.6f}")
            break

    if args.restore_best_at_end:
        best_ckpt = os.path.join(output_directory, 'model_best.pth.tar')
        if os.path.isfile(best_ckpt):
            ckpt = _load_training_checkpoint(best_ckpt, map_location='cuda')
            model.load_state_dict(ckpt['model'].state_dict())
            print(f"=> 已恢复最优权重: {best_ckpt} (用于后续评估或导出)")
        else:
            print(f"=> 未找到 {best_ckpt}，跳过恢复最优权重")

def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    model.train()
    end = time.time()

    # ⬇️ 仔细核对：下面所有的代码都必须在 for 循环的缩进内 ⬇️
    for i, data in enumerate(train_loader):
        # 1. 安全解包
        if len(data) == 3:
            input, target_gt, target_teacher = data
            target_teacher = target_teacher.cuda()
        else:
            input, target_gt = data
            target_teacher = None

        input, target_gt = input.cuda(), target_gt.cuda()
        data_time = time.time() - end

        # 2. 前向传播
        pred_main, pred_aux = model(input, return_aux=True)
        pred = (pred_main, pred_aux) if pred_aux is not None else pred_main

        # 3. 计算 Loss
        distill_scale = _distill_scale(epoch, args)
        if isinstance(criterion, criteria.DistillationLoss) and target_teacher is not None:
            loss = criterion(pred, target_gt, target_teacher, distill_scale=distill_scale)
        else:
            loss = criterion(pred_main, target_gt)

        # 4. 反向传播更新权重
        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        gpu_time = time.time() - (end + data_time)

        # 5. 记录指标
        result = Result()
        result.evaluate(pred_main.data, target_gt.data, args.depth_scale)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # 6. 打印训练日志
        if (i + 1) % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i+1}/{len(train_loader)}]\t'
                  f'RMSE={result.rmse:.3f}({average_meter.average().rmse:.3f})\t'
                  f'Loss={loss.detach().item():.4f}')
    # ⬆️ for 循环结束 ⬆️

    # 写入训练日志 (这部分在 for 循环外面)
    avg = average_meter.average()
    with open(train_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                         'mae': avg.mae, 'edge_mae': avg.edge_mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                         'gpu_time': avg.gpu_time, 'data_time': avg.data_time})
def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval()
    img_merge = None

    # 准备评估 CSV
    eval_dir = os.path.join(output_directory, 'FastDepth')
    os.makedirs(eval_dir, exist_ok=True)
    eval_file = os.path.join(eval_dir, 'evaluation.csv')

    with open(eval_file, "a") as f:
        if epoch == 0:
            f.write("Frame,RMSE,MAE,EDGE_MAE,GPU_TIME\n")

        # 【修改点：使用 _ 接收占位的 Teacher 标签】
        for i, data in enumerate(val_loader):
            if len(data) == 3:
                input, target_gt, _ = data
            else:
                input, target_gt = data
            input, target_gt = input.cuda(), target_gt.cuda()
            torch.cuda.synchronize()

            start_time = time.time()
            with torch.no_grad():
                pred = model(input)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time

            result = Result()
            # 同样仅使用真实 GT 进行评估
            result.evaluate(pred.data, target_gt.data, args.depth_scale)
            result.edge_mae = _edge_mae(pred.data, target_gt.data) * args.depth_scale
            average_meter.update(result, gpu_time, 0.0, input.size(0))

            f.write(f'{i+1},{result.rmse:.3f},{result.mae:.3f},{result.edge_mae:.3f},{gpu_time:.4f}\n')

            # 图像可视化逻辑 (每隔 skip 张保存一次对比图，改为 target_gt)
            skip = 50
            if i == 0:
                img_merge = merge_into_row_with_gt(input, target_gt, pred, (target_gt - pred).abs())
            elif i % skip == 0 and i < 8 * skip:
                row = merge_into_row_with_gt(input, target_gt, pred, (target_gt - pred).abs())
                img_merge = add_row(img_merge, row)

    if img_merge is not None:
        save_image(img_merge, os.path.join(output_directory, f'comparison_{epoch}.png'))

    avg = average_meter.average()
    print(
        f'\n[验证 Epoch {epoch}] RMSE={avg.rmse:.3f} | MAE={avg.mae:.3f} '
        f'| EDGE_MAE={avg.edge_mae:.3f} | t_GPU={avg.gpu_time:.4f}'
    )

    if write_to_file:
        with open(test_csv, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                             'mae': avg.mae, 'edge_mae': avg.edge_mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                             'data_time': 0, 'gpu_time': avg.gpu_time})
    return avg, img_merge

if __name__ == '__main__':
    main()