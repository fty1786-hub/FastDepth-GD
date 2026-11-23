import os
import time
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import cv2

# --- 处理 Data.py 结构重构后的导入问题 ---
try:
    from Data import CustomDataLoader
except ImportError:
    class CustomDataLoader:
        modality_names = ['rgb']
# -----------------------------------------------

cmap = plt.cm.viridis

def rgb2grayscale(rgb):
    return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114

class DenseToSparse:
    def __init__(self):
        pass
    def dense_to_sparse(self, rgb, depth):
        pass

class UniformSampling(DenseToSparse):
    name = "uar"
    def __init__(self, num_samples, max_depth=np.inf):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth

    def dense_to_sparse(self, rgb, depth):
        mask_keep = depth > 0
        if self.max_depth is not np.inf:
            mask_keep = np.bitwise_and(mask_keep, depth <= self.max_depth)
        n_keep = np.count_nonzero(mask_keep)
        if n_keep == 0:
            return mask_keep
        else:
            prob = float(self.num_samples) / n_keep
            return np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)

class SimulatedStereo(DenseToSparse):
    name = "sim_stereo"
    def __init__(self, num_samples, max_depth=np.inf, dilate_kernel=3, dilate_iterations=1):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.dilate_kernel = dilate_kernel
        self.dilate_iterations = dilate_iterations

    def dense_to_sparse(self, rgb, depth):
        gray = rgb2grayscale(rgb)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        depth_mask = np.bitwise_and(depth != 0.0, depth <= self.max_depth)
        edge_fraction = float(self.num_samples) / np.size(depth)
        mag = cv2.magnitude(gx, gy)
        min_mag = np.percentile(mag[depth_mask], 100 * (1.0 - edge_fraction))
        mag_mask = mag >= min_mag
        if self.dilate_iterations >= 0:
            kernel = np.ones((self.dilate_kernel, self.dilate_kernel), dtype=np.uint8)
            cv2.dilate(mag_mask.astype(np.uint8), kernel, iterations=self.dilate_iterations)
        mask = np.bitwise_and(mag_mask, depth_mask)
        return mask

def parse_command():
    model_names = ['resnet18', 'resnet50', 'MobileNet']
    loss_names = ['l1', 'l2']
    sparsifier_names = [x.name for x in [UniformSampling, SimulatedStereo]]

    from models import Decoder
    decoder_names = Decoder.names

    # --- 修改 1：将 usegeo 添加到合法数据集列表中 ---
    data_names = ['nyudepthv2', 'usegeo']
    modality_names = CustomDataLoader.modality_names

    import argparse
    parser = argparse.ArgumentParser(description='FastDepth')

    # 修改默认值为 usegeo
    parser.add_argument('--data', metavar='DATA', default='usegeo',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: usegeo)')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
    parser.add_argument('-t', '--train', default='', type=str)
    parser.add_argument('--gpu', default='0', type=str, metavar='N', help="gpu id")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='MobileNet', choices=model_names)
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names)
    parser.add_argument('-s', '--num-samples', default=0, type=int)
    parser.add_argument('--max-depth', default=-1.0, type=float)
    parser.add_argument('--sparsifier', metavar='SPARSIFIER', default=UniformSampling.name, choices=sparsifier_names)
    parser.add_argument('--decoder', '-d', metavar='DECODER', default='fpnlite', choices=decoder_names)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names)
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--input-height', default=512, type=int,
                        help='input image height for train/val resize')
    parser.add_argument('--input-width', default=768, type=int,
                        help='input image width for train/val resize')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)
    parser.add_argument('--lr-step-size', default=20, type=int,
                        help='epochs between lr decays')
    parser.add_argument('--lr-gamma', default=0.5, type=float,
                        help='multiplicative factor of learning rate decay')
    parser.add_argument('--min-lr', default=1e-6, type=float,
                        help='minimum learning rate after decay')
    parser.add_argument('--lr-scheduler', default='cosine', choices=['step', 'cosine', 'none'],
                        help='step: decay every lr-step-size; cosine: smooth decay to eta-min; none: fixed lr')
    parser.add_argument('--cosine-tmax', default=0, type=int,
                        help='cosine period in epochs; 0 means use --epochs')
    parser.add_argument('--cosine-eta-min', default=1e-6, type=float,
                        help='minimum lr for cosine scheduler')
    parser.add_argument('--max-grad-norm', default=1.0, type=float,
                        help='clip global grad norm; 0 disables')
    parser.add_argument('--restore-best-at-end', action='store_true',
                        help='after training, load weights from model_best.pth.tar (in-memory model)')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
    parser.add_argument('--print-freq', '-p', default=10, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--depth-scale', default=100.0, type=float,
                        help='divide GT depth values by this factor (e.g. 100 for cm->m); metrics reported in original scale')
    parser.add_argument('--teacher-scale', default=100.0, type=float,
                        help='divide teacher depth values by this factor')
    parser.add_argument('--teacher-path', default=r"data\teacher_depth", type=str,
                        help='path to teacher depth labels')
    parser.add_argument('--debug-depth-stats', action='store_true',
                        help='print depth stats for a few batches before training')
    parser.add_argument('--debug-depth-batches', default=2, type=int,
                        help='number of batches used by --debug-depth-stats')
    parser.add_argument('--distill-gamma', default=0.01, type=float,
                        help='teacher gradient loss weight in distillation')
    parser.add_argument('--distill-multiscale-lambda', default=0.05, type=float,
                        help='teacher multi-scale distillation weight')
    parser.add_argument('--distill-warmup-epochs', default=3, type=int,
                        help='linearly ramp distillation loss in first N epochs')
    parser.add_argument('--edge-lambda', default=0.2, type=float,
                        help='edge-aware GT loss weight')
    parser.add_argument('--edge-weight', default=4.0, type=float,
                        help='edge emphasis factor in edge-aware GT loss')
    parser.add_argument('--silog-lambda', default=0.1, type=float,
                        help='scale-invariant logarithmic depth loss weight')
    parser.add_argument('--aux-lambda', default=0.2, type=float,
                        help='auxiliary decoder output supervision weight')
    parser.add_argument('--grad-gt-lambda', default=0.0, type=float,
                        help='GT gradient consistency loss weight (0 disables)')
    parser.add_argument('--ssim-lambda', default=0.0, type=float,
                        help='SSIM structural loss weight for edge sharpness (0 disables)')
    parser.add_argument('--teacher-edge-lambda', default=0.1, type=float,
                        help='teacher edge guidance loss weight (0 disables)')
    parser.add_argument('--distill-zscore-lambda', default=0.05, type=float,
                        help='per-image z-score normalized teacher L1 distillation weight (0 disables)')
    parser.add_argument('--distill-ssim-lambda', default=0.05, type=float,
                        help='SSIM-based teacher structure distillation weight (0 disables)')
    parser.add_argument('--finetune-height', default=0, type=int,
                        help='stage-2 finetune resize height, 0 disables two-stage training')
    parser.add_argument('--finetune-width', default=0, type=int,
                        help='stage-2 finetune resize width, 0 disables two-stage training')
    parser.add_argument('--stage2-start-epoch', default=0, type=int,
                        help='epoch index to switch to stage-2 finetune resolution')
    parser.add_argument('--disable-ca', action='store_true',
                        help='disable coordinate attention block')
    parser.add_argument('--ablation-tag', default='', type=str,
                        help='extra tag appended to output directory for ablation studies (e.g. nodistill, nossim, noga)')
    parser.add_argument('--use-guided-upsample', action='store_true',
                        help='replace blind bilinear upsampling with RGB-guided upsampling for sharper depth edges')
    parser.add_argument('--early-stop-patience', default=8, type=int,
                        help='stop if val rmse does not improve for N epochs')
    parser.add_argument('--early-stop-min-delta', default=1e-4, type=float,
                        help='minimum val rmse improvement to reset early-stop counter')
    parser.add_argument('--early-stop-warmup', default=5, type=int,
                        help='do not early-stop during first N epochs')
    parser.add_argument('--no-pretrain', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=True)

    args = parser.parse_args()

    # 处理 rgb 模式下的强制约束
    if args.modality == 'rgb':
        args.num_samples = 0
        if args.max_depth != 0.0:
            args.max_depth = 0.0
    if args.depth_scale <= 0:
        parser.error('--depth-scale must be > 0')
    if args.input_height <= 0 or args.input_width <= 0:
        parser.error('--input-height and --input-width must be > 0')
    if args.teacher_scale <= 0:
        parser.error('--teacher-scale must be > 0')
    if args.debug_depth_batches <= 0:
        parser.error('--debug-depth-batches must be > 0')
    if args.lr_step_size <= 0:
        parser.error('--lr-step-size must be > 0')
    if not (0 < args.lr_gamma <= 1):
        parser.error('--lr-gamma must be in (0, 1]')
    if args.min_lr <= 0:
        parser.error('--min-lr must be > 0')
    if args.distill_gamma < 0:
        parser.error('--distill-gamma must be >= 0')
    if args.distill_multiscale_lambda < 0:
        parser.error('--distill-multiscale-lambda must be >= 0')
    if args.distill_warmup_epochs < 0:
        parser.error('--distill-warmup-epochs must be >= 0')
    if args.edge_lambda < 0:
        parser.error('--edge-lambda must be >= 0')
    if args.edge_weight < 0:
        parser.error('--edge-weight must be >= 0')
    if args.silog_lambda < 0:
        parser.error('--silog-lambda must be >= 0')
    if args.aux_lambda < 0:
        parser.error('--aux-lambda must be >= 0')
    if args.finetune_height < 0 or args.finetune_width < 0:
        parser.error('--finetune-height and --finetune-width must be >= 0')
    if args.stage2_start_epoch < 0:
        parser.error('--stage2-start-epoch must be >= 0')
    if args.early_stop_patience < 1:
        parser.error('--early-stop-patience must be >= 1')
    if args.early_stop_min_delta < 0:
        parser.error('--early-stop-min-delta must be >= 0')
    if args.early_stop_warmup < 0:
        parser.error('--early-stop-warmup must be >= 0')
    if args.cosine_eta_min < 0:
        parser.error('--cosine-eta-min must be >= 0')
    if args.max_grad_norm < 0:
        parser.error('--max-grad-norm must be >= 0')
    if args.teacher_edge_lambda < 0:
        parser.error('--teacher-edge-lambda must be >= 0')
    if args.distill_zscore_lambda < 0:
        parser.error('--distill-zscore-lambda must be >= 0')
    if args.distill_ssim_lambda < 0:
        parser.error('--distill-ssim-lambda must be >= 0')
    return args

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, f'checkpoint-{epoch}.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)

    # 自动清理旧的 checkpoint 节省空间
    # Windows 下文件可能被防病毒/索引服务短暂锁定，重试 3 次
    if epoch > 0:
        prev_checkpoint = os.path.join(output_directory, f'checkpoint-{epoch-1}.pth.tar')
        if os.path.exists(prev_checkpoint):
            for attempt in range(3):
                try:
                    os.remove(prev_checkpoint)
                    break
                except PermissionError:
                    if attempt < 2:
                        time.sleep(0.5)
                    else:
                        print(f"  [WARN] 无法删除旧 checkpoint: {prev_checkpoint}，跳过")

def adjust_learning_rate(optimizer, epoch, args):
    """Set learning rate from args.lr_scheduler."""
    if args.lr_scheduler == 'none':
        lr = args.lr
    elif args.lr_scheduler == 'cosine':
        tmax = args.cosine_tmax if args.cosine_tmax > 0 else max(args.epochs, 1)
        eta_min = args.cosine_eta_min
        lr_init = args.lr
        lr = eta_min + (lr_init - eta_min) * 0.5 * (1.0 + math.cos(math.pi * epoch / tmax))
        lr = max(eta_min, lr)
    else:
        lr = max(args.min_lr, args.lr * (args.lr_gamma ** (epoch // args.lr_step_size)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_output_directory(args):
    # 简化的输出目录命名，方便管理
    ca_tag = "noca" if getattr(args, 'disable_ca', False) else "ca"
    guided_tag = "guided" if getattr(args, 'use_guided_upsample', False) else "bilinear"
    base_name = f"{args.data}.{args.arch}.{args.decoder}.{args.input_height}x{args.input_width}.lr={args.lr}.bs={args.batch_size}.{ca_tag}.{guided_tag}"
    ablation_tag = getattr(args, 'ablation_tag', '')
    if ablation_tag:
        base_name = base_name + "." + ablation_tag
    output_directory = os.path.join('results', base_name)
    return output_directory

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None or d_max is None:
        d_min = np.min(depth)
        d_max = np.max(depth)

    # 避免除以 0
    if d_max - d_min == 0:
        depth_relative = np.zeros_like(depth)
    else:
        depth_relative = (depth - d_min) / (d_max - d_min)

    return (255 * cmap(depth_relative)[:,:,:3]).astype('uint8')

def merge_into_row(input, depth_target, depth_pred):
    # 处理输入图像 (C, H, W) -> (H, W, C)
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0))
    rgb = rgb.astype('uint8')

    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))

    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    return np.hstack([rgb, depth_target_col, depth_pred_col])

def merge_into_row_with_gt(input, depth_target, depth_pred, depth_diff):
    # 注意：根据 main.py 中的调用，这里接收 4 个参数
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0))
    rgb = rgb.astype('uint8')

    dt = np.squeeze(depth_target.cpu().numpy())
    dp = np.squeeze(depth_pred.data.cpu().numpy())
    dd = np.squeeze(depth_diff.cpu().numpy())

    # 统一色标
    d_min, d_max = dt.min(), dt.max()

    col_target = colored_depthmap(dt, d_min, d_max)
    col_pred = colored_depthmap(dp, d_min, d_max)
    col_diff = colored_depthmap(dd, dd.min(), dd.max()) # 误差图使用独立色标

    return np.hstack([rgb, col_target, col_pred, col_diff])

def add_row(img_merge, row):
    return np.vstack([img_merge, row])

def save_image(img_merge, filename):
    img = Image.fromarray(img_merge)
    img.save(filename)