import torch
import torch.nn as nn
import torch.nn.functional as F


def _gradient_xy(x):
    """Compute spatial gradients via finite difference (replicate padding)."""
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    r = F.pad(x, (0, 1, 0, 1), mode='replicate')
    g_x = r[:, :, :h_x, 1:] - r[:, :, :h_x, :w_x]
    g_y = r[:, :, 1:, :w_x] - r[:, :, :h_x, :w_x]
    return g_x, g_y


def _gradient_magnitude(x):
    """Compute gradient magnitude map (B,1,H,W)."""
    gx, gy = _gradient_xy(x)
    return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)


# ═══════════════════════════════════════════════════════════════
#  GT-supervised losses (absolute depth supervision)
# ═══════════════════════════════════════════════════════════════

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


def silog_loss(pred, target, lam=0.85):
    """Scale-invariant log loss (Eigen et al.). Already robust to global scale."""
    valid_mask = (target > 0).detach()
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    p = pred[valid_mask].clamp(min=1e-6)
    t = target[valid_mask].clamp(min=1e-6)
    d = torch.log(p) - torch.log(t)
    return torch.sqrt(torch.mean(d * d) - lam * torch.mean(d) * torch.mean(d) + 1e-12)


def _match_spatial(tensor, ref):
    if tensor.shape[-2:] == ref.shape[-2:]:
        return tensor
    return F.interpolate(tensor, size=ref.shape[-2:], mode='bilinear', align_corners=False)


def ssim_loss(pred, target_gt, window_size=11):
    """
    SSIM-based loss: penalizes structural dissimilarity including edge position and width.
    SSIM's luminance normalization makes it inherently robust to global scale/shift.
    """
    valid_mask = (target_gt > 0).detach()
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2

    mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_t = F.avg_pool2d(target_gt.detach(), window_size, stride=1, padding=pad)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sig_p = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_p2
    sig_t = F.avg_pool2d(target_gt.detach() * target_gt.detach(), window_size, stride=1, padding=pad) - mu_t2
    sig_pt = F.avg_pool2d(pred * target_gt.detach(), window_size, stride=1, padding=pad) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / (
        (mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2) + 1e-12
    )
    return (1.0 - ssim_map)[valid_mask].mean()


def gradient_consistency_loss(pred, target_gt):
    """Penalizes gradient difference between pred and GT (absolute depth supervision)."""
    valid_mask = (target_gt > 0).detach()
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    grad_p_x, grad_p_y = _gradient_xy(pred)
    grad_t_x, grad_t_y = _gradient_xy(target_gt.detach())
    loss_x = torch.abs(grad_p_x - grad_t_x)
    loss_y = torch.abs(grad_p_y - grad_t_y)
    loss_map = loss_x + loss_y
    return loss_map[valid_mask].mean()


def edge_aware_gt_loss(pred, target_gt, edge_weight=4.0):
    """
    Edge-aware depth loss: larger weight on high-gradient GT regions to preserve edges.
    Uses tanh-normalized gradient magnitude for stable weight scaling.
    """
    valid_mask = (target_gt > 0).detach()
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    grad_t_x, grad_t_y = _gradient_xy(target_gt.detach())
    grad_mag = torch.sqrt(grad_t_x.pow(2) + grad_t_y.pow(2) + 1e-12)
    grad_norm = grad_mag / (grad_mag.mean(dim=(2, 3), keepdim=True) + 1e-6)
    weight_map = 1.0 + edge_weight * torch.tanh(grad_norm)

    abs_diff = torch.abs(pred - target_gt)
    weighted_diff = abs_diff * weight_map
    return weighted_diff[valid_mask].mean()


# ═══════════════════════════════════════════════════════════════
#  Teacher distillation losses — ALL scale-invariant
# ═══════════════════════════════════════════════════════════════
#
#  Problem: Each teacher image is independently aligned to GT via
#  per-image (scale, shift) least-squares. Absolute depth/gradient
#  values in the teacher are therefore NOT comparable across images.
#
#  Solution: All teacher distillation losses below operate on
#  scale-invariant quantities:
#    - Gradient DIRECTION (cosine similarity), not absolute difference
#    - Per-image NORMALIZED depths (z-scored or mean-normalized)
#    - Per-image PERCENTILE thresholds for edge detection
#
#  The only exception is multi-scale L1 which uses z-score normalization.


def imgrad_cosine_loss(pred, target_teacher):
    """
    Scale-invariant gradient distillation via cosine similarity.

    Instead of |grad_p - grad_t| (which penalizes magnitude mismatch
    caused by per-image scale/shift alignment), we compute:

        loss = 1 - cos_sim(grad_p, grad_t)

    This ONLY penalizes when pred and teacher disagree on gradient
    DIRECTION — i.e. when the edge is in the wrong place or missing.
    Gradient magnitude (how "strong" the edge is) is ignored, because
    that's determined by the per-image scale parameter.

    Computed only on pixels where teacher has valid (>0) depth.
    """
    valid_mask = (target_teacher > 0).detach()
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    grad_p_x, grad_p_y = _gradient_xy(pred * valid_mask.float())
    grad_t_x, grad_t_y = _gradient_xy(target_teacher.detach())

    # Gradient magnitude (per-pixel 2D vector norm)
    mag_p = torch.sqrt(grad_p_x.pow(2) + grad_p_y.pow(2) + 1e-12)
    mag_t = torch.sqrt(grad_t_x.pow(2) + grad_t_y.pow(2) + 1e-12)

    # Cosine similarity: dot product / (mag_p * mag_t)
    dot = grad_p_x * grad_t_x + grad_p_y * grad_t_y
    cos_sim = dot / (mag_p * mag_t + 1e-8)

    # Loss = 1 - cos_sim ∈ [0, 2], 0 when gradients are perfectly aligned
    # Clamp cos_sim to [-1, 1] for numerical stability
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    loss_map = 1.0 - cos_sim

    return loss_map[valid_mask].mean()


def teacher_edge_guidance_loss(pred, target_teacher, edge_thresh_pct=75, edge_weight=3.0):
    """
    Scale-invariant structural edge guidance.

    Strategy (unchanged from original — was already scale-invariant):
      1. Compute teacher gradient magnitude.
      2. Use per-image PERCENTILE threshold (not absolute value).
      3. Create soft edge weight map normalized to [0, edge_weight].
      4. Hinge loss: penalize pred where it lacks gradient at teacher edges.

    Key improvement: pred_grad_mag is also normalized per-image so the
    hinge threshold is meaningful regardless of absolute depth scale.
    """
    valid_mask = (target_teacher > 0).detach()
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    with torch.no_grad():
        grad_t_x, grad_t_y = _gradient_xy(target_teacher)
        teacher_grad_mag = torch.sqrt(grad_t_x.pow(2) + grad_t_y.pow(2) + 1e-12)

        # Per-image percentile threshold → scale-invariant
        flat = teacher_grad_mag.view(teacher_grad_mag.size(0), -1)
        thresh = torch.quantile(flat, edge_thresh_pct / 100.0, dim=1)
        thresh = thresh.view(-1, 1, 1, 1)
        # Soft weight: 1 at threshold, grows linearly above it
        edge_map = (teacher_grad_mag / (thresh + 1e-8)).clamp(0.0, edge_weight)
        edge_map = edge_map * valid_mask.float()

    # Compute pred gradient magnitude, ALSO normalize per-image
    grad_p_x, grad_p_y = _gradient_xy(pred)
    pred_grad_mag = torch.sqrt(grad_p_x.pow(2) + grad_p_y.pow(2) + 1e-12)

    # Normalize pred_grad_mag so its threshold is comparable across images
    pred_flat = pred_grad_mag.view(pred_grad_mag.size(0), -1)
    pred_median = torch.quantile(pred_flat, 0.5, dim=1).view(-1, 1, 1, 1)
    pred_grad_norm = pred_grad_mag / (pred_median + 1e-8)

    # Hinge: where teacher sees edges, pred should have ≥1x median gradient
    hinge = F.relu(1.0 - pred_grad_norm)
    loss_map = edge_map * hinge

    return loss_map[valid_mask].mean()


def scale_invariant_teacher_l1(pred, target_teacher):
    """
    Per-image z-score normalized L1 distillation loss.

    For each image independently:
        z = (x - mean) / std
    Then compute L1 on z-scored values.

    This eliminates the per-image (scale, shift) differences while
    preserving the RELATIVE depth ordering within each image —
    which is exactly what ZoeDepth provides and what we want to transfer.
    """
    valid_mask = (target_teacher > 0).detach()
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    B = pred.size(0)
    total_loss = 0.0

    for b in range(B):
        mask_b = valid_mask[b, 0]
        if mask_b.sum() < 10:
            continue
        p_b = pred[b, 0][mask_b]
        t_b = target_teacher[b, 0][mask_b].detach()

        # Z-score normalization per image (scale+shift invariant)
        p_mean = p_b.mean()
        p_std = p_b.std() + 1e-8
        t_mean = t_b.mean()
        t_std = t_b.std() + 1e-8

        p_norm = (p_b - p_mean) / p_std
        t_norm = (t_b - t_mean) / t_std

        total_loss += F.l1_loss(p_norm, t_norm)

    return torch.tensor(total_loss / max(B, 1), device=pred.device)


def scale_invariant_multiscale_loss(pred, target_teacher, scales=(2, 4)):
    """
    Multi-scale distillation with per-scale z-score normalization.

    At each downsampled scale, both pred and teacher are independently
    z-scored before L1 comparison. This eliminates scale/shift differences
    at every resolution level, allowing the student to learn the teacher's
    multi-scale RELATIVE depth structure.
    """
    valid_mask = (target_teacher > 0).detach()
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    total_loss = 0.0
    n_scales = 0.0

    for s in scales:
        t_s = F.avg_pool2d(target_teacher, kernel_size=s, stride=s)
        p_s = F.avg_pool2d(pred, kernel_size=s, stride=s)

        t_s_mask = (t_s > 0).detach()
        if t_s_mask.sum() < 10:
            continue

        # Z-score normalize at this scale
        p_s_valid = p_s[t_s_mask]
        t_s_valid = t_s[t_s_mask]

        p_mean = p_s_valid.mean()
        p_std = p_s_valid.std() + 1e-8
        t_mean = t_s_valid.mean()
        t_std = t_s_valid.std() + 1e-8

        p_norm = (p_s_valid - p_mean) / p_std
        t_norm = (t_s_valid - t_mean) / t_std

        total_loss += F.l1_loss(p_norm, t_norm)
        n_scales += 1.0

    if n_scales == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return total_loss / n_scales


def teacher_ssim_distill_loss(pred, target_teacher, window_size=11):
    """
    SSIM-based distillation: SSIM's internal luminance/contrast normalization
    makes it inherently scale-and-shift invariant at the local patch level.

    This transfers the teacher's local STRUCTURE (edge positions, relative
    ordering) without being affected by per-image alignment parameters.
    """
    valid_mask = (target_teacher > 0).detach()
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2

    t_detached = target_teacher.detach()
    mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_t = F.avg_pool2d(t_detached, window_size, stride=1, padding=pad)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sig_p = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_p2
    sig_t = F.avg_pool2d(t_detached * t_detached, window_size, stride=1, padding=pad) - mu_t2
    sig_pt = F.avg_pool2d(pred * t_detached, window_size, stride=1, padding=pad) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / (
        (mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2) + 1e-12
    )
    return (1.0 - ssim_map)[valid_mask].mean()


# ═══════════════════════════════════════════════════════════════
#  Unified DistillationLoss
# ═══════════════════════════════════════════════════════════════

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation from ZoeDepth teacher.

    GT-supervised terms (absolute depth):
      - alpha * MaskedL1Loss        (pixel-level depth)
      - edge_lambda * edge_aware_gt_loss  (edge-weighted depth)
      - silog_lambda * SiLog        (scale-invariant log)
      - grad_gt_lambda * gradient_consistency  (gradient from GT)
      - ssim_lambda * SSIM          (structure from GT)

    Teacher distillation terms (ALL scale-invariant):
      - gamma * imgrad_cosine_loss  (gradient direction, not magnitude)
      - teacher_edge_lambda * teacher_edge_guidance_loss  (edge presence)
      - distill_multiscale_lambda * scale_invariant_multiscale_loss
      - distill_zscore_lambda * scale_invariant_teacher_l1  (z-scored depth)
      - distill_ssim_lambda * teacher_ssim_distill_loss  (SSIM vs teacher)

    aux_lambda * aux_loss for auxiliary prediction head.
    """

    def __init__(
        self,
        alpha=1.0,
        gamma=0.1,
        edge_lambda=0.2,
        edge_weight=4.0,
        silog_lambda=0.0,
        aux_lambda=0.2,
        distill_multiscale_lambda=0.05,
        grad_gt_lambda=0.0,
        ssim_lambda=0.0,
        teacher_edge_lambda=0.1,
        distill_zscore_lambda=0.05,
        distill_ssim_lambda=0.05,
    ):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.edge_lambda = edge_lambda
        self.edge_weight = edge_weight
        self.silog_lambda = silog_lambda
        self.aux_lambda = aux_lambda
        self.distill_multiscale_lambda = distill_multiscale_lambda
        self.grad_gt_lambda = grad_gt_lambda
        self.ssim_lambda = ssim_lambda
        self.teacher_edge_lambda = teacher_edge_lambda
        self.distill_zscore_lambda = distill_zscore_lambda
        self.distill_ssim_lambda = distill_ssim_lambda
        self.loss_gt = MaskedL1Loss()
        self.loss_gt_edge = edge_aware_gt_loss

    def forward(self, pred, target_gt, target_teacher, distill_scale=1.0):
        if isinstance(pred, (tuple, list)):
            pred_main = pred[0]
            pred_aux = pred[1] if len(pred) > 1 else None
        else:
            pred_main = pred
            pred_aux = None

        # ── GT-supervised terms ──
        l_gt = self.loss_gt(pred_main, target_gt)
        l_gt_edge = self.loss_gt_edge(pred_main, target_gt, edge_weight=self.edge_weight)
        l_silog = silog_loss(pred_main, target_gt)
        l_grad_gt = gradient_consistency_loss(pred_main, target_gt)
        l_ssim = ssim_loss(pred_main, target_gt)

        l_aux = torch.tensor(0.0, device=pred_main.device)
        if pred_aux is not None:
            l_aux = self.loss_gt(_match_spatial(pred_aux, target_gt), target_gt)

        # ── Teacher distillation terms (ALL scale-invariant) ──
        l_grad_cos = imgrad_cosine_loss(pred_main, target_teacher)
        l_teacher_edge = teacher_edge_guidance_loss(pred_main, target_teacher)
        l_ms_si = scale_invariant_multiscale_loss(pred_main, target_teacher, scales=(2, 4))
        l_zscore = scale_invariant_teacher_l1(pred_main, target_teacher)
        l_teacher_ssim = teacher_ssim_distill_loss(pred_main, target_teacher)

        return (
            self.alpha * l_gt
            + self.edge_lambda * l_gt_edge
            + self.silog_lambda * l_silog
            + self.aux_lambda * l_aux
            + self.grad_gt_lambda * l_grad_gt
            + self.ssim_lambda * l_ssim
            + self.teacher_edge_lambda * l_teacher_edge
            + distill_scale * (
                self.gamma * l_grad_cos
                + self.distill_multiscale_lambda * l_ms_si
                + self.distill_zscore_lambda * l_zscore
                + self.distill_ssim_lambda * l_teacher_ssim
            )
        )
