"""
消融实验综合评估与对比脚本
============================
加载各消融变体的训练结果，在验证集上统一评估，生成对比表格和可视化。

消融变体:
  1. Full       - CA + GuidedUpsample + Distill + SSIM + gradGT
  2. No-CA      - GuidedUpsample + Distill + SSIM + gradGT
  3. No-Guided  - CA + Distill + SSIM + gradGT (bilinear upsample)
  4. No-Distill - CA + GuidedUpsample + SSIM + gradGT
  5. No-SSIM    - CA + GuidedUpsample + Distill
  6. Baseline   - CA removed + bilinear + no distill + no SSIM/gradGT

Usage:
  python ablation_eval.py [--gpu 0] [--save-dir ablation_results]
"""

import os
import sys
import csv
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from collections import OrderedDict

import models
from Data import createDataLoaders, UseGeoDataset
from metrics import AverageMeter, Result
from utils import merge_into_row_with_gt, colored_depthmap, save_image

cudnn.benchmark = True


# ─────────────────── Ablation variants definition ───────────────────
ABLATION_VARIANTS = OrderedDict([
    ("Full", {
        "checkpoint": "results/usegeo.MobileNet.fpnlite.512x768.lr=0.0001.bs=8/model_best.pth.tar",
        "use_ca": True,
        "use_guided_upsample": True,
        "desc": "CA + Guided + Distill + SSIM + gradGT",
        "color": "#2ecc71",
    }),
    ("No-CA", {
        "checkpoint": "results/usegeo.MobileNet.fpnlite.512x768.lr=0.0001.bs=8.noca.guided/model_best.pth.tar",
        "use_ca": False,
        "use_guided_upsample": True,
        "desc": "Guided + Distill + SSIM + gradGT (no CoordAtt)",
        "color": "#3498db",
    }),
    ("No-Guided", {
        "checkpoint": "results/usegeo.MobileNet.fpnlite.512x768.lr=0.0001.bs=8.ca.bilinear.noga/model_best.pth.tar",
        "use_ca": True,
        "use_guided_upsample": False,
        "desc": "CA + bilinear upsample + Distill + SSIM + gradGT",
        "color": "#e74c3c",
    }),
    ("No-Distill", {
        "checkpoint": "results/usegeo.MobileNet.fpnlite.512x768.lr=0.0001.bs=8.ca.guided.nodistill/model_best.pth.tar",
        "use_ca": True,
        "use_guided_upsample": True,
        "desc": "CA + Guided + SSIM + gradGT (no distillation)",
        "color": "#f39c12",
    }),
    ("No-SSIM", {
        "checkpoint": "results/usegeo.MobileNet.fpnlite.512x768.lr=0.0001.bs=8.ca.guided.nossim/model_best.pth.tar",
        "use_ca": True,
        "use_guided_upsample": True,
        "desc": "CA + Guided + Distill (no SSIM/gradGT)",
        "color": "#9b59b6",
    }),
    ("Baseline", {
        "checkpoint": "results/usegeo.MobileNet.fpnlite.512x768.lr=0.0001.bs=8.noca.bilinear.baseline/model_best.pth.tar",
        "use_ca": False,
        "use_guided_upsample": False,
        "desc": "Only MobileNetV2 + FPNLite + bilinear + edge-L1",
        "color": "#95a5a6",
    }),
])


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation study evaluation")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--save-dir", default="ablation_results", type=str,
                        help="output directory for ablation comparison files")
    parser.add_argument("--input-height", default=512, type=int)
    parser.add_argument("--input-width", default=768, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--sample-images", default=10, type=int,
                        help="number of comparison images to save for each variant")
    return parser.parse_args()


def _load_training_checkpoint(path, map_location=None):
    kwargs = {}
    if map_location is not None:
        kwargs['map_location'] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_model(variant_cfg, device):
    """Load a model from checkpoint with the specified configuration.

    Auto-detects training resolution from the checkpoint's saved args.
    Returns (model, eval_height, eval_width).
    """
    ckpt = _load_training_checkpoint(variant_cfg["checkpoint"], map_location="cpu")
    state_dict = ckpt["model"].state_dict() if "model" in ckpt else ckpt

    # Auto-detect evaluation resolution from checkpoint
    saved_args = ckpt.get('args')
    if saved_args is not None:
        eval_h = getattr(saved_args, 'input_height', 512)
        eval_w = getattr(saved_args, 'input_width', 768)
    else:
        eval_h, eval_w = 512, 768

    model = models.MobileNet(
        decoder="fpnlite",
        output_size=(eval_h, eval_w),
        pretrained=False,
        use_ca=variant_cfg["use_ca"],
        use_guided_upsample=variant_cfg["use_guided_upsample"],
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, eval_h, eval_w


def evaluate_model(model, val_loader, device):
    """Run evaluation on validation set, return AverageMeter result."""
    # Warmup to avoid CUDA kernel compilation skewing timing
    dummy = torch.randn(1, 3, 512, 768).to(device)
    for _ in range(3):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    avg_meter = AverageMeter()
    with torch.no_grad():
        for data in val_loader:
            if len(data) == 3:
                input_img, target_gt, _ = data
            else:
                input_img, target_gt = data
            input_img = input_img.to(device)
            target_gt = target_gt.to(device)

            torch.cuda.synchronize()
            start = time.time()
            pred = model(input_img)
            torch.cuda.synchronize()
            gpu_time = time.time() - start

            result = Result()
            result.evaluate(pred.data, target_gt.data, loader_args.depth_scale)
            avg_meter.update(result, gpu_time, 0.0, input_img.size(0))

    return avg_meter.average()


def count_parameters(model):
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def write_summary_table(results, save_dir):
    """Write a comprehensive markdown summary table."""
    os.makedirs(save_dir, exist_ok=True)
    md_path = os.path.join(save_dir, "ablation_summary.md")
    csv_path = os.path.join(save_dir, "ablation_summary.csv")

    # Sort: put Full first, then by RMSE (ascending)
    ordered = sorted(results.items(), key=lambda x: x[1]["rmse"])
    # Ensure Full is first
    if "Full" in results:
        ordered.insert(0, ("Full", results["Full"]))
        seen = {"Full"}
        ordered = [("Full", results["Full"])] + [
            (k, v) for k, v in sorted(results.items(), key=lambda x: x[1]["rmse"])
            if k != "Full"
        ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 消融实验对比汇总\n\n")
        f.write(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Main comparison table
        f.write("## 核心指标对比\n\n")
        f.write("| 消融变体 | RMSE↓ | MAE↓ | AbsRel↓ | δ1↑ | Edge MAE↓ | GPU(ms)↓ | Params(M) | 配置说明 |\n")
        f.write("|---------|-------|------|---------|-----|-----------|----------|-----------|----------|\n")

        base_rmse = results["Full"]["rmse"] if "Full" in results else None

        for name, r in ordered:
            delta_str = ""
            if base_rmse and name != "Full":
                delta = r["rmse"] - base_rmse
                delta_str = f" (+{delta*1000:.2f}‰)" if delta > 0 else f" ({delta*1000:.2f}‰)"

            f.write(
                f"| **{name}** "
                f"| {r['rmse']:.4f}{delta_str} "
                f"| {r['mae']:.4f} "
                f"| {r['absrel']:.4f} "
                f"| {r['delta1']:.4f} "
                f"| {r['edge_mae']:.4f} "
                f"| {r['gpu_time']*1000:.2f} "
                f"| {r['params']:.2f} "
                f"| {ABLATION_VARIANTS.get(name, {}).get('desc', '-')} |\n"
            )

        # Component contributions
        f.write("\n## 各组件贡献分析\n\n")
        f.write("| 组件 | 对比对 | ΔRMSE (‰) | ΔMAE (‰) | ΔAbsRel (‰) | 影响说明 |\n")
        f.write("|------|--------|-----------|----------|-------------|----------|\n")

        # CoordAtt contribution: Full vs No-CA
        _write_component_row(f, results, "CoordAtt", "Full", "No-CA",
                             "CoordAtt对道路场景精度提升几乎为零")

        # GuidedUpsample contribution: Full vs No-Guided
        _write_component_row(f, results, "GuidedUpsample", "Full", "No-Guided",
                             "RGB引导上采样对边缘清晰度贡献显著")

        # Distillation contribution: Full vs No-Distill
        _write_component_row(f, results, "Knowledge Distillation", "Full", "No-Distill",
                             "知识蒸馏从ZoeDepth teacher引入结构先验")

        # SSIM+gradGT contribution: Full vs No-SSIM
        _write_component_row(f, results, "SSIM + gradGT", "Full", "No-SSIM",
                             "结构损失和梯度一致性对边缘保持的贡献")

        # Cumulative: Full vs Baseline
        _write_component_row(f, results, "All Enhancements", "Full", "Baseline",
                             "所有改进的累计贡献（Full vs Baseline）")

        # Write detailed per-variant section
        f.write("\n## 各变体详细指标\n\n")
        for name, r in ordered:
            f.write(f"### {name}\n\n")
            f.write(f"- **配置**: {ABLATION_VARIANTS.get(name, {}).get('desc', '-')}\n")
            f.write(f"- **RMSE**: {r['rmse']:.6f}\n")
            f.write(f"- **MAE**: {r['mae']:.6f}\n")
            f.write(f"- **AbsRel**: {r['absrel']:.6f}\n")
            f.write(f"- **δ1**: {r['delta1']:.6f}\n")
            f.write(f"- **δ2**: {r['delta2']:.6f}\n")
            f.write(f"- **δ3**: {r['delta3']:.6f}\n")
            f.write(f"- **Edge MAE**: {r['edge_mae']:.6f}\n")
            f.write(f"- **MSE**: {r['mse']:.6f}\n")
            f.write(f"- **GPU Time**: {r['gpu_time']:.6f}s\n")
            f.write(f"- **Parameters**: {r['params']:.2f}M\n")
            f.write("\n")

    # Also write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["Variant", "RMSE", "MAE", "AbsRel", "delta1", "delta2", "delta3",
                          "Edge_MAE", "MSE", "GPU_Time", "Params_M", "Config"])
        for name, r in ordered:
            writer.writerow([
                name, r["rmse"], r["mae"], r["absrel"], r["delta1"], r["delta2"],
                r["delta3"], r["edge_mae"], r["mse"], r["gpu_time"], r["params"],
                ABLATION_VARIANTS.get(name, {}).get("desc", "-")
            ])

    print(f"=> Ablation summary saved to: {md_path}")
    print(f"=> Ablation CSV saved to: {csv_path}")
    return ordered


def _write_component_row(f, results, component, key_a, key_b, note):
    """Helper to write a component contribution row."""
    if key_a in results and key_b in results:
        dr = (results[key_a]["rmse"] - results[key_b]["rmse"]) * 1000
        dm = (results[key_a]["mae"] - results[key_b]["mae"]) * 1000
        da = (results[key_a]["absrel"] - results[key_b]["absrel"]) * 1000
        sign_r = "↓" if dr <= 0 else "↑"
        sign_m = "↓" if dm <= 0 else "↑"
        sign_a = "↓" if da <= 0 else "↑"
        f.write(f"| {component} | {key_a} vs {key_b} | {abs(dr):.2f} {sign_r} | {abs(dm):.2f} {sign_m} | {abs(da):.2f} {sign_a} | {note} |\n")


def plot_bar_chart(ordered, save_dir):
    """Generate bar charts comparing key metrics across variants."""
    names = [x[0] for x in ordered]
    rmses = [x[1]["rmse"] for x in ordered]
    maes = [x[1]["mae"] for x in ordered]
    absrels = [x[1]["absrel"] for x in ordered]
    gpu_times = [x[1]["gpu_time"] * 1000 for x in ordered]
    params_list = [x[1]["params"] for x in ordered]
    colors = [ABLATION_VARIANTS.get(n, {}).get("color", "#333333") for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # RMSE
    bars = axes[0].bar(names, rmses, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title("RMSE (m) ↓", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("RMSE (m)")
    axes[0].tick_params(axis='x', rotation=30, labelsize=9)
    for bar, val in zip(bars, rmses):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # MAE
    bars = axes[1].bar(names, maes, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title("MAE (m) ↓", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("MAE (m)")
    axes[1].tick_params(axis='x', rotation=30, labelsize=9)
    for bar, val in zip(bars, maes):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # AbsRel
    bars = axes[2].bar(names, absrels, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_title("AbsRel ↓", fontsize=13, fontweight='bold')
    axes[2].set_ylabel("AbsRel")
    axes[2].tick_params(axis='x', rotation=30, labelsize=9)
    for bar, val in zip(bars, absrels):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # GPU Time
    bars = axes[3].bar(names, gpu_times, color=colors, edgecolor='black', linewidth=0.5)
    axes[3].set_title("GPU推理时间 (ms) ↓", fontsize=13, fontweight='bold')
    axes[3].set_ylabel("时间 (ms)")
    axes[3].tick_params(axis='x', rotation=30, labelsize=9)
    for bar, val in zip(bars, gpu_times):
        axes[3].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # Overall figure title
    fig.suptitle("消融实验关键指标对比", fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    bar_path = os.path.join(save_dir, "ablation_metrics_bar.png")
    fig.savefig(bar_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"=> Bar chart saved to: {bar_path}")


def generate_radar_chart(ordered, save_dir):
    """Generate radar chart comparing relative performance."""
    from matplotlib.patches import RegularPolygon
    import matplotlib.lines as mlines

    # Normalize each metric to [0, 1] where 1 is best
    all_rmses = np.array([x[1]["rmse"] for x in ordered])
    all_maes = np.array([x[1]["mae"] for x in ordered])
    all_absrels = np.array([x[1]["absrel"] for x in ordered])
    all_edges = np.array([x[1]["edge_mae"] for x in ordered])
    all_gpu = np.array([x[1]["gpu_time"] for x in ordered])

    def normalize_best(arr):
        """Lower is better → invert so 1=best."""
        if arr.max() == arr.min():
            return np.ones_like(arr)
        return 1.0 - (arr - arr.min()) / (arr.max() - arr.min())

    metrics_labels = ["RMSE", "MAE", "AbsRel", "Edge MAE", "GPU Speed"]
    metrics_data = [
        normalize_best(all_rmses),
        normalize_best(all_maes),
        normalize_best(all_absrels),
        normalize_best(all_edges),
        normalize_best(all_gpu),
    ]

    num_vars = len(metrics_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for idx, (name, _) in enumerate(ordered):
        values = [m[idx] for m in metrics_data]
        values += values[:1]
        color = ABLATION_VARIANTS.get(name, {}).get("color", "#333333")
        ax.fill(angles, values, alpha=0.12, color=color)
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=name, markersize=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title("消融实验 — 归一化性能雷达图\n(外圈 = 更优)", fontsize=14, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10)

    radar_path = os.path.join(save_dir, "ablation_radar.png")
    fig.savefig(radar_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"=> Radar chart saved to: {radar_path}")


def plot_rmse_improvement_waterfall(ordered, save_dir):
    """Waterfall plot showing RMSE improvement of each component relative to Baseline."""
    # Find baseline
    baseline_rmse = None
    full_rmse = None
    for name, r in ordered:
        if name == "Baseline":
            baseline_rmse = r["rmse"]
        if name == "Full":
            full_rmse = r["rmse"]

    if baseline_rmse is None or full_rmse is None:
        print("=> Waterfall requires Baseline and Full variants.")
        return

    contributions = [
        ("Baseline", baseline_rmse),
        ("+ CoordAtt", results_dict.get("No-Guided", {}).get("rmse", baseline_rmse)),
        ("+ GuidedUpsample", results_dict.get("No-SSIM", {}).get("rmse", baseline_rmse)),
        ("+ Distillation", results_dict.get("No-Distill", {}).get("rmse", baseline_rmse)),
        ("+ SSIM+gradGT", results_dict.get("Full", {}).get("rmse", baseline_rmse)),
    ]

    # Simplified waterfall
    labels = ["Baseline", "+CoordAtt", "+Guided\nUpsample", "+Distillation", "+SSIM\n+gradGT"]
    values = []
    for name, _ in ordered:
        if name in results_dict:
            values.append(results_dict[name]["rmse"])
    # This is a simplified version — actual cumulative build requires more variants

    fig, ax = plt.subplots(figsize=(12, 5))

    variant_order = ["Baseline", "No-CA", "No-Guided", "No-Distill", "No-SSIM", "Full"]
    variant_labels = ["Baseline\n(no enhancements)", "Baseline\n+ Guided", "+ CA\n+ Guided", "+ Distill\n+ CA + Guided", "+ SSIM\n+ CA + Guided\n+ Distill", "Full\n(all enhancements)"]
    plot_values = []
    plot_labels = []

    for v in variant_order:
        if v in results_dict:
            plot_values.append(results_dict[v]["rmse"])
            plot_labels.append(v)

    colors_bar = [ABLATION_VARIANTS.get(n, {}).get("color", "#333") for n in plot_labels]

    bars = ax.bar(plot_labels, plot_values, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("RMSE (m) ↓", fontsize=13, fontweight='bold')
    ax.set_title("消融实验：各变体RMSE级联对比（数值越低越优）", fontsize=14, fontweight='bold')
    ax.axhline(y=baseline_rmse, color='gray', linestyle='--', alpha=0.5, label=f'基线 RMSE={baseline_rmse:.4f}')
    ax.axhline(y=full_rmse, color='green', linestyle='--', alpha=0.5, label=f'完整模型 RMSE={full_rmse:.4f}')
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', rotation=20, labelsize=10)

    for bar, val in zip(bars, plot_values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_rmse_cascade.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"=> RMSE cascade chart saved to: {path}")


# Global dict for chart functions
results_dict = {}


def _make_val_loader(height, width, batch_size):
    """Create validation DataLoader for given resolution (single-process for Windows safety)."""
    class Args:
        pass
    loader_args = Args()
    loader_args.data = "usegeo"
    loader_args.modality = "rgb"
    loader_args.input_height = height
    loader_args.input_width = width
    loader_args.batch_size = batch_size
    loader_args.workers = 0  # single-process avoids Windows multiprocessing issues
    loader_args.num_samples = 0
    loader_args.max_depth = 0.0
    loader_args.sparsifier = "uar"
    loader_args.data_path = "data"
    loader_args.teacher_path = "data/teacher_depth"
    loader_args.depth_scale = 100.0
    loader_args.teacher_scale = 100.0
    loader_args.data_aug = False
    loader_args.evaluate = ""
    _, val_loader = createDataLoaders(loader_args)
    return val_loader


def main():
    global results_dict
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Evaluate each variant at its own training resolution
    results_dict = {}
    for name, cfg in ABLATION_VARIANTS.items():
        ckpt_path = cfg["checkpoint"]
        if not os.path.exists(ckpt_path):
            print(f"=> [{name}] Checkpoint not found: {ckpt_path} — SKIPPING")
            results_dict[name] = {
                "rmse": float("nan"), "mae": float("nan"), "absrel": float("nan"),
                "delta1": float("nan"), "delta2": float("nan"), "delta3": float("nan"),
                "edge_mae": float("nan"), "mse": float("nan"), "gpu_time": float("nan"),
                "params": float("nan"),
            }
            continue

        print(f"\n{'='*60}")
        print(f"=> Evaluating [{name}]: {cfg['desc']}")
        print(f"=> Checkpoint: {ckpt_path}")
        print(f"=> Config: use_ca={cfg['use_ca']}, guided={cfg['use_guided_upsample']}")

        model, eval_h, eval_w = load_model(cfg, device)
        params_m = count_parameters(model)
        print(f"=> Parameters: {params_m:.2f}M")
        print(f"=> Evaluation resolution: {eval_h}x{eval_w}")

        val_loader = _make_val_loader(eval_h, eval_w, args.batch_size)
        print(f"=> Validation set: {len(val_loader)} batches")

        avg = evaluate_model(model, val_loader, device)
        print(f"=> RMSE={avg.rmse:.6f}, MAE={avg.mae:.6f}, AbsRel={avg.absrel:.6f}, "
              f"δ1={avg.delta1:.4f}, EdgeMAE={avg.edge_mae:.6f}, GPU={avg.gpu_time:.4f}s")

        results_dict[name] = {
            "rmse": avg.rmse,
            "mae": avg.mae,
            "absrel": avg.absrel,
            "delta1": avg.delta1,
            "delta2": avg.delta2,
            "delta3": avg.delta3,
            "edge_mae": avg.edge_mae,
            "mse": avg.mse,
            "gpu_time": avg.gpu_time,
            "params": params_m,
        }

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Generate outputs
    print(f"\n{'='*60}")
    print("=> Generating ablation comparison outputs...")

    ordered = write_summary_table(results_dict, save_dir)
    plot_bar_chart(ordered, save_dir)
    generate_radar_chart(ordered, save_dir)
    plot_rmse_improvement_waterfall(ordered, save_dir)

    # Generate LaTeX table for thesis
    latex_path = os.path.join(save_dir, "ablation_table.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% Ablation study results — auto-generated\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("  \\centering\n")
        f.write("  \\caption{Ablation study: contribution of each component.}\n")
        f.write("  \\label{tab:ablation}\n")
        f.write("  \\begin{tabular}{lcccccc}\n")
        f.write("    \\toprule\n")
        f.write("    Variant & RMSE (m) & MAE (m) & AbsRel & $\\delta_1$ & Edge MAE & Params (M) \\\\\n")
        f.write("    \\midrule\n")
        for name, r in ordered:
            f.write(f"    {name} & {r['rmse']:.4f} & {r['mae']:.4f} & {r['absrel']:.4f} & {r['delta1']:.4f} & {r['edge_mae']:.4f} & {r['params']:.2f} \\\\\n")
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"=> LaTeX table saved to: {latex_path}")

    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"All outputs saved to: {save_dir}/")
    print(f"  - ablation_summary.md       (Markdown report)")
    print(f"  - ablation_summary.csv      (CSV data)")
    print(f"  - ablation_metrics_bar.png  (Bar chart)")
    print(f"  - ablation_radar.png        (Radar chart)")
    print(f"  - ablation_rmse_cascade.png (RMSE cascade)")
    print(f"  - ablation_table.tex        (LaTeX for thesis)")


if __name__ == "__main__":
    main()
