import argparse
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval import get_model_from_run
from samplers import get_data_sampler
from tasks import get_task_sampler


def _select_device(model: torch.nn.Module) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_true_w(task) -> Optional[torch.Tensor]:
    if hasattr(task, "w_b"):
        return task.w_b[0, :, 0]
    return None


def _project_to_row_space(w: torch.Tensor, xs_ctx: torch.Tensor) -> torch.Tensor:
    if xs_ctx.numel() == 0:
        return torch.zeros_like(w)
    # xs_ctx: (k, d). Project w onto span{rows(xs_ctx)}
    x = xs_ctx
    gram = x @ x.t()
    proj_matrix = x.t() @ torch.linalg.pinv(gram) @ x
    return proj_matrix @ w


def _estimate_norm_band(data_sampler, device: torch.device, num_samples: int = 16384) -> Tuple[float, float]:
    batch_size = min(512, num_samples)
    collected = []
    remaining = num_samples
    while remaining > 0:
        cur = min(batch_size, remaining)
        xs = data_sampler.sample_xs(n_points=1, b_size=cur).to(device)
        norms = xs[:, 0, :].norm(dim=1)
        collected.append(norms)
        remaining -= cur
    norms = torch.cat(collected)
    low = torch.quantile(norms, 0.005).item()
    high = torch.quantile(norms, 0.995).item()
    return low, high


def _prepare(run_path: str):
    model, conf = get_model_from_run(run_path)
    device = _select_device(model)
    model = model.to(device).eval()

    n_dims = conf.model.n_dims
    data_sampler = get_data_sampler(conf.training.data, n_dims, **getattr(conf.training, "data_kwargs", {}))
    task_sampler = get_task_sampler(
        conf.training.task,
        n_dims,
        batch_size=1,
        **conf.training.task_kwargs,
    )
    return model, conf, data_sampler, task_sampler, device


def plot_prefix_conditioned_function(
    run_path: str,
    num_dirs: int = 3,
    ks: Optional[Sequence[int]] = None,
    sweep_radius: float = 15.0,
    num_steps: int = 201,
    seed: Optional[int] = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    model, conf, data_sampler, task_sampler, device = _prepare(run_path)
    task = task_sampler()
    w = _get_true_w(task)
    w = w.to(device) if w is not None else None

    if ks is None:
        d = conf.model.n_dims
        max_pts = conf.training.curriculum.points.end
        ks = [max(1, d // 2), d, min(2 * d, max_pts)]

    ks = list(dict.fromkeys(sorted(ks)))
    band_low, band_high = _estimate_norm_band(data_sampler, device)

    ts = torch.linspace(-sweep_radius, sweep_radius, steps=num_steps, device=device)
    fig, axes = plt.subplots(1, num_dirs, figsize=(14, 4), sharey=True)
    if num_dirs == 1:
        axes = [axes]

    for idx in range(num_dirs):
        ax = axes[idx]
        u = torch.randn(conf.model.n_dims, device=device)
        u = u / (u.norm() + 1e-8)
        xs_ctx_for_proj = None

        for k in ks:
            xs_ctx = data_sampler.sample_xs(n_points=k, b_size=1).to(device)
            ys_ctx = task.evaluate(xs_ctx).to(device)

            preds = []
            for t in ts:
                x_query = (t * u).view(1, 1, -1)
                xs_in = torch.cat([xs_ctx, x_query], dim=1)
                ys_in = torch.cat([ys_ctx, torch.zeros_like(ys_ctx[:, :1])], dim=1)
                with torch.no_grad():
                    out = model(xs_in, ys_in, inds=[k])
                preds.append(out[0, 0].item())

            if xs_ctx_for_proj is None:
                xs_ctx_for_proj = xs_ctx[0]

            if k == conf.model.n_dims:
                label = "#dims in-context"
            elif k == ks[-1]:
                label = f"{k} in-context"
            else:
                label = f"k={k}"
            ax.plot(ts.detach().cpu().numpy(), preds, lw=2, label=label)

        if w is not None:
            ground_truth = (ts * torch.dot(u, w)).detach().cpu().numpy()
            ax.plot(ts.detach().cpu().numpy(), ground_truth, color="C0", lw=2, label="ground truth")

            if xs_ctx_for_proj is not None:
                w_proj = _project_to_row_space(w, xs_ctx_for_proj)
                gt_proj = (ts * torch.dot(u, w_proj)).detach().cpu().numpy()
                ax.plot(ts.detach().cpu().numpy(), gt_proj, color="C0", lw=2, ls="--", label="ground truth proj.")

        ax.axvspan(-band_high, -band_low, color="#000000", alpha=0.08)
        ax.axvspan(band_low, band_high, color="#000000", alpha=0.08)
        ax.set_xlabel("query scale")
        if idx == 0:
            ax.set_ylabel("model prediction")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout()
    plt.show()


def _cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    denom = (u.norm() * v.norm()).item()
    if denom < 1e-8:
        return float("nan")
    return float(torch.dot(u, v).item() / denom)


def compute_gradient_alignment_curves(
    run_path: str,
    ks: Optional[Sequence[int]] = None,
    num_prompts: int = 1280,
    seed: Optional[int] = None,
) -> Dict[str, List[Tuple[int, float]]]:
    if seed is not None:
        torch.manual_seed(seed)

    model, conf, data_sampler, task_sampler, device = _prepare(run_path)
    if ks is None:
        d = conf.model.n_dims
        max_pts = conf.training.curriculum.points.end
        ks = [max(1, d // 2), d, min(2 * d, max_pts)]
    ks = list(dict.fromkeys(sorted(ks)))
    max_k = ks[-1]

    series_proj = defaultdict(list)
    series_true = defaultdict(list)

    for _ in range(num_prompts):
        task = task_sampler()
        w = _get_true_w(task)
        if w is None:
            continue
        w = w.to(device)

        xs = data_sampler.sample_xs(n_points=max_k + 1, b_size=1).to(device)
        ys = task.evaluate(xs).to(device)

        for k in ks:
            ctx_xs = xs[:, :k, :]
            ctx_ys = ys[:, :k]
            x_query = xs[:, k : k + 1, :].clone().detach().requires_grad_(True)

            xs_in = torch.cat([ctx_xs, x_query], dim=1)
            ys_in = torch.cat([ctx_ys, torch.zeros_like(ctx_ys[:, :1])], dim=1)

            pred = model(xs_in, ys_in, inds=[k])
            grad = torch.autograd.grad(pred.sum(), x_query, retain_graph=False)[0].view(-1)

            w_proj = _project_to_row_space(w, ctx_xs[0])

            series_true[k].append(_cosine(grad, w))
            series_proj[k].append(_cosine(grad, w_proj))

    def _finalize(series_dict):
        values = []
        for k in ks:
            data = np.array(series_dict[k], dtype=float)
            if data.size == 0:
                values.append((k, float("nan")))
            else:
                values.append((k, float(np.nanmean(data))))
        return values

    return {
        "with_true_w": _finalize(series_true),
        "with_projected_w": _finalize(series_proj),
    }


def plot_gradient_alignment(
    run_path: str,
    ks: Optional[Sequence[int]] = None,
    num_prompts: int = 1280,
    seed: Optional[int] = None,
):
    curves = compute_gradient_alignment_curves(run_path, ks=ks, num_prompts=num_prompts, seed=seed)

    plt.figure(figsize=(6, 4))
    xs_true = [k for k, _ in curves["with_true_w"]]
    ys_true = [val for _, val in curves["with_true_w"]]
    plt.plot(xs_true, ys_true, marker="o", label="grad vs w")

    xs_proj = [k for k, _ in curves["with_projected_w"]]
    ys_proj = [val for _, val in curves["with_projected_w"]]
    plt.plot(xs_proj, ys_proj, marker="o", label="grad vs proj(w)")

    plt.xlabel("# in-context examples (k)")
    plt.ylabel("normalized inner product")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_learning_curve(run_path: str, use_log_scale: bool = True):
    """
    Plot learning curve: MSE vs context length k for Transformer, OLS, Ridge.
    Load metrics from metrics.json file.
    """
    import json
    import os
    
    metrics_path = os.path.join(run_path, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Error: metrics.json not found at {metrics_path}")
        return
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    plt.figure(figsize=(10, 6))
    
    # Extract models from "standard" evaluation
    if "standard" in metrics:
        standard_eval = metrics["standard"]
        ks = list(range(1, len(next(iter(standard_eval.values()))["mean"]) + 1))
        
        for model_name, data in standard_eval.items():
            if isinstance(data, dict) and "mean" in data:
                means = data["mean"]
                plt.plot(ks, means, marker="o", label=model_name, lw=2, markersize=4)
    
    plt.xlabel("# in-context examples (k)")
    plt.ylabel("MSE")
    if use_log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_prediction_scatter(run_path: str, k: Optional[int] = None, num_samples: int = 500, seed: Optional[int] = None):
    """
    Plot prediction vs ground truth scatter plot.
    Shows bias/shrinkage effects: Transformer vs OLS.
    Generates predictions on-the-fly by evaluating on test data.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    model, conf, data_sampler, task_sampler, device = _prepare(run_path)
    
    d = conf.model.n_dims
    if k is None:
        k = d  # Use k = d for visualization
    
    # Collect predictions from both Transformer and OLS
    transformer_preds = []
    ols_preds = []
    y_true_list = []
    
    for i in range(num_samples):
        task = task_sampler()
        xs = data_sampler.sample_xs(n_points=k + 1, b_size=1).to(device)
        ys = task.evaluate(xs).to(device)
        
        ctx_xs = xs[:, :k, :]
        ctx_ys = ys[:, :k]
        x_query = xs[:, k : k + 1, :]
        y_query = ys[:, k : k + 1, 0]
        
        # Transformer prediction
        xs_in = torch.cat([ctx_xs, x_query], dim=1)
        ys_in = torch.cat([ctx_ys, torch.zeros_like(ctx_ys[:, :1])], dim=1)
        with torch.no_grad():
            transformer_pred = model(xs_in, ys_in, inds=[k]).cpu().numpy().flatten()
        
        # OLS prediction
        X = ctx_xs[0].cpu().numpy()
        y = ctx_ys[0, :, 0].cpu().numpy()
        try:
            w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
            x_q = x_query[0, 0].cpu().numpy()
            ols_pred = np.dot(w_ols, x_q)
        except:
            ols_pred = np.array([0.0])
        
        transformer_preds.append(transformer_pred[0])
        ols_preds.append(ols_pred if isinstance(ols_pred, (int, float)) else ols_pred[0])
        y_true_list.append(y_query[0, 0].cpu().item())
    
    transformer_preds = np.array(transformer_preds)
    ols_preds = np.array(ols_preds)
    y_true = np.array(y_true_list)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = [(transformer_preds, "Transformer", "red"), (ols_preds, "OLS", "blue")]
    
    for idx, (preds, name, color) in enumerate(models):
        ax = axes[idx]
        ax.scatter(y_true, preds, alpha=0.5, s=20, color=color)
        
        # Perfect prediction line
        lim = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
        ax.plot(lim, lim, "k--", lw=2, label="perfect")
        
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.set_title(f"{name} (k={k})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_weight_recovery(run_path: str, num_prompts: int = 1280, seed: Optional[int] = None):
    """
    Plot histogram of cosine similarity between predicted weight and true weight.
    Compares Transformer vs OLS weight recovery.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    model, conf, data_sampler, task_sampler, device = _prepare(run_path)
    
    d = conf.model.n_dims
    max_pts = conf.training.curriculum.points.end
    k = d  # Use k = d for comparison
    
    transformer_sims = []
    ols_sims = []
    
    for _ in range(num_prompts):
        task = task_sampler()
        w_true = _get_true_w(task)
        if w_true is None:
            continue
        w_true = w_true.to(device)
        
        xs = data_sampler.sample_xs(n_points=k + 1, b_size=1).to(device)
        ys = task.evaluate(xs).to(device)
        
        ctx_xs = xs[:, :k, :]
        ctx_ys = ys[:, :k]
        x_query = xs[:, k : k + 1, :].clone().detach().requires_grad_(True)
        
        # Transformer weight estimate via gradient
        xs_in = torch.cat([ctx_xs, x_query], dim=1)
        ys_in = torch.cat([ctx_ys, torch.zeros_like(ctx_ys[:, :1])], dim=1)
        
        pred = model(xs_in, ys_in, inds=[k])
        grad_transformer = torch.autograd.grad(pred.sum(), x_query, retain_graph=False)[0].view(-1)
        
        # OLS weight estimate
        X = ctx_xs[0]
        y = ctx_ys[0, :, 0]
        w_ols = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.view(-1)
        
        transformer_sims.append(_cosine(grad_transformer, w_true))
        ols_sims.append(_cosine(w_ols, w_true))
    
    plt.figure(figsize=(10, 6))
    plt.hist(transformer_sims, bins=30, alpha=0.6, label="Transformer", color="red", density=True)
    plt.hist(ols_sims, bins=30, alpha=0.6, label="OLS", color="blue", density=True)
    
    plt.xlabel("Cosine Similarity with true weight")
    plt.ylabel("Density")
    plt.title("Weight Recovery: Transformer vs OLS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main(args: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Reproduce Figure 3 diagnostics.")
    parser.add_argument("run_path", type=str, help="Path to a trained run directory.")
    parser.add_argument("--num_dirs", type=int, default=3, help="number of random prompts for Fig 3a")
    parser.add_argument("--num_prompts", type=int, default=1280, help="number of random prompts for Fig 3b")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--no_fig3a", action="store_true", help="skip prefix-conditioned function plot")
    parser.add_argument("--no_fig3b", action="store_true", help="skip gradient alignment plot")
    parser.add_argument("--learning_curve", action="store_true", help="plot learning curve vs context length")
    parser.add_argument("--scatter", action="store_true", help="plot prediction vs ground truth scatter")
    parser.add_argument("--weight_recovery", action="store_true", help="plot weight recovery histogram")
    parsed = parser.parse_args(args=args)

    if not parsed.no_fig3a:
        plot_prefix_conditioned_function(parsed.run_path, num_dirs=parsed.num_dirs, seed=parsed.seed)
    if not parsed.no_fig3b:
        plot_gradient_alignment(parsed.run_path, num_prompts=parsed.num_prompts, seed=parsed.seed)
    if parsed.learning_curve:
        plot_learning_curve(parsed.run_path)
    if parsed.scatter:
        plot_prediction_scatter(parsed.run_path, num_samples=parsed.num_prompts, seed=parsed.seed)
    if parsed.weight_recovery:
        plot_weight_recovery(parsed.run_path, num_prompts=parsed.num_prompts, seed=parsed.seed)


if __name__ == "__main__":
    main()


# python figure3_4.py <run_path> --learning_curve --scatter --weight_recovery