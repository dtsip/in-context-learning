import os

import matplotlib.pyplot as plt
import seaborn as sns

from eval import get_run_metrics, baseline_names, get_model_from_run
from models import build_model

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")

relevant_model_names = {
    "sparse_regression_killer": [
        "Transformer",
        "Least Squares",
        "Ridge (alpha=0.5)",
    ],
    "heavy_tail_noise_killer": [
        "Transformer",
        "Least Squares",
        "Ridge (alpha=0.5)",
    ],
    "bounded_support_killer": [
        "Transformer",
        "Least Squares",
        "Ridge (alpha=0.5)",
    ],
    "mixture_tasks_killer": [
        "Transformer",
        "Least Squares",
        "Ridge (alpha=0.5)",
    ],
    "transfer_tradeoff_task": [
        "Transformer",
        "Least Squares",
        "Ridge (alpha=0.5)",
    ],
    "wlaplace_noisypoisson": [
        "Transformer",
        "Least Squares",
        "Ridge (alpha=0.5)",
    ],
    "laplace_weighted_regression": [
        "Transformer",
        "Least Squares",
        "Ridge (alpha=0.5)",
    ],
    "exponential_weighted_regression": [
        "Transformer",
        "Least Squares",
        "Ridge (alpha=0.5)",
    ],
    "uniform_hypersphere_regression": [
        "Transformer",
        "Least Squares",
        "Ridge (alpha=0.5)",
        "Ridge (alpha=0.1)",
        "3-Nearest Neighbors",
        "Averaging",
    ],
    "noisy_linear_regression": [
        "Transformer",
        "Least Squares",
        "Ridge (alpha=0.1)",
        "Ridge (alpha=0.5)",
        "Ridge (alpha=1.0)",
        "Ridge (alpha=2.0)",
        "Ridge (alpha=3.0)",
        "3-Nearest Neighbors",
        "Averaging"
    ],
    "linear_regression": [
        "Transformer", 
        "Least Squares",
        "Ridge (alpha=0.1)",
        "Ridge (alpha=0.5)",
        "Ridge (alpha=1.0)",
        "Ridge (alpha=2.0)",
        "Ridge (alpha=3.0)",
        "3-Nearest Neighbors",
        "Averaging"
    ],
    "sparse_linear_regression": [
        "Transformer",
        "Least Squares", 
        "3-Nearest Neighbors",
        "Averaging",
        "Lasso (alpha=0.001)",
        "Lasso (alpha=0.01)",
        "Lasso (alpha=0.1)",
        "Lasso (alpha=1.0)",
        "Ridge (alpha=0.5)"
    ],
    "decision_tree": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Decision Tree (max_depth=4)", 
        "Decision Tree (unlimited)",
        "XGBoost",
        "Averaging"
    ],
    "relu_2nn_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN (Adam)",
        "Averaging"
    ],
    "ar1_linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Ridge (alpha=0.1)",
        "Ridge (alpha=1.0)",
        "Ridge Var Adj (alpha=1.0, ar=0.5)",
        "Feasible GLS", 
        "GLS (ar=0.5)",
        "Averaging"
    ]
    ,
}


def basic_plot(metrics, models=None, trivial=1.0):
    fig, ax = plt.subplots(1, 1)

    if models is not None:
        print(models)
        available = [m for m in models if m in metrics]
        missing = [m for m in models if m not in metrics]
        if missing:
            print("Missing metrics for:", missing)
        metrics = {k: metrics[k] for k in available}

    color = 0
    ax.axhline(trivial, ls="--", color="gray")
    for name, vs in metrics.items():
        ax.plot(vs["mean"], "-", label=name, color=palette[color % 10], lw=2)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        color += 1
    ax.set_xlabel("in-context examples")
    ax.set_ylabel("squared error")
    ax.set_xlim(-1, len(low) + 0.1)
    ax.set_ylim(-0.1, 5)
    


    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    # legend = ax.legend(loc="best")

    fig.set_size_inches(4, 3)
    for line in legend.get_lines():
        line.set_linewidth(3)

    return fig, ax


def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None):
    all_metrics = {}
    for _, r in df.iterrows():
        if valid_row is not None and not valid_row(r):
            continue

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        print(r.run_name, r.run_id)
        metrics = get_run_metrics(run_path, skip_model_load=True)

        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            for model_name, m in results.items():
                # if "gpt2" in model_name in model_name:
                #     model_name = r.model
                # code fix
                if "gpt2" in model_name:
                    model_name = r.model  # r.model = "Transformer"
                else:
                    model_name = baseline_names(model_name)
                if rename_model is not None:
                    model_name = rename_model(model_name, r)
                m_processed = {}
                n_dims = conf.model.n_dims

                xlim = 2 * n_dims + 1
                if r.task in ["relu_2nn_regression", "decision_tree"]:
                    xlim = 200

                normalization = n_dims
                if r.task == "sparse_linear_regression":
                    try:
                        normalization = int(r.kwargs.split("=")[-1])
                    except (ValueError, AttributeError):
                        # Use default sparsity or n_dims if kwargs is empty
                        normalization = n_dims
                if r.task == "decision_tree":
                    normalization = 1

                for k, v in m.items():
                    # v = v[:xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
            all_metrics[eval_name].update(processed_results)
    return all_metrics
