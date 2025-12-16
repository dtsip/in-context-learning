import os
from random import randint
import uuid

import curriculum
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def _sanitize_training_kwargs(args):
    """
    Remove conflicting/irrelevant kwargs to avoid sampler/task constructor errors.
    Rules:
    - data_kwargs: keep 'k' ONLY when data == 'sparse_gaussian' (k = number of non-zero coords).
    - task_kwargs: keep 'sparsity' ONLY when task == 'sparse_linear_regression'.
    - In addition, apply per-task and per-data whitelists to drop unsupported keys.
    """
    # Defensive copy
    data_kwargs = dict(getattr(args.training, "data_kwargs", {}) or {})
    task_kwargs = dict(getattr(args.training, "task_kwargs", {}) or {})

    # Per-data whitelists
    data_whitelist = {
        "gaussian": {"bias", "scale"},
        "sparse_gaussian": {"k", "bias", "scale"},
        "ar1": {"rho", "noise_std", "bias", "scale", "compute_gradient"},
        "vr1": {"ar1_mat", "noise_std", "bias", "scale"},
        "ar2": {"ar1_coef", "ar2_coef", "noise_std", "bias", "scale"},
        "vr2": {"ar1_mat", "ar2_mat", "noise_std", "bias", "scale"},
        "nonstation": {"coef_base", "coef_amplitude", "noise_std", "bias", "scale"},
        "exponential": {"bias", "scale", "rate"},
        "laplace": {"bias", "scale", "loc", "laplace_scale"},
        "gamma": {"bias", "scale", "concentration", "rate"},
        "beta": {"bias", "scale", "alpha", "beta"},
    }

    data_name = args.training.data
    if data_name in data_whitelist:
        allowed = data_whitelist[data_name]
        data_kwargs = {k: v for k, v in data_kwargs.items() if k in allowed}
    else:
        # Unknown data: drop potentially conflicting keys
        data_kwargs = {}

    # Per-task whitelists
    task_whitelist = {
        "linear_regression": {"scale", "uniform"},
        "sparse_linear_regression": {"scale", "sparsity", "valid_coords"},
        "linear_classification": {"scale", "uniform"},
        "relu_2nn_regression": {"scale", "hidden_layer_size"},
        "decision_tree": {"depth"},
        "noisy_linear_regression": {"scale", "noise_std", "renormalize_ys", "noise_type", "uniform", "w_distribution", "w_kwargs"},
        "ar1_linear_regression": {"scale", "ar_coef", "noise_std", "compute_gradient"},
        "uniform_hypersphere_regression": {"scale"},
        "wlaplace_noisypoisson": {"scale", "weight_scale", "poisson_rate"},
        "sparse_regression_killer": {"scale", "k_sparse"},
        "heavy_tail_noise_killer": {"scale", "noise_type", "df", "noise_scale"},
        "bounded_support_killer": {"scale", "rate"},
        "mixture_tasks_killer": {"scale"},
        "transfer_tradeoff_task": {"scale", "prior_type", "mixture_std"},

    }

    task_name = args.training.task
    if task_name in task_whitelist:
        allowed = task_whitelist[task_name]
        task_kwargs = {k: v for k, v in task_kwargs.items() if k in allowed}
    else:
        # Unknown task: be conservative
        task_kwargs = {}

    args.training.data_kwargs = data_kwargs
    args.training.task_kwargs = task_kwargs


def train(model, args):
    # Determine device - can override with --cpu_only flag
    use_cpu = getattr(args, 'cpu_only', False)
    device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    print(f"[TRAIN] Getting data sampler for {args.training.data}")
    data_kwargs = getattr(args.training, "data_kwargs", {}) or {}
    data_sampler = get_data_sampler(
        args.training.data, n_dims=n_dims, **data_kwargs
    )
    print(f"[TRAIN] Getting task sampler for {args.training.task}")
    task_kwargs = getattr(args.training, "task_kwargs", {}) or {}
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **task_kwargs
    )
    print("[TRAIN] Creating tqdm progress bar")
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if args.training.task == "sparse_linear_regression":
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated

        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
            device=device
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)
        # Ensure ys is on the same device as xs
        ys = ys.to(xs.device)

        loss_func = task.get_training_metric()
        # Disable mixed precision for now - testing numeric stability
        loss, output = train_step(model, xs, ys, optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()
        pbar.set_description(f"loss {loss}")

        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))

def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    
    # Check if we should use CUDA
    use_cuda = torch.cuda.is_available() and not getattr(args, 'cpu_only', False)
    device = "cuda" if use_cuda else "cpu"
    model = model.to(device)
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)