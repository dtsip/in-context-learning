import os
import uuid
import yaml
import argparse
import sys
import tempfile
from quinine import QuinineArgumentParser

from schema import schema as quinine_schema
from train import main as train_main


def prepare_out_dir(args):
    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        # Persist the resolved config for this run (mirrors train.py behaviour)
        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)


def run_one_experiment(
    base_config_path: str,
    task: str,
    task_kwargs: dict,
    data_kwargs: dict,
    run_name: str,
    resume_id: str = None,
    data_type: str = None,
    train_steps: int = None,
    sequence_length: int = None,
):
    """
    Run a single experiment with specified task, task_kwargs, and data_kwargs.
    
    Args:
        base_config_path: Path to base config yaml file
        task: Task name (e.g., 'sparse_linear_regression', 'noisy_linear_regression')
        task_kwargs: Dictionary of task-specific kwargs (e.g., {'noise_type': 'normal', 'sparsity': 3})
        data_kwargs: Dictionary of data sampler kwargs (e.g., {'sparsity': 5})
        run_name: Name for wandb run
        resume_id: Optional resume_id for the run
        data_type: Optional data type override (e.g., 'sparse_gaussian' for sparse data experiments)
    """
    config_dir = os.path.dirname(base_config_path)

    # Read base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Modify config for this experiment
    # Ensure training section exists
    if 'training' not in base_config:
        base_config['training'] = {}
    
    base_config['training']['task'] = task
    base_config['training']['task_kwargs'] = task_kwargs
    base_config['training']['data_kwargs'] = data_kwargs
    if data_type is not None:
        base_config['training']['data'] = data_type
    if resume_id is not None:
        base_config['training']['resume_id'] = resume_id
    if train_steps is not None:
        base_config['training']['train_steps'] = int(train_steps)
    if sequence_length is not None:
        curriculum_points = base_config['training'].setdefault('curriculum', {}).setdefault('points', {})
        curriculum_points['start'] = sequence_length
        curriculum_points['end'] = sequence_length
        curriculum_points['inc'] = 0
        curriculum_points.setdefault('interval', 1)
    
    # Ensure wandb section exists
    if 'wandb' not in base_config:
        base_config['wandb'] = {}
    base_config['wandb']['name'] = run_name

    # Create temporary config file
    temp_config_file = tempfile.NamedTemporaryFile(
        mode='w+t', 
        delete=False, 
        suffix='.yaml',
        dir=config_dir
    )
    
    try:
        # Write modified config to temp file
        yaml.dump(base_config, temp_config_file, default_flow_style=False)
        temp_config_file.close()

        # Parse config using Quinine
        cli_args_list = ["--config", temp_config_file.name]
        qparser = QuinineArgumentParser(schema=quinine_schema)
        original_argv = sys.argv
        try:
            sys.argv = ["run_one_script_placeholder"] + cli_args_list
            args = qparser.parse_quinfig()
        finally:
            sys.argv = original_argv

        # Prepare output directory and run training
        prepare_out_dir(args)
        print(f"\n{'='*60}")
        print(f"Running: {run_name}")
        print(f"Task: {task}")
        print(f"Task kwargs: {task_kwargs}")
        print(f"Data kwargs: {data_kwargs}")
        if data_type is not None:
            print(f"Data type: {data_type}")
        if train_steps is not None:
            print(f"Train steps override: {train_steps}")
        if sequence_length is not None:
            print(f"Sequence length override: {sequence_length}")

        print(f"{'='*60}\n")
        train_main(args)

    finally:
        # Clean up temp file
        if os.path.exists(temp_config_file.name):
            os.remove(temp_config_file.name)


def get_default_experiments():
    """
    Define default experiments for sparse_linear_regression and noisy_linear_regression.
    Returns a list of experiment configs: (task, task_kwargs, data_kwargs, run_name, data_type)
    """
    experiments = []

    # ===== Sparse Linear Regression Experiments =====
    for sparsity in [3, 5, 7]:
        experiments.append({
            "task": "sparse_linear_regression",
            "task_kwargs": {"sparsity": sparsity},
            "data_kwargs": {},
            "run_name": f"sparse_w_sparsity_{sparsity}",
            "data_type": None,
        })

    for data_sparsity in [5, 10, 15]:
        experiments.append({
            "task": "sparse_linear_regression",
            "task_kwargs": {"sparsity": 3},
            "data_kwargs": {"sparsity": data_sparsity},
            "run_name": f"sparse_data_sparsity_{data_sparsity}",
            "data_type": "sparse_gaussian",
        })

    noise_types = [
        "normal",
        "uniform",
        "laplace",
        "t-student",
        "cauchy",
        "exponential",
        "rayleigh",
        "beta",
        "poisson",
    ]

    for noise_type in noise_types:
        experiments.append({
            "task": "noisy_linear_regression",
            "task_kwargs": {"noise_type": noise_type, "noise_std": 2.0},
            "data_kwargs": {},
            "run_name": f"noisy_{noise_type}",
            "data_type": None,
        })

    for noise_std in [0.5, 1.0, 2.0, 3.0]:
        experiments.append({
            "task": "noisy_linear_regression",
            "task_kwargs": {"noise_type": "normal", "noise_std": noise_std},
            "data_kwargs": {},
            "run_name": f"noisy_normal_std_{noise_std}",
            "data_type": None,
        })

    return experiments


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run experiments for sparse_linear_regression and noisy_linear_regression"
    )
    parser.add_argument(
        "--config",
        default="src/conf/template.yaml",
        help="Base config yaml (e.g., src/conf/template.yaml)",
    )
    parser.add_argument(
        "--task",
        choices=["sparse", "noisy", "both", "custom"],
        default="both",
        help="Which task(s) to run: 'sparse', 'noisy', 'both', or 'custom'",
    )
    parser.add_argument(
        "--sparse_w_sparsities",
        nargs="*",
        type=int,
        default=[3, 5, 7],
        help="Weight sparsity values for sparse_linear_regression (w sparsity)",
    )
    parser.add_argument(
        "--sparse_data_sparsities",
        nargs="*",
        type=int,
        default=[5, 10, 15],
        help="Data sparsity values for sparse_linear_regression (data sparsity)",
    )
    parser.add_argument(
        "--noise_types",
        nargs="*",
        default=[
            "normal",
            "uniform",
            "laplace",
            "t-student",
            "cauchy",
            "exponential",
            "rayleigh",
            "beta",
            "poisson",
        ],
        help="Noise types for noisy_linear_regression",
    )
    parser.add_argument(
        "--noise_stds",
        nargs="*",
        type=float,
        default=[0.5, 1.0, 2.0, 3.0],
        help="Noise standard deviations for noisy_linear_regression",
    )
    parser.add_argument(
        "--base_run_name",
        default="sweep",
        help="Base prefix for wandb.name",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=None,
        help="Override training.train_steps for all experiments",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs that already have config.yaml in output directory",
    )
    parser.add_argument(
        "--sequence_lengths",
        nargs="*",
        type=int,
        default=[],
        help="Optional list of sequence lengths (curriculum.n_points) to sweep over",
    )
    return parser


def main():
    parser = build_parser()
    cli_args = parser.parse_args()

    experiments = []
    
    # Build experiment list based on task selection
    if cli_args.task in ["sparse", "both"]:
        # Sparse w experiments (weight sparsity, regular gaussian data)
        for sparsity in cli_args.sparse_w_sparsities:
            experiments.append({
                "task": "sparse_linear_regression",
                "task_kwargs": {"sparsity": sparsity},
                "data_kwargs": {},
                "run_name": f"{cli_args.base_run_name}_sparse_w_{sparsity}",
                "data_type": None,
            })
        
        # Sparse data experiments (sparse_gaussian data)
        for data_sparsity in cli_args.sparse_data_sparsities:
            experiments.append({
                "task": "sparse_linear_regression",
                "task_kwargs": {"sparsity": 3},
                "data_kwargs": {"sparsity": data_sparsity},
                "run_name": f"{cli_args.base_run_name}_sparse_data_{data_sparsity}",
                "data_type": "sparse_gaussian",
            })
    
    if cli_args.task in ["noisy", "both"]:
        # Different noise types
        for noise_type in cli_args.noise_types:
            experiments.append({
                "task": "noisy_linear_regression",
                "task_kwargs": {"noise_type": noise_type, "noise_std": 2.0},
                "data_kwargs": {},
                "run_name": f"{cli_args.base_run_name}_noisy_{noise_type}",
                "data_type": None,
            })
        
        # Different noise_std for normal noise
        for noise_std in cli_args.noise_stds:
            experiments.append({
                "task": "noisy_linear_regression",
                "task_kwargs": {"noise_type": "normal", "noise_std": noise_std},
                "data_kwargs": {},
                "run_name": f"{cli_args.base_run_name}_noisy_normal_std_{noise_std}",
                "data_type": None,
            })
    
    if cli_args.task == "custom":
        default_experiments = get_default_experiments()
        experiments = [
            {
                "task": exp["task"],
                "task_kwargs": exp["task_kwargs"],
                "data_kwargs": exp["data_kwargs"],
                "run_name": f"{cli_args.base_run_name}_{exp['run_name']}",
                "data_type": exp["data_type"],
            }
            for exp in default_experiments
        ]

    if cli_args.sequence_lengths:
        expanded_experiments = []
        for exp in experiments:
            for seq_len in cli_args.sequence_lengths:
                new_exp = dict(exp)
                new_exp["sequence_length"] = seq_len
                new_exp["run_name"] = f"{exp['run_name']}_seq_{seq_len}"
                expanded_experiments.append(new_exp)
        experiments = expanded_experiments
    
    # Run experiments
    print(f"\n{'='*60}")
    print(f"Total experiments to run: {len(experiments)}")
    print(f"{'='*60}\n")
    
    for idx, exp in enumerate(experiments, 1):
        task = exp["task"]
        task_kwargs = exp["task_kwargs"]
        data_kwargs = exp["data_kwargs"]
        run_name = exp["run_name"]
        data_type = exp.get("data_type")
        sequence_length = exp.get("sequence_length")

        print(f"\n[{idx}/{len(experiments)}] Preparing: {run_name}")
        
        # Check if should skip existing
        if cli_args.skip_existing:
            # Try to find existing run by checking base out_dir
            with open(cli_args.config, 'r') as f:
                base_config = yaml.safe_load(f)
            base_out_dir = base_config.get('out_dir', '../models')
            # Handle empty out_dir
            if not base_out_dir or base_out_dir.strip() == '':
                base_out_dir = '../models'
            # Check if any subdirectory has this run_name in config
            if os.path.exists(base_out_dir):
                task_dir = os.path.join(base_out_dir, task)
                if os.path.exists(task_dir):
                    for run_id in os.listdir(task_dir):
                        run_path = os.path.join(task_dir, run_id)
                        config_path = os.path.join(run_path, 'config.yaml')
                        if os.path.exists(config_path):
                            with open(config_path) as f:
                                existing_config = yaml.safe_load(f)
                                if existing_config.get('wandb', {}).get('name') == run_name:
                                    print(f"  -> Skipping (already exists): {run_name}")
                                    continue
        
        # Generate resume_id from run_name (sanitize for filesystem)
        resume_id = run_name.replace(" ", "_").replace("/", "_")
        
        try:
            run_one_experiment(
                cli_args.config,
                task,
                task_kwargs,
                data_kwargs,
                run_name,
                resume_id=resume_id,
                data_type=data_type,
                train_steps=cli_args.train_steps,
                sequence_length=sequence_length,
            )
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"ERROR in experiment: {run_name}")
            print(f"Error: {str(e)}")
            print(f"{'!'*60}\n")
            # Continue with next experiment
            continue
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
