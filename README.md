# Code for the paper "What Can Transformers Learn In-Context?  A Case Study of Simple Function Classes"
We provide the code we used to train and evaluate all our models. The key entry points
are as follows:
- `train.py` takes as argument a configuration yaml from `conf` and trains the
  corresponding model.
- In `eval_utils.py`, the function `get_run_metrics` reads a training run (produced as
  above) and return all the performance metrics, including baseline evaluation and
  out-of-distribution prompt robutness.
- `function_vis_utils.py` contains functions that compute the function visualizations
  from Figure 3.
