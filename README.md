his repository contains the code and models for our paper:

## Getting started
You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. To train a model, populate `conf/wandb.yaml` with you wandb info.

You can explore our pre-trained models or train your own. The key entry points
are as follows (starting from `src`):
- The `eval.ipynb` notebook contains code to load our own pre-trained models, plot the pre-computed metrics, and evaluate them on new data. 
This notebook also contains code to train a toy transformer with randomized labels. In the code, you can specify the number of indices to randomize, observe the output, and plot the losses.

- `train.py` takes as argument a configuration yaml from `conf` and trains the corresponding model. You can try `python train.py --config conf/seq_linear.yaml` for a quick training run for our recursive linear function.
To train on a sliding window sequential task, you can run `python train.py --config conf/seq_relu_2nn.yaml`


Most of the code in our codebase comes from
**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br><br>

![](setting.jpg)

```bibtex
    @InProceedings{garg2022what,
        title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
        author={Shivam Garg and Dimitris Tsipras and Percy Liang and Gregory Valiant},
        year={2022},
        booktitle={arXiv preprint}
    }
```
