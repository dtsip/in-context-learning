This repository contains the code and data for our pape:

**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: https://arxiv.org/abs/TBD <br>

![](setting.jpg)

```bibtex
    @InProceedings{garg2022what,
        title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
        author={Shivam Garg and Dimitris Tsipras and Percy Liang and Gregory Valiant},
        year={2022},
        booktitle={arXiv preprint}
    }
```

## Getting started
You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    ```

2. Download [model checkpoints](https://github.com/dtsip/in-context-learning/releases/download/v1/models.zip) and extract them in the current directory.

That's it! You can now explore our pre-trained models or train your own. The key entry points
are as follows:
- `train.py` takes as argument a configuration yaml from `conf` and trains the
  corresponding model.
- The `eval.ipynb` notebook contains code to load existing models, plot the pre-computed metrics, and evaluate them on new data.

# Maintainers

* [Shivam Garg](https://cs.stanford.edu/~shivamg/)
* [Dimitris Tsipras](https://dtsipras.com/)
