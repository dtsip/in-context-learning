This repository contains forked code from the paper for the [Berkeley CS 182](https://inst.eecs.berkeley.edu/~cs182/fa23/) Course Project:

**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br><br>


## Getting started
You can start by cloning the repo and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup. *(we did not)*

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. Start training with:

    ```
    cd src/
    python train.py --config conf/any-config_relu_attn.yml
    ```
    Note: we have only implemented ReLU-attention as described in the [ViT-relu](https://arxiv.org/pdf/2309.08586.pdf) paper

## Additional Info

- You can find our (original) proposal at [`reports/proposal.pdf`](https://github.com/nelson-lojo/in-context-learning/blob/main/reports/proposal.pdf) and our initial submission at [`reports/draft_1.pdf`](https://github.com/nelson-lojo/in-context-learning/blob/main/reports/proposal.pdf)
- We are working on getting this code to run on Kaggle and Colab (update: Kaggle works!)
