import torch


SEQ_MODELS = ["gpt2", "lstm", "relu_attn"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"