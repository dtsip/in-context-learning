import torch


SEQ_MODELS = ["gpt2", "lstm", "relu_attn", "nystrom"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
