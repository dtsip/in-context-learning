import torch


SEQ_MODELS = ["gpt2", "lstm", "relu_attn"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available() and model.name.split("_")[0] in SEQ_MODELS: device = "cuda"