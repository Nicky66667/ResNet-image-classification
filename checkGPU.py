import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#if gpu is available, device will be set to "cuda", instand "cpu"
print(f"Using device: {device}")