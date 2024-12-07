import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    compute_capability = torch.cuda.get_device_capability(device)
    print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
else:
    print("CUDA device not available.")
