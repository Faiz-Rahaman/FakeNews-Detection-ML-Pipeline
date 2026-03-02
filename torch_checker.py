import torch

# Check if CUDA (GPU) is available
print(f"CUDA available: {torch.cuda.is_available()}")

# If available, print GPU details
if torch.cuda.is_available():
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("GPU not available - using CPU")