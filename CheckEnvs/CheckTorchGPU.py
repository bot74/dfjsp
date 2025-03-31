import torch

# 检查 GPU 是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")          # 使用默认 GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")