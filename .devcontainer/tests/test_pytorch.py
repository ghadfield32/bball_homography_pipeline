print("PyTorch quick check")
try:
    import torch
    print("version:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("devices:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i))
        x = torch.ones(100, 100, device='cuda:0')
        print("sum:", float(torch.sum(x)))
except Exception as e:
    print("error:", e)
