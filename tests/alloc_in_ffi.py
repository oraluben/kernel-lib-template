import klib
import torch


if __name__ == '__main__':
    print(klib.alloc(torch.randn(1, device="cuda")))
