import klib
import torch


if __name__ == '__main__':
    klib.add(torch.randn(1, device="cuda"))
