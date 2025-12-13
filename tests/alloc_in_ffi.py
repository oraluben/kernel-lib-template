import klib
import torch


if __name__ == '__main__':
    print(klib.empty_like(torch.randn(1, device="cuda")))


    d = torch.empty_strided([10, 1], [100, 1], device='cuda')

    d2 = klib.empty_like(d)

    assert (d.shape, d.stride()) == (d2.shape, d2.stride())
