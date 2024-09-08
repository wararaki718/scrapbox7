import torch

from quantizer import quantize


def main() -> None:
    x = torch.rand((3, 3), dtype=torch.float32)
    print(x)
    print(x.dtype)
    print()

    w, x = quantize(x, 0.1)
    print(f"w = {w}")
    print(x)
    print()

    x = w * x
    print(x)
    print(x.dtype)
    print("DONE")


if __name__ == "__main__":
    main()
