import torch


def main() -> None:
    x = torch.FloatTensor([
        [0.12, 0.34, 0.56],
        [0.56, 0.78, 0.79],
        [0.12, 0.34, 0.55],
    ])
    print(x)
    print()

    scale = 0.1
    zero_point = 0
    y = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
    print(y)
    print()
    print("DONE")


if __name__ == "__main__":
    main()
