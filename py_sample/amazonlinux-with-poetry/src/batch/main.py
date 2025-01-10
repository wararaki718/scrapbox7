import torch


def main() -> None:
    x = torch.Tensor([1, 2, 3])
    print(x.shape)
    print(torch.cuda.is_available())
    print("DONE")


if __name__ == "__main__":
    main()
