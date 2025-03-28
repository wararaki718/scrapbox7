import thunder
import torch

from model import SampleModel


def main() -> None:
    model = SampleModel()
    thunder_model = thunder.compile(model)

    X = torch.randn(2048, 512)
    y = model(X)
    y_thunder = thunder_model(X)
    print(y.shape)
    print(y_thunder.shape)

    torch.testing.assert_close(y, y_thunder)
    print("DONE")


if __name__ == "__main__":
    main()
