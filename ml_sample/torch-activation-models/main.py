import torch

from model import (
    SigmoidModel,
    SoftmaxModel,
    ReLUModel,
    SoftplusModel,
)


def show(model: torch.nn.Module, X: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        y: torch.Tensor = model(X)
    print(y.shape)
    print(y)
    print()


def main() -> None:
    n_input = 16
    n_output = 4

    X = torch.rand(5, n_input)
    print(X)
    print()

    sigmoid_model = SigmoidModel(n_input, n_output)
    softmax_model = SoftmaxModel(n_input, n_output)
    relu_model = ReLUModel(n_input, n_output)
    softplus_model = SoftplusModel(n_input, n_output)

    print("sigmoid:")
    show(sigmoid_model, X)

    print("softmax:")
    show(softmax_model, X)

    print("relu:")
    show(relu_model, X)

    print("softplus:")
    show(softplus_model, X)
    
    # activate
    print("## activate only")
    X = torch.rand(4, 5) - 0.5
    print(X)
    print()

    print("sigmoid:")
    sigmoid = torch.nn.Sigmoid()
    show(sigmoid, X)

    print("softmax:")
    softmax = torch.nn.Softmax()
    show(softmax, X)

    print("relu:")
    relu = torch.nn.ReLU()
    show(relu, X)

    print("softplus:")
    softplus = torch.nn.Softplus()
    show(softplus, X)

    print("DONE")


if __name__ == "__main__":
    main()
