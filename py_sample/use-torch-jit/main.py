import torch


class MyCell(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._layer = torch.nn.Linear(4, 4)
        self._activate = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_ = self._layer(x)
        h_ = self._activate(x_ + h)
        return h_, h_


def main() -> None:
    my_cell = MyCell()
    x = torch.randn(3, 4)
    h = torch.randn(3, 4)

    traced_cell = torch.jit.trace(my_cell, (x, h))
    scripted_cell = torch.jit.script(my_cell)

    print("## cell diff")
    print("[traced_cell]")
    print(traced_cell)
    print()
    print("[scripted_cell]")
    print(scripted_cell)
    print()

    print("## graph diff")
    print("[traced_cell]")
    print(traced_cell.graph)
    print("[scripted_cell]")
    print(scripted_cell.graph)
    print()

    print("## code diff")
    print("[traced_cell]")
    print(traced_cell.code)
    print("[scripted_cell]")
    print(scripted_cell.code)
    print()

    print("## output")
    print("[traced_cell]")
    print(traced_cell(x, h))
    print()
    print("[scripted_cell]")
    print(scripted_cell(x, h))
    print()

    print("DONE")


if __name__ == "__main__":
    main()
