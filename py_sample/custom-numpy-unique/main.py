import numpy as np

from unique import np_unique


def show(x: np.ndarray) -> None:
    print(x)
    print(f"shape={x.shape}")
    print()


def main() -> None:
    a = np.random.randint(0, 3, 20).reshape((10, 2))
    print("# a (before)")
    show(a)

    b = np.unique(a, axis=0)
    print("# b (np.unique)")
    show(b)

    print("# a (after np.unique)")
    show(a)

    c = np_unique(a, axis=0)
    print(" c (custom)")
    show(c)

    print("# a (after custom)")
    show(a)

    c[0][0] = 1000
    show(c)

    show(a)

    print("DONE")


if __name__ == "__main__":
    main()
