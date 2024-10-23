import numpy as np


def main() -> None:
    a = np.random.randint(0, 5, 6).reshape(3, 2)
    b = np.ascontiguousarray(a)
    print("before: ascontiguousarray")
    print(a)
    print(b)
    print()

    print("after: ascontiguousarray")
    a[0][0] = 100
    print(a)
    print(b)
    print()

    b = np.moveaxis(a, 1, 0)
    print("before: moveaxis")
    print(a)
    print(b)
    print()

    print("after: moveaxis")
    a[0][0] = 200
    print(a)
    print(b)
    print()


    print("DONE")


if __name__ == "__main__":
    main()
