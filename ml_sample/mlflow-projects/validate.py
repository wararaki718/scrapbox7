from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("filepath", default="default", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"filepath: {args.filepath}")
    print("validate")


if __name__ == "__main__":
    main()
