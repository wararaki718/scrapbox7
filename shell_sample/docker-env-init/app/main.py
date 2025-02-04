import os


def main() -> None:
    value = os.getenv("SAMPLE_ENV_VALUE")
    print(value)
    print("DONE")


if __name__ == "__main__":
    main()
