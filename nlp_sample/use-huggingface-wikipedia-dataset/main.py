from datasets import load_dataset


def main() -> None:
    data = load_dataset("range3/wikipedia-ja-20230101")
    train = data["train"]

    print(f"train: {len(train)}")
    for title, text in zip(train["title"][:5], train["text"][:5]):
        print("#-----")
        print(title)
        print(text[:100])
        print()
    print("DONE")


if __name__ == "__main__":
    main()
