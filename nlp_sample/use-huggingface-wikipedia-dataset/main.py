from datasets import load_dataset

from text import TextProcessor

def main() -> None:
    data = load_dataset("range3/wikipedia-ja-20230101")
    train = data["train"]

    processor = TextProcessor()

    print(f"train: {len(train)}")
    for title, text in zip(train["title"][:3], train["text"][:3]):
        print("#-----")
        print(title)
        results = processor.transform(text)
        print(f"words: {results}")
        print(f"number of sentences: {len(results)}")
        print()
    print("DONE")


if __name__ == "__main__":
    main()
