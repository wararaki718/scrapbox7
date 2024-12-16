from datetime import datetime

from vectorizer import DatetimeVectorizer


def main() -> None:
    data = [
        datetime.now(),
        datetime(2024, 1, 1, 1, 1, 1),
    ]
    vectorizer = DatetimeVectorizer()
    X = vectorizer.transform(data)
    print(X)
    print(X.shape)

    print("DONE")


if __name__ == "__main__":
    main()
