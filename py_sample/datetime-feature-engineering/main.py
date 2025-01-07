from datetime import datetime

from preprocessor import DatetimePreprocessor
from vectorizer import DatetimeVectorizer


def main() -> None:
    datetimes = [
        datetime.now(),
        datetime(2024, 2, 2, 1, 1, 1),
        datetime(2025, 2, 2, 1, 1, 1),
    ]
    preprocessor = DatetimePreprocessor()
    data = preprocessor.transform(datetimes)
    print(data)
    print()

    vectorizer = DatetimeVectorizer()
    X = vectorizer.transform(data)
    print(X)
    print(X.shape)

    print("DONE")


if __name__ == "__main__":
    main()
