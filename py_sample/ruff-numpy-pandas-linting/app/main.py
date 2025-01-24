from pandera.typing import DataFrame

from app.processor import UserProcessor
from app.schema import Item, User


def main() -> None:
    user_df = DataFrame[User](
        {
            "user_id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "age": [10, 20, 30],
        }
    )
    print(user_df)
    print()

    item_df = DataFrame[Item](
        {
            "item_id": [5, 6, 7],
            "name": ["x", "y", "z"],
            "price": [50.0, 60.0, 70.0],
        }
    )
    print(item_df)
    print()

    print("preprocess:")
    user_processor = UserProcessor()
    user_df, ages = user_processor.transform(user_df)
    print(user_df)
    print(ages)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
