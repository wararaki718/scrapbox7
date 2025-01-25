from typing import cast

import pandas as pd
from pandera.typing import DataFrame

from app.processor import CustomUserProcessor, UserProcessor
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

    print("custom process:")
    custom_processor = CustomUserProcessor()
    user_df = custom_processor.transform(user_df)
    print(user_df)
    print()

    # error pyright
    # tmp_df = pd.read_csv("data/users.csv")

    # ok
    tmp_df = cast(DataFrame[User], pd.read_csv("data/users.csv"))
    print(tmp_df)
    print()

    print("custom2:")
    tmp_df = custom_processor.transform(tmp_df)
    print(tmp_df)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
