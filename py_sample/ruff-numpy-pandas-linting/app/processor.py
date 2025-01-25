from typing import cast

import numpy as np
from pandera.typing import DataFrame, Series

from app.schema import CustomUser, User


class UserAgeProcessor:
    def transform(self, ages: Series[int]) -> np.ndarray:
        # error: pyright
        # return ages.apply(lambda age: age + 100).values

        # ok
        return ages.apply(lambda age: age + 100).to_numpy(copy=False)


class UserProcessor:
    def __init__(self) -> None:
        self._age_processor = UserAgeProcessor()

    def transform(self, user_df: DataFrame[User]) -> tuple[DataFrame[User], np.ndarray]:
        # error: pyright
        # ages = self._age_processor.transform(user_df["age"])

        # ok
        ages = self._age_processor.transform(user_df.age)
        return user_df, ages


class CustomUserProcessor:
    def transform(self, user_df: DataFrame[User]) -> DataFrame[CustomUser]:
        customs = ["custom" for _ in range(len(user_df))]
        custom_df = user_df.assign(custom=customs)

        # error pyright
        # return custom_df

        # ok
        return cast(DataFrame[CustomUser], custom_df)
