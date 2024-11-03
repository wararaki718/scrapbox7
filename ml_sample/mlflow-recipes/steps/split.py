import pandas as pd


def create_dataset_filter(dataset: pd.DataFrame) -> pd.Series:
    return (
        (dataset["fare_amount"] > 0)
        & (dataset["trip_distance"] < 400)
        & (dataset["trip_distance"] > 0)
        & (dataset["fare_amount"] < 1000)
    ) | (
        ~dataset.isna().any(axis=1)
    )
