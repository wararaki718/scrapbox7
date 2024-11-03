import pandas as pd


def load_file_as_dataframe(file_path: str, file_format: str) -> pd.DataFrame:
    return pd.read_parquet(file_path)
