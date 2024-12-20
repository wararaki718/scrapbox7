import pandas as pd


class Preprocessor:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        _ = df.pop("Time")
        return df
