import numpy as np
import pandas as pd


def encode(x: np.ndarray, max_values: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
    x_cos = np.cos(2 * np.pi * x / max_values)
    x_sin = np.sin(2 * np.pi * x / max_values)
    return x_cos.reshape(-1, 1), x_sin.reshape(-1, 1)


class DatetimeVectorizer:
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X_years = np.log(df.year.values).reshape(-1, 1)
        X_cos_month, X_sin_month = encode(df.month.values, 12)
        X_cos_day, X_sin_day = encode(df.day.values, df.month_last_day.values)
        X_cos_hour, X_sin_hour = encode(df.hour.values, 24)
        X_cos_minute, X_sin_minute = encode(df.minute.values, 60)
        X_cos_second, X_sin_second = encode(df.second.values, 60)

        X = np.hstack([
            X_years,
            X_cos_month,
            X_sin_month,
            X_cos_day,
            X_sin_day,
            X_cos_hour,
            X_sin_hour,
            X_cos_minute,
            X_sin_minute,
            X_cos_second,
            X_sin_second,
        ], dtype=np.float32)
        return X
