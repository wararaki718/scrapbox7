from datetime import datetime

import numpy as np


def encode(x: np.ndarray, max_value: int) -> tuple[np.ndarray, np.ndarray]:
    x_cos = np.cos(2 * np.pi * x / 12)
    x_sin = np.sin(2 * np.pi * x / 12)
    return x_cos.reshape(-1, 1), x_sin.reshape(-1, 1)


class DatetimeVectorizer:
    def transform(self, dts: list[datetime]) -> np.ndarray:
        years = np.array([dt.year for dt in dts])
        months = np.array([dt.month for dt in dts])
        days = np.array([dt.day for dt in dts])
        hours = np.array([dt.hour for dt in dts])
        minutes = np.array([dt.minute for dt in dts])
        seconds = np.array([dt.second for dt in dts])

        X_years = np.log(years).reshape(-1, 1)
        X_cos_month, X_sin_month = encode(months, 12)
        X_cos_day, X_sin_day = encode(days, 31)
        X_cos_hour, X_sin_hour = encode(hours, 31)
        X_cos_minute, X_sin_minute = encode(minutes, 31)
        X_cos_second, X_sin_second = encode(seconds, 31)

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
