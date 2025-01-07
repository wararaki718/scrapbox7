import calendar
from datetime import datetime

import numpy as np
import pandas as pd


class DatetimePreprocessor:
    def transform(self, datetimes: list[datetime]) -> pd.DataFrame:
        years = np.array([dt.year for dt in datetimes])
        months = np.array([dt.month for dt in datetimes])
        days = np.array([dt.day for dt in datetimes])
        hours = np.array([dt.hour for dt in datetimes])
        minutes = np.array([dt.minute for dt in datetimes])
        seconds = np.array([dt.second for dt in datetimes])
        month_last_days = list(map(lambda dt: calendar.monthrange(dt.year, dt.month)[1], datetimes))
        
        df = pd.DataFrame(
            {
                "year":  years,
                "month": months,
                "day": days,
                "hour": hours,
                "minute": minutes,
                "second": seconds,
                "month_last_day": month_last_days,
            }
        )
        return df
