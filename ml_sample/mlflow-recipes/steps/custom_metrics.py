import pandas as pd
from sklearn.metrics import mean_squared_error


def evaluate_metrics(
    eval_df: pd.DataFrame,
    builtin_metrics: dict[str, float],
) -> float:
    return mean_squared_error(
        eval_df.prediction,
        eval_df.target,
        sample_weight=(1.0 / eval_df.prediction),
    )
