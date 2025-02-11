import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def show_shap(feature_names: np.ndarray, X: np.ndarray, y: np.ndarray) -> None:
    print("# show shap")
    shap.initjs()

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model)
    shap_test = explainer(X_test)
    shap_df = pd.DataFrame(shap_test.values, columns=feature_names)
    results: pd.Series = shap_df.apply(np.abs).mean().sort_values(ascending=False)
    print(results)
    print()
