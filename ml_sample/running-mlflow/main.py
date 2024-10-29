import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import generate_data


def connect_mlflow() -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment = mlflow.set_experiment("apple models")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")


def main() -> None:
    # setup
    connect_mlflow()
    np.random.seed(9999)

    # define
    run_name = "apples_rf_test"
    artifact_path = "rf_apples"

    # dataset
    data = generate_data()
    X = data.drop(columns=["date", "demand"]).values
    y = data["demand"].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_valid: {X_valid.shape}")
    print(f"y_valid: {y_valid.shape}")

    # model
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "bootstrap": True,
        "oob_score": False,
        "random_state": 888,
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    print("model trained.")

    # evaluate
    y_pred = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, y_pred)
    mse = mean_squared_error(y_valid, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_valid, y_pred)
    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model, input_example=X_valid, artifact_path=artifact_path
        )

    print("DONE")


if __name__ == "__main__":
    main()
