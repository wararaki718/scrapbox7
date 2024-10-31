from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def main() -> None:
    X, y = make_regression(
        n_features=4,
        n_informative=2,
        random_state=0,
        shuffle=False,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    params = {
        "max_depth": 2, "random_state": 42,
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    print("model trained!")

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    with mlflow.start_run() as run:
        # 1: log_model
        y_pred = model.predict(X_test)
        signature = infer_signature(X_test, y_pred)
        print("model predict!")

        mlflow.log_params(params)
        mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            signature=signature,
            registered_model_name="sk-learn-random-forest-reg-model",
        )
        print("model uploaded!")
        print()

        # 2: api
        model_uri = f"runs:/{run.info.run_id}/sklearn-model"
        result = mlflow.register_model(model_uri, "RandomForestRegressionModel")
        print(f"Name: {result.name}")
        print(f"Version: {result.version}")
        print("load registered model.")
        print()

        # 3: client
        client = mlflow.MlflowClient()
        result = client.create_model_version(
            name="sk-learn-random-forest-reg-model",
            source=f"runs:/{run.info.run_id}/sklearn-model",
            description=f"save: {run.info.run_id}",
        )
        print(result)
        print("model created!")
        print()
    print("DONE")


if __name__ == "__main__":
    main()
