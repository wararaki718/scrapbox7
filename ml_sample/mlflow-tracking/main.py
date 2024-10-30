import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def connect_mlflow() -> None:
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment("MLflow Quickstart")


def main() -> None:
    connect_mlflow()


    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)
    print("model trained")

    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy: {accuracy}")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        signature = infer_signature(X_train, lr.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )
        print("model uploaded")
    
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = loaded_model.predict(X_test)
    print(f"model loaded from mlflow. ({model_info.model_uri})")
    
    iris_feature_names = load_iris().feature_names

    result = pd.DataFrame(X_test, columns=iris_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    print(result[:4])
    print("DONE")


if __name__ == "__main__":
    main()
