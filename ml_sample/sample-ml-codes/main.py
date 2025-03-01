from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data import get_data
from evaluate import evaluate


def main() -> None:
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=10,
    )
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print()

    print("train:")
    hgb = HistGradientBoostingClassifier(max_depth=5, random_state=1)
    rfc = RandomForestClassifier(max_depth=5, random_state=1)
    lr = LogisticRegression()

    hgb.fit(X_train, y_train)
    rfc.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    print("model trained.")

    print("Hist Gradient Boosting:")
    results = evaluate(hgb, X_test, y_test)
    print(results[2], results[5])

    print("Random Forest:")
    results = evaluate(rfc, X_test, y_test)
    print(results[2], results[5])

    print("Logistic Regression:")
    results = evaluate(lr, X_test, y_test)
    print(results[2], results[5])

    print("DONE")


if __name__ == "__main__":
    main()
