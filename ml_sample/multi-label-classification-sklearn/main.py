from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def main() -> None:
    print("dataset:")
    X, y = make_multilabel_classification(n_samples=5000, n_features=100)
    print(X.shape)
    print(y.shape)
    print()

    print("train test dataset:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print()

    print("model train:")
    svc = OneVsRestClassifier(estimator=SVC(kernel="linear"))
    svc.fit(X_train, y_train)
    print("svc trained!")

    lr = OneVsRestClassifier(estimator=LogisticRegression())
    lr.fit(X_train, y_train)
    print("lr trained!")

    cb = OneVsRestClassifier(
        estimator=CalibratedClassifierCV(
            estimator=LogisticRegression()
        )
    )
    cb.fit(X_train, y_train)
    print("cb trained!")

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    print("rf trained!")
    print()

    print("evaluation:")
    print(f"- svc score={svc.score(X_test, y_test)}")
    print(f"- lr score ={lr.score(X_test, y_test)}")
    print(f"- cb score ={cb.score(X_test, y_test)}")
    print(f"- rf score ={rf.score(X_test, y_test)}")
    print()

    print("DONE")


if __name__ == "__main__":
    main()
