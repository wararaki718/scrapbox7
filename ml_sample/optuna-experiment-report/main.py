import optuna


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


def main() -> None:
    study: optuna.Study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print()

    print(f"study_name: {study.study_name}")
    print(f"best_param: {study.best_params}")
    print(f"best_value: {study.best_value}")
    print(f"best_trial: {study.best_trial}")
    print(study.best_trial.__dict__)
    print(study.trials_dataframe())

    print("DONE")


if __name__ == "__main__":
    main()
