from setfit import SetFitModel, TrainingArguments, Trainer

from data import get_data


def main() -> None:
    train_data, test_data = get_data()
    print(train_data.shape)
    print(test_data.shape)

    model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    print("model loaded!")

    training_args = TrainingArguments(
        num_epochs=3,
        num_iterations=20,
        report_to="none",
    )
    training_args.eval_strategy = training_args.evaluation_strategy

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        metric="f1",
    )
    trainer.train()
    print("model trained!")

    print(trainer.evaluate())
    print()

    print("DONE")


if __name__ == "__main__":
    main()
