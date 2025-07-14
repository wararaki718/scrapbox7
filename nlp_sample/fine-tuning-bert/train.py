from typing import Callable
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)


def get_trainer(
    model: AutoModelForSequenceClassification,
    data_collator: DataCollatorWithPadding,
    train_dataset: Dataset,
    test_dataset: Dataset,
    compute_metrics: Callable,
) -> Trainer:
    # training arguments
    training_args = TrainingArguments(
        "model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        save_strategy="epoch",
        report_to="none",
    )

    # define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer


if __name__ == "__main__":
    main()
