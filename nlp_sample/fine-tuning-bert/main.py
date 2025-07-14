from transformers import DataCollatorWithPadding

from data import get_data
from evaluation import compute_metrics
from model import load_model
from train import get_trainer
from tune import freeze_task_layer, freeze_encode_layer


def main() -> None:
    # load data
    train_dataset, test_dataset = get_data()
    print(train_dataset.shape)
    print(test_dataset.shape)
    print()

    # load model
    model_name = "bert-base-cased"
    model, tokenizer = load_model(model_name)
    print(f"load '{model_name}'.")
    print()

    # preprocess
    tokenize = lambda x: tokenizer(x["text"], truncation=True)
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    print(train_dataset.shape)
    print(test_dataset.shape)
    print()

    # get trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = get_trainer(
        model=model,
        data_collator=data_collator,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    print(trainer.evaluate())
    print()

    # train with freeze task-layer
    model, tokenizer = load_model(model_name)
    model = freeze_task_layer(model)
    trainer.train()
    print("freeze task layer:")
    print(trainer.evaluate())
    print()

    # train with freeze encoder
    model, tokenizer = load_model(model_name)
    model = freeze_encode_layer(model)
    trainer.train()
    print("freeze encode layer:")
    print(trainer.evaluate())
    print()

    print("DONE")


if __name__ == "__main__":
    main()
