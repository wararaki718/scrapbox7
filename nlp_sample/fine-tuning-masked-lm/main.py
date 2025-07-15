from data import get_data
from evaluate import evaluate
from model import load_models

from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

def main() -> None:
    # load data
    train_data, test_data = get_data()
    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    print()

    # load models
    tokenizer, model = load_models()
    print("model loaded!")

    tokenize = lambda x: tokenizer(x["text"], truncation=True)
    train_data = train_data.map(tokenize, batched=True)
    test_data = test_data.map(tokenize, batched=True)
    print("Data tokenized!")

    # set training arguments
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        "model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        save_strategy="epoch",
        report_to="none",
    )

    # training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    print("model trained")

    # evaluate
    evaluate(model, tokenizer, "What a horrible [MASK]!")

    print("DONE")


if __name__ == "__main__":
    main()
