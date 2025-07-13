from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from data import get_data


def main() -> None:
    # load dataset
    train_dataset, valid_dataset, scores = get_data()
    print(len(train_dataset))
    print()

    # model
    model = SentenceTransformer("bert-base-uncased")
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=3,
    )

    # evaluator
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=valid_dataset["sentence1"],
        sentences2=valid_dataset["sentence2"],
        scores=scores,
        main_similarity="cosine",
    )

    # args
    training_args = SentenceTransformerTrainingArguments(
        output_dir="base_embedding_model",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        fp16=True,
        eval_steps=100,
        logging_steps=100,
        report_to="none",
    )
    print("model defined!")

    # trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train()
    print("model trained!")
    print()

    # evaluate
    print(evaluator(model=model))
    print()

    print("DONE")


if __name__ == "__main__":
    main()
