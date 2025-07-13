import nltk
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from data import get_data
from model import load_model


def main() -> None:
    nltk.download("punkt")

    # load data
    train_dataset, valid_dataset, valid_scores = get_data()
    print(f"train_dataset: {len(train_dataset)}")
    print(f"valid_dataset: {len(valid_dataset)}")
    print(f"valid_scores: {len(valid_scores)}")
    print()

    # define evaluator
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=valid_dataset["sentence1"],
        sentences2=valid_dataset["sentence2"],
        scores=valid_scores,
        main_similarity="cosine"
    )
    print("evaluator defined!")

    # define model
    model = load_model()
    print("model defined!")

    # define loss
    train_loss = losses.DenoisingAutoEncoderLoss(model=model, tie_encoder_decoder=True)
    train_loss.decoder = train_loss.decoder.to("cuda")
    print("loss defined!")

    # define args
    train_args = SentenceTransformerTrainingArguments(
        output_dir="tsdae_embedding_model",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        fp16=True,
        eval_steps=100,
        logging_steps=100,
    )
    print("args defined!")

    # define trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        train_loss=train_loss,
        evaluator=evaluator
    )
    trainer.train()
    print("model trained!")
    print()

    # evaluate
    print(evaluator(model))
    print()

    print("DONE")