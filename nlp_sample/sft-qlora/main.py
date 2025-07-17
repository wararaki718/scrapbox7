from transformers import TrainingArguments, AutoTokenizer, Pipeline, pipeline
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM

from data import get_data
from model import get_model
from prompt import FormatPrompt
from tokenizer import get_tokenizer
from template import PROMPT_TEMPLATE


def main() -> None:
    # load data
    prompt = FormatPrompt()
    data = get_data(prompt, n_data=3000)
    print(f"Loaded {len(data)} data samples.")

    # load models
    model, peft_config = get_model()
    tokenizer = get_tokenizer()
    print("model and tokenizer loaded.")

    # train
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=512,
        peft_config=peft_config,
    )
    trainer.train()

    model_path = "./models/qlora_model"
    trainer.model.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    # merge
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    merged_model = model.merge_and_unload()
    print("Model merged.")

    pipe = pipeline(
        "text-generation",
        model=merged_model,
        tokenizer=tokenizer,
    )
    result = pipe(PROMPT_TEMPLATE)[0]["generated_text"]
    print(f"Result: {result}")
    

    print("DONE")


if __name__ == "__main__":
    main()
