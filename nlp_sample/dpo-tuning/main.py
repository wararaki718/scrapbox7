from transformers import TrainingArguments
from trl import SFTTrainer, DPOConfig, DPOTrainer
from peft import AutoPeftModelForCausalLM

from data import get_data
from model import get_model
from tokenizer import get_tokenizer


def main() -> None:
    # load data
    data = get_data()
    print(f"Loaded {len(data)} data samples.")

    # load models
    model, peft_config = get_model()
    tokenizer = get_tokenizer()
    print("model and tokenizer loaded.")

    # train
    training_args = DPOConfig(
        output_dir="./results",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=1e-5,
        logging_steps=10,
        max_steps=200,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        warmup_ratio=0.1,
    )
    trainer = DPOTrainer(
        model=model,
        train_dataset=data,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_length=512,
        beta=0.1,
        peft_config=peft_config,
        max_prompt_length=512,
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

   
    

    print("DONE")


if __name__ == "__main__":
    main()
