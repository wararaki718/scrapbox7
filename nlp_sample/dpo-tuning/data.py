from datasets import Dataset, load_dataset


def _format_prompt(example: dict) -> dict:
    system = "<|system|>\n" + example["system"] + "</s>\n"
    prompt = "<|user|>\n" + example["input"] + "</s>\n<|assistant|>\n"
    chosen = example["chosen"] + "</s>\n"
    rejected = example["rejected"] + "</s>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def get_data() -> Dataset:
    dpo_dataset = load_dataset(
        "argilla/distilabel-intel-orca-dpo-pairs", split="train"
    )
    dpo_dataset = dpo_dataset.filter(
        lambda r: r["status"] != "tie" and r["chosen_score"] >= 8 and not r["in_gsm8k_train"]
    )
    dpo_dataset = dpo_dataset.map(
        _format_prompt, remove_columns=dpo_dataset.column_names
    )
    return dpo_dataset
