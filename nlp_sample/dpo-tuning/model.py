from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def get_model(model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T") -> tuple[PeftModel, LoraConfig]:
    # define config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16",
    )

    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["k_proj", "gate_proj", "v_proj", "up_proj", "q_proj", "o_proj", "down_proj"],
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    return model, peft_config
