import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def main() -> None:
    # load model
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    print("model loaded!")
    
    # pipeline define
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=512,
        do_sample=False,
    )
    print("pipeline defined!")

    # generate text
    prompt = [
        {"role": "user", "content": "Create a funny joke about chickens."},
    ]
    output = pipe(prompt)
    print(output[0]["generated_text"])
    print("DONE")


if __name__ == "__main__":
    main()
