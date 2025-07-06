from langchain_community.llms.llamacpp import LlamaCpp

from chains.single_prompt import chain_single_prompt
from chains.multi_prompt import chain_multi_prompt


def main() -> None:
    llm = LlamaCpp(
        model_path="models/Phi-3-mini-4k-instruct-fp16.gguf",
        n_gpu_layers=-1,
        max_tokens=500,
        n_ctx=2048,
        seed=42,
        verbose=False,
    )

    # single prompt
    text = "Hi! my name is Maarten. What is 1 + 1?"
    response = chain_single_prompt(text, llm)
    print(f"single prompt: {response}")
    print()

    # multi prompt
    text = "a girl that lost her mother"
    response = chain_multi_prompt(text, llm)
    print(f"multi prompt: {response}")
    print()
    
    print("DONE")


if __name__ == "__main__":
    main()
