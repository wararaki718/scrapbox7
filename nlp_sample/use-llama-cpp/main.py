import json

from llama_cpp import Llama


def main() -> None:
    llm = Llama.from_pretrained(
        repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
        filename="*q8_0.gguf",
        chat_format="llama-2",
        verbose=False
    )

    print("## text completion")
    prompt = "Q: Name the planets in the solar system? A: "
    output = llm(
        prompt,
        max_tokens=32,
        stop=["Q:", "\n"],
        echo=True,
    )
    print(json.dumps(output, indent=4))
    print()

    print("## chat")
    messages = [
        {
            "role": "system",
            "content": "You are an assistant who perfectly describes images."
        },
        {
            "role": "user",
            "content": "Describe this image in detail please."
        }
    ]
    output = llm.create_chat_completion(messages)
    print(json.dumps(output, indent=4))
    print()

    print("DONE")


if __name__ == "__main__":
    main()
