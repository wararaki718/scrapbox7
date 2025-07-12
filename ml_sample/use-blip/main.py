import torch
from model import load_model
from tasks import task_caption, task_conversation
from utils import get_image


def main() -> None:
    image = get_image()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(device=device)

    # caption
    generated_text = task_caption(model, processor, image)
    print(f"Caption: {generated_text}")

    # conversation
    prompt = "Question: Write down what you see in this picture. Answer:"
    generated_text = task_conversation(model, processor, image, prompt)
    print(f"Conversation: {generated_text}")

    # conversation with a different prompt
    prompt = "Question: Write down what you see in this picture. Answer: A sports car driving on the road at sunset. Question: What would it cost me to drive that car? Answer:"
    generated_text = task_conversation(model, processor, image, prompt)
    print(f"Conversation with different prompt: {generated_text}")

    print("DONE")


if __name__ == "__main__":
    main()
