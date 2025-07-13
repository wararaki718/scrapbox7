import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration


def task_caption(
    model: Blip2ForConditionalGeneration,
    processor: AutoProcessor,
    image: Image,
) -> str:
    inputs = processor(images=image, return_tensors="pt").to(model.device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text[0].strip()


def task_conversation(
    model: Blip2ForConditionalGeneration,
    processor: AutoProcessor,
    image: Image,
    prompt: str,
) -> str:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text[0].strip()
