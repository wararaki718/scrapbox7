from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch


def load_model(
    model_name: str = "Salesforce/blip2-opt-2.7b",
    revision: str = "51572668da0eb669e01a189dc22abe6088589a24",
    device: str = "cuda",
) -> tuple[Blip2ForConditionalGeneration, AutoProcessor]:
    # load model and processor
    processor = AutoProcessor.from_pretrained(model_name, revision=revision)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, revision=revision, torch_dtype=torch.float16)
    
    # gpu setting
    model.to(device)

    return model, processor
