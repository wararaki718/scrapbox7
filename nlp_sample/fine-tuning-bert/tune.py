from transformers import AutoModelForSequenceClassification


def freeze_task_layer(model: AutoModelForSequenceClassification) -> AutoModelForSequenceClassification:
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def freeze_encode_layer(model: AutoModelForSequenceClassification) -> AutoModelForSequenceClassification:
    for index, (name, param) in enumerate(model.named_parameters()):
        if index < 165:
            param.requires_grad = False
    return model
