import torch
from src.core.model_loader import get_model


def generate_with_scores(prompt: str, min_new_tokens: int = 1):
    model, tokenizer = get_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

    return output, tokenizer