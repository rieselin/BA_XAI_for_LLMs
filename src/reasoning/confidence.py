import torch
import numpy as np


SPECIAL_TOKENS = {"<|eot_id|>", "<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>"}

def compute_token_confidence(output, tokenizer):
    input_len = output.sequences.shape[1] - len(output.scores)
    generated_ids = output.sequences[0][input_len:]

    token_list = []
    probs = []

    for i, score in enumerate(output.scores):
        probs_dist = torch.softmax(score[0], dim=-1)
        token_id = generated_ids[i]
        token_str = tokenizer.decode(token_id)

        if token_str in SPECIAL_TOKENS:
            continue  # skip special tokens entirely

        prob = probs_dist[token_id].item()
        token_list.append(token_str)
        probs.append(prob)

    return token_list, probs


def aggregate_confidence(probs):
    if not probs:
        return {"mean": 0.0, "min": 0.0}

    return {
        "mean": float(np.mean(probs)),
        "min": float(np.min(probs))
    }


def map_step_confidence(steps, token_probs, tokens):
    if not steps:
        return []
    
    total_chars = sum(len(s) for s in steps)
    if total_chars == 0:
        return [0.0] * len(steps)
    
    step_conf = []
    prob_idx = 0
    total_tokens = len(token_probs)
    
    for step in steps:
        # Allocate tokens proportional to this step's character length
        proportion = len(step) / total_chars
        count = max(1, round(proportion * total_tokens))
        chunk = token_probs[prob_idx:prob_idx + count]
        step_conf.append(float(np.mean(chunk)) if chunk else 0.0)
        prob_idx += count
    
    return step_conf