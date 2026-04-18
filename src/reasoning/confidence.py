import torch
import numpy as np
from typing import List, Tuple
from src.schemas.response import TokenConfidence, StepConfidence


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

def map_step_confidence(
    steps: List[str],
    final_answer: str,
    tokens: List[str],
    probs: List[float],
) -> tuple[List[StepConfidence], StepConfidence]:

    segments = steps + [final_answer]

    full_text = "".join(tokens)  # no unpacking needed

    segment_spans = []
    cursor = 0
    for seg in segments:
        seg_stripped = seg.strip()
        idx = full_text.find(seg_stripped, cursor)
        if idx == -1:
            segment_spans.append((cursor, cursor))
        else:
            segment_spans.append((idx, idx + len(seg_stripped)))
            cursor = idx + len(seg_stripped)

    segment_token_data: List[List[TokenConfidence]] = [[] for _ in segments]

    char_pos = 0
    for token, prob in zip(tokens, probs):  # zip here instead
        token_mid = char_pos + len(token) / 2

        assigned = False
        for seg_idx, (seg_start, seg_end) in enumerate(segment_spans):
            if seg_start <= token_mid < seg_end:
                segment_token_data[seg_idx].append(TokenConfidence(token=token, prob=prob))
                assigned = True
                break

        if not assigned and segment_token_data:
            nearest = min(
                range(len(segment_spans)),
                key=lambda i: min(
                    abs(token_mid - segment_spans[i][0]),
                    abs(token_mid - segment_spans[i][1]),
                )
            )
            segment_token_data[nearest].append(TokenConfidence(token=token, prob=prob))

        char_pos += len(token)

    def build_step_confidence(text: str, token_confidences: List[TokenConfidence]) -> StepConfidence:
        mean_conf = float(np.mean([t.prob for t in token_confidences])) if token_confidences else 0.0
        return StepConfidence(step=text, tokens=token_confidences, mean_confidence=mean_conf)

    step_confidences = [build_step_confidence(steps[i], segment_token_data[i]) for i in range(len(steps))]
    final_answer_confidence = build_step_confidence(final_answer, segment_token_data[len(steps)])

    return step_confidences, final_answer_confidence