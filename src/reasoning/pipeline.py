from src.core.prompt_builder import build_reasoning_prompt, build_step_regen_prompt
from src.core.generation import generate_with_scores
from src.reasoning.cot import extract_cot
from src.reasoning.confidence import (
    compute_token_confidence,
    aggregate_confidence,
    map_step_confidence,
)
from src.schemas.request import ReasoningRequest, StepRegenRequest
from src.schemas.response import ReasoningResponse, TokenConfidence, ConfidenceSummary
import numpy as np


def _generate_single_step(user_input: str, prior_steps: list[str], step_number: int):
    """Generate one step and return (step_text, confidence, tokens, probs)."""
    prompt = build_step_regen_prompt(user_input, prior_steps, step_number)
    output, tokenizer = generate_with_scores(prompt)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    generated_ids = output.sequences[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    text = f"[STEP {step_number}]\n" + generated_text
    steps, _ = extract_cot(text)

    step_text = steps[0] if steps else generated_text.strip()

    tokens, probs = compute_token_confidence(output, tokenizer)
    conf = float(np.mean(probs)) if probs else 0.0

    return step_text, conf, tokens, probs


def run_step_regen_pipeline(request: StepRegenRequest):
    """Regenerate a single step, retrying if below threshold."""
    best_text, best_conf, best_tokens, best_probs = None, -1.0, [], []

    for attempt in range(request.max_attempts):
        step_text, conf, tokens, probs = _generate_single_step(
            request.input, request.prior_steps, request.step_number
        )

        if conf > best_conf:
            best_text, best_conf, best_tokens, best_probs = step_text, conf, tokens, probs

        if request.threshold is None or conf >= request.threshold:
            break

    token_objs = [TokenConfidence(token=t, prob=p) for t, p in zip(best_tokens, best_probs)]

    return ReasoningResponse(
        final_answer="",
        steps=[best_text],
        confidence=ConfidenceSummary(
            mean=float(np.mean(best_probs)) if best_probs else 0.0,
            min=float(np.min(best_probs)) if best_probs else 0.0,
        ),
        step_confidence=[best_conf],
        tokens=token_objs,
    )


def run_reasoning_pipeline(request: ReasoningRequest):
    prompt = build_reasoning_prompt(request.input)
    output, tokenizer = generate_with_scores(prompt)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    generated_ids = output.sequences[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    text = "[STEP 1]\n" + generated_text

    steps, final_answer = extract_cot(text)
    tokens, probs = compute_token_confidence(output, tokenizer)
    step_confs = map_step_confidence(steps, probs, tokens)

    if request.threshold is not None:
        for i, (step, conf) in enumerate(zip(steps, step_confs)):
            if conf < request.threshold:
                regen_req = StepRegenRequest(
                    input=request.input,
                    prior_steps=steps[:i],
                    step_number=i + 1,
                    max_attempts=request.max_attempts,
                    threshold=request.threshold,
                )
                result = run_step_regen_pipeline(regen_req)
                steps[i] = result.steps[0]
                step_confs[i] = result.step_confidence[0]

    token_objs = [TokenConfidence(token=t, prob=p) for t, p in zip(tokens, probs)]

    return ReasoningResponse(
        final_answer=final_answer,
        steps=steps,
        confidence=ConfidenceSummary(
            mean=float(np.mean(probs)) if probs else 0.0,
            min=float(np.min(probs)) if probs else 0.0,
        ),
        step_confidence=step_confs,
        tokens=token_objs,
    )


def run_final_regen_pipeline(request: dict):
    user_input = request["input"]
    steps = request["steps"]

    prior = ""
    for i, s in enumerate(steps, 1):
        prior += f"[STEP {i}]\n{s}\n\n"

    # Prompt ends mid-sentence so the model is forced to continue,
    # rather than ending at [FINAL] which can look like a terminus.
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise reasoning engine. Output ONLY the final answer text, no tags.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_input}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{prior}[FINAL]
The answer is:""".strip()

    output, tokenizer = generate_with_scores(prompt, min_new_tokens=5)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    generated_ids = output.sequences[0][prompt_len:]
    final = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Strip any [FINAL] tag the model may have echoed
    if final.lower().startswith("[final"):
        final = final.split("]", 1)[-1].strip()

    if not final:
        final = "No answer generated."

    tokens, probs = compute_token_confidence(output, tokenizer)

    token_objs = [TokenConfidence(token=t, prob=p) for t, p in zip(tokens, probs)]

    return ReasoningResponse(
        final_answer=final,
        steps=[],
        confidence=ConfidenceSummary(
            mean=float(np.mean(probs)) if probs else 0.0,
            min=float(np.min(probs)) if probs else 0.0,
        ),
        step_confidence=[],
        tokens=token_objs,
    )