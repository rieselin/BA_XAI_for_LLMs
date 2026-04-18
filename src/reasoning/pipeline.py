from src.core.prompt_builder import build_reasoning_prompt, build_step_regen_prompt, build_final_regen_prompt
from src.core.generation import generate_with_scores
from src.reasoning.cot import extract_cot
from src.reasoning.confidence import (
    compute_token_confidence,
    map_step_confidence,
)
from src.schemas.request import ReasoningRequest, StepRegenRequest, FinalRegenRequest
from src.schemas.response import ReasoningResponse, TokenConfidence, RegenerationType

import numpy as np

def run_step_regen_pipeline(request: StepRegenRequest) -> ReasoningResponse:
    steps = list(request.steps)
    step_confs = list(request.step_confidences)
    step_regenerated = list(request.step_regenerated)
    i = request.step_to_regenerate_index
    final_answer = request.final_answer
    final_answer_conf = request.final_answer_confidence
    final_regenerated = request.final_regenerated

    for attempt in range(request.max_attempts):
        # Build prompt with all steps prior to the one being regenerated
        prior_steps = steps[:i]
        prompt = build_step_regen_prompt(request.input, prior_steps, step_number=i + 1)
        output, tokenizer = generate_with_scores(prompt)

        prefix = f"[STEP {i + 1}]\n"
        generated_text = _decode_generation(prompt, output, tokenizer, prefix=prefix)

        # extract_cot returns all steps + final; we only want the regenerated step
        new_steps, new_final = extract_cot(generated_text)

        if not new_steps:
            continue

        new_step_text = new_steps[0]
        tokens, probs = compute_token_confidence(output, tokenizer)
        new_step_confs, new_final_conf = map_step_confidence(
            [new_step_text], new_final, tokens, probs
        )
        new_conf = new_step_confs[0]

        # Accept if above threshold or this is the last attempt
        if new_conf.mean_confidence >= request.threshold or attempt == request.max_attempts - 1:
            steps[i] = new_step_text
            step_confs[i] = new_conf
            step_regenerated[i] = RegenerationType.AUTO
            break

    return ReasoningResponse(
        steps=steps,
        final_answer=final_answer,
        step_confidences=step_confs,
        final_answer_confidence=final_answer_conf,
        step_regenerated=step_regenerated,
        final_regenerated=final_regenerated,
    )


def run_final_regen_pipeline(request: FinalRegenRequest) -> ReasoningResponse:
    steps = list(request.steps)
    step_confs = list(request.step_confidences)
    step_regenerated = list(request.step_regenerated)

    final_answer = ""
    final_answer_conf = None
    final_regenerated = RegenerationType.NOT_REGENERATED

    for attempt in range(request.max_attempts):
        prompt = build_final_regen_prompt(request.input, steps)
        output, tokenizer = generate_with_scores(prompt)

        generated_text = _decode_generation(prompt, output, tokenizer, prefix="")

        # For final regen the prompt instructs plain text output (no tags),
        # so treat the entire decoded output as the final answer
        _, extracted_final = extract_cot(generated_text)
        final_answer = extracted_final if extracted_final else generated_text.strip()

        tokens, probs = compute_token_confidence(output, tokenizer)
        _, final_answer_conf = map_step_confidence(steps, final_answer, tokens, probs)

        final_regenerated = RegenerationType.AUTO

        if final_answer_conf.mean_confidence >= request.threshold or attempt == request.max_attempts - 1:
            break

    return ReasoningResponse(
        steps=steps,
        final_answer=final_answer,
        step_confidences=step_confs,
        final_answer_confidence=final_answer_conf,
        step_regenerated=step_regenerated,
        final_regenerated=final_regenerated,
    )

def run_reasoning_pipeline(request: ReasoningRequest):

    prompt = build_reasoning_prompt(request.input)
    output, tokenizer = generate_with_scores(prompt)

    generated_text = _decode_generation(prompt, output, tokenizer, prefix="[STEP 1]\n")

    steps, final_answer = extract_cot(generated_text)

    tokens, probs = compute_token_confidence(output, tokenizer)

    step_confs, final_answer_conf = map_step_confidence(steps, final_answer, tokens, probs)

    step_regenerated = [RegenerationType.NOT_REGENERATED] * len(steps)
    final_regenerated = RegenerationType.NOT_REGENERATED

    # now check if any steps or the final answer are below threshold and regenerate if needed
    if request.threshold > 0:
        for i, step in enumerate(step_confs):
            if step.mean_confidence < request.threshold:
                step_regen_req = StepRegenRequest(
                    input=request.input,
                    steps=steps,
                    tokens=_tokens_to_objects(tokens, probs),
                    step_regenerated=step_regenerated,
                    step_confidences=step_confs,
                    final_answer_confidence=final_answer_conf,
                    step_to_regenerate_index=i,
                    final_answer=final_answer,
                    final_regenerated=final_regenerated,
                    max_attempts=request.max_attempts,
                    threshold=request.threshold,
                )
                result = run_step_regen_pipeline(step_regen_req)
                steps[i] = result.steps[0]
                step_confs[i] = result.step_confidences[0]
                step_regenerated[i] = RegenerationType.AUTO

        # Check if final answer needs regeneration (if any step was regenerated or final is missing)
        if any(s == RegenerationType.AUTO for s in step_regenerated) or not final_answer or final_answer_conf.mean_confidence < request.threshold:
            final_answer_regen_req = FinalRegenRequest(
                input=request.input,
                steps=steps,
                tokens=_tokens_to_objects(tokens, probs),
                step_regenerated=step_regenerated,
                step_confidences=step_confs,
                max_attempts=request.max_attempts,
                threshold=request.threshold,
            )
            result = run_final_regen_pipeline(final_answer_regen_req)
            final_answer = result.final_answer
            final_answer_conf = result.final_answer_confidence
            final_regenerated = result.final_regenerated

    return ReasoningResponse(
        steps=steps,
        final_answer=final_answer,
        step_confidences=step_confs,
        final_answer_confidence=final_answer_conf,
        step_regenerated=step_regenerated,
        final_regenerated=final_regenerated,
    )



# -------------------------
# Shared utilities
# -------------------------

def _decode_generation(prompt: str, output, tokenizer, prefix: str = ""):
    """Extract generated text after the prompt."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    generated_ids = output.sequences[0][prompt_len:]
    return prefix + tokenizer.decode(generated_ids, skip_special_tokens=True)

def _tokens_to_objects(tokens, probs):
    return [TokenConfidence(token=t, prob=p) for t, p in zip(tokens, probs)]


# # -------------------------
# # Step generation
# # -------------------------

# def _generate_single_step(user_input: str, prior_steps: list[str], step_number: int):
#     """Generate one step and return structured output."""
#     prompt = build_step_regen_prompt(user_input, prior_steps, step_number)
#     output, tokenizer = generate_with_scores(prompt)

#     generated_text = _decode_generation(prompt, output, tokenizer)

#     text = f"[STEP {step_number}]\n{generated_text}"
#     steps, _ = extract_cot(text)

#     step_text = steps[0] if steps else generated_text.strip()

#     tokens, probs = compute_token_confidence(output, tokenizer)
#     mean_conf, _ = _safe_stats(probs)

#     return step_text, mean_conf, tokens, probs


# def run_step_regen_pipeline(request: StepRegenRequest):
#     """Regenerate a single step with retry + best selection."""
#     best = {
#         "text": None,
#         "conf": -1.0,
#         "tokens": [],
#         "probs": [],
#     }

#     for _ in range(request.max_attempts):
#         step_text, conf, tokens, probs = _generate_single_step(
#             request.input, request.prior_steps, request.step_number
#         )

#         if conf > best["conf"]:
#             best.update(text=step_text, conf=conf, tokens=tokens, probs=probs)

#         if request.threshold is None or conf >= request.threshold:
#             break

#     mean_conf, min_conf = _safe_stats(best["probs"])

#     return ReasoningResponse(
#         final_answer="",
#         steps=[best["text"]],
#         confidence=ConfidenceSummary(mean=mean_conf, min=min_conf),
#         step_confidence=[best["conf"]],
#         tokens=_tokens_to_objects(best["tokens"], best["probs"]),
#         step_regenerated=["manual"],
#         final_regenerated=None,
#     )


# # -------------------------
# # Final answer regeneration
# # -------------------------

# def run_final_regen_pipeline(user_input: str, steps: list[str], label: str = "manual"):
#     """
#     Regenerate final answer from steps.

#     Args:
#         label: "manual" when called directly, "auto" when triggered by threshold logic.
#     """
#     prior = "\n\n".join(
#         f"[STEP {i}] \n{s}" for i, s in enumerate(steps, 1)
#     )

#     prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a precise reasoning engine. Output ONLY the final answer text, no tags.
# <|eot_id|><|start_header_id|>user<|end_header_id|>
# {user_input}
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# {prior}

# [FINAL]
# The answer is:""".strip()

#     output, tokenizer = generate_with_scores(prompt, min_new_tokens=5)
#     final = _decode_generation(prompt, output, tokenizer).strip()

#     # Cleanup artifacts
#     if final.lower().startswith("[final"):
#         final = final.split("]", 1)[-1].strip()

#     if not final:
#         final = "No answer generated."

#     tokens, probs = compute_token_confidence(output, tokenizer)
#     mean_conf, min_conf = _safe_stats(probs)

#     return final, tokens, probs, mean_conf, min_conf, label


# # -------------------------
# # Main reasoning pipeline
# # -------------------------

# def run_reasoning_pipeline(request: ReasoningRequest):
#     """Full reasoning pipeline with optional step + final regeneration."""

#     # ---- Initial generation ----
#     prompt = build_reasoning_prompt(request.input)
#     output, tokenizer = generate_with_scores(prompt)

#     generated_text = _decode_generation(prompt, output, tokenizer)
#     text = "[STEP 1]\n" + generated_text

#     steps, final_answer = extract_cot(text)

#     tokens, probs = compute_token_confidence(output, tokenizer)
#     step_confs = map_step_confidence(steps, probs, tokens)

#     step_regenerated = [None] * len(steps)
#     final_regenerated: str | None = None

#     # ---- Step regeneration ----
#     if request.threshold is not None:
#         for i, conf in enumerate(step_confs):
#             if conf < request.threshold:
#                 regen_req = StepRegenRequest(
#                     input=request.input,
#                     prior_steps=steps[:i],
#                     step_number=i + 1,
#                     max_attempts=request.max_attempts,
#                     threshold=request.threshold,
#                 )

#                 result = run_step_regen_pipeline(regen_req)

#                 steps[i] = result.steps[0]
#                 step_confs[i] = result.step_confidence[0]
#                 step_regenerated[i] = "auto"

#     # ---- Final answer regeneration ----
#     if request.threshold is not None:
#         # Trigger auto-regen if any step was regenerated or final answer is missing
#         if any(s == "auto" for s in step_regenerated) or not final_answer:
#             final_answer, final_tokens, final_probs, _, _, final_regenerated = (
#                 run_final_regen_pipeline(request.input, steps, label="auto")
#             )
#             tokens += final_tokens
#             probs += final_probs

#     # ---- Final stats ----
#     mean_conf, min_conf = _safe_stats(probs)

#     return ReasoningResponse(
#         final_answer=final_answer,
#         steps=steps,
#         confidence=ConfidenceSummary(mean=mean_conf, min=min_conf),
#         step_confidence=step_confs,
#         tokens=_tokens_to_objects(tokens, probs),
#         step_regenerated=step_regenerated,
#         final_regenerated=final_regenerated,  # "auto" | None (pipeline never sets "manual")
#     )