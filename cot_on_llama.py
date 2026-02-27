import os
import json
import datetime
from pydantic import BaseModel
from unsloth import FastLanguageModel
from transformers import TextStreamer

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# ── Config ────────────────────────────────────────────────────────────────────
max_seq_length = 8192

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class Step(BaseModel):
    '''Required steps to answer the question.'''
    explanation: str

class ChainOfThought(BaseModel):
    '''Final answer with the list of steps.'''
    steps: list[Step]
    final_answer: str

def repair_json(s: str) -> str:
    """Close any unclosed braces or brackets in a truncated JSON string."""
    opening = {'{': '}', '[': ']'}
    closing = set('}]')
    stack = []
    in_string = False
    escape = False

    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in opening:
                stack.append(opening[ch])
            elif ch in closing and stack and stack[-1] == ch:
                stack.pop()

    # Append any missing closing characters in reverse order
    return s + ''.join(reversed(stack))

# ── Load model ────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name                 = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    load_in_4bit               = True,
    use_gradient_checkpointing = "unsloth",
    max_seq_length             = max_seq_length,
    device_map                 = "cuda",
)
FastLanguageModel.for_inference(model)

# ── Prompt setup ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a bot that ONLY responds with an instance of JSON without any additional information. \
You have access to a JSON schema, which will determine how the JSON should be structured."""

TASK = "What is the main landmark of France and what is its capital?"

schema = json.dumps(ChainOfThought.model_json_schema(), indent=2)

PROMPT_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Make sure to return ONLY an instance of the JSON, NOT the schema itself. Do not add any additional information.

JSON schema:
{schema}

Task: {task}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

prompt = PROMPT_TEMPLATE.format(
    system_prompt=SYSTEM_PROMPT,
    schema=schema,
    task=TASK,
)

# ── Generate ──────────────────────────────────────────────────────────────────
gen_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

print("=" * 60)
print("GENERATING CHAIN OF THOUGHT RESPONSE")
print("=" * 60)


output_ids = model.generate(
    **gen_inputs,
    max_new_tokens     = 512,
    eos_token_id       = tokenizer.eos_token_id,
    pad_token_id       = tokenizer.eos_token_id,
    repetition_penalty = 1.2,
    do_sample          = False,
)

input_len    = gen_inputs["input_ids"].shape[1]
response_ids = output_ids[0, input_len:]
raw_response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

print(f"\nRaw response:\n{raw_response}")

# ── Parse & validate ──────────────────────────────────────────────────────────
try:
    repaired = repair_json(raw_response)
    answer = ChainOfThought.model_validate_json(repaired)
    print("\n" + "=" * 60)
    print("PARSED CHAIN OF THOUGHT")
    print("=" * 60)
    for i, step in enumerate(answer.steps, 1):
        print(f"  Step {i}: {step.explanation}")
    print(f"\n  Final answer: {answer.final_answer}")
except Exception as e:
    print(f"\nFailed to parse response as ChainOfThought: {e}")
    print("Raw response was:", raw_response)