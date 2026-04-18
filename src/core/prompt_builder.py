def build_reasoning_prompt(user_input: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise reasoning engine. You MUST respond using ONLY this exact format with square bracket tags. Do not use markdown headers.
You MUST end your response with a [FINAL] tag containing your conclusion. Never stop before reaching [FINAL]

[STEP 1]
your reasoning here

[STEP 2]
your reasoning here

[FINAL]
your final answer here
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_input}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
[STEP 1]
""".strip()

def build_final_regen_prompt(user_input: str, steps: list[str]) -> str:
    steps_str = "\n\n".join(f"[STEP {i}]\n{s}" for i, s in enumerate(steps, 1))
    return  f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise reasoning engine. Output ONLY the final answer text, no tags.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_input}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{steps_str}


[FINAL]
""".strip()




def build_step_regen_prompt(user_input: str, prior_steps: list[str], step_number: int) -> str:
    prior = ""
    for i, s in enumerate(prior_steps, 1):
        prior += f"[STEP {i}]\n{s}\n\n"

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise reasoning engine. You MUST respond using ONLY this exact format with square bracket tags. Do not use markdown headers.

[STEP 1]
your reasoning here

[STEP 2]
your reasoning here

[FINAL]
your final answer here
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_input}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{prior}[STEP {step_number}]
""".strip()