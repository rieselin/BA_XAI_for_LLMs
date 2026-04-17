import re

def extract_cot(text: str):
    steps = re.findall(
        r'\[STEP\s*\d+\]\s*(.*?)(?=\[STEP\s*\d+\]|\[FINAL\]|\Z)',
        text,
        re.DOTALL | re.IGNORECASE
    )
    steps = [s.strip() for s in steps if s.strip()]

    final_match = re.search(
        r'\[FINAL(?:\s*ANSWER)?\]\s*(.*?)(?:\s*<\|eot_id\|>)?\s*$',
        text,
        re.DOTALL | re.IGNORECASE
    )

    if final_match:
        final_answer = final_match.group(1).strip()
    elif steps:
        # Fallback: treat the last step as the answer
        final_answer = steps.pop()
    else:
        final_answer = ""

    return steps, final_answer