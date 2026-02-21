import datetime
from unsloth import FastLanguageModel
from transformers import TextStreamer

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
max_seq_length = 8192
N_SHAP_SAMPLES = 128

# ── Load model ────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name                 = "unsloth/Llama-3.1-8B-bnb-4bit",
    load_in_4bit               = True,
    use_gradient_checkpointing = "unsloth",
    max_seq_length             = max_seq_length,
    device_map                 = "cuda",
)
FastLanguageModel.for_inference(model)

MASK_TOKEN = "[MASKED]"

PROMPT_TEMPLATE = """\
{system_prompt}

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""

def build_prompt(system_prompt, instruction, inp, response=""):
    return PROMPT_TEMPLATE.format(system_prompt=system_prompt, instruction=instruction, input=inp, response=response)

# ── Step 1: generate a clean response ────────────────────────────────────────
SYSTEM_PROMPT = """Below is an instruction that describes a task, paired with an input that \
provides further context. Write a response that appropriately completes the request."""
INSTRUCTION = "Answer like a human would in an engaging way"
INPUT_TEXT  = "What is the main landmark of France and what is its capital?"

prompt_for_gen = build_prompt(SYSTEM_PROMPT,INSTRUCTION, INPUT_TEXT)
gen_inputs     = tokenizer([prompt_for_gen], return_tensors="pt").to("cuda")

print("=" * 60)
print("STANDARD GENERATION OUTPUT")
print("=" * 60)

# FIX: add eos_token_id + repetition_penalty to prevent looping
output_ids = model.generate(
    **gen_inputs,
    streamer           = TextStreamer(tokenizer),
    max_new_tokens     = 128,
    eos_token_id       = tokenizer.eos_token_id,
    pad_token_id       = tokenizer.eos_token_id,
    repetition_penalty = 1.3,          # penalises repeating itself
    do_sample          = False,
)

# Decode only newly generated tokens
input_len    = gen_inputs["input_ids"].shape[1]
response_ids = output_ids[0, input_len:]
raw_response = tokenizer.decode(response_ids, skip_special_tokens=True)

# FIX: strip anything from the first repeated section marker onwards
# (safety net in case repetition_penalty wasn't enough)
for stop_marker in ["\n\n### Instruction:", "\n### Instruction:", "### Input:"]:
    if stop_marker in raw_response:
        raw_response = raw_response.split(stop_marker)[0]

RESPONSE_TEXT = raw_response.strip()
print(f"\n\nCaptured response for SHAP: {repr(RESPONSE_TEXT)}")

# ── Step 2: define word-level SHAP features ───────────────────────────────────
def tokenize_section(text, label):
    words = text.split()
    return words, [f"[{label}] {w}" for w in words]

sys_words,  sys_labels  = tokenize_section(SYSTEM_PROMPT, "SYSTEM")
inst_words, inst_labels = tokenize_section(INSTRUCTION, "INSTRUCTION")
inp_words,  inp_labels  = tokenize_section(INPUT_TEXT,  "INPUT")

all_words  = sys_words  + inst_words  + inp_words
all_labels = sys_labels + inst_labels + inp_labels
n_features = len(all_words)
n_sys      = len(sys_words)
n_inst     = len(inst_words)

print(f"\nSHAP features: {n_features} words "
      f"({n_sys} system, {n_inst} instruction, {len(inp_words)} input)")


@torch.no_grad()
def compute_log_prob(full_prompt_with_response: str) -> float:
    """Mean per-token log-prob of the response portion."""
    enc_full = tokenizer(
        full_prompt_with_response, return_tensors="pt",
        truncation=True, max_length=max_seq_length
    ).to("cuda")

    # Find where response starts by encoding prompt-only
    prompt_only = full_prompt_with_response.split("### Response:\n")[0] + "### Response:\n"
    enc_prompt  = tokenizer(
        prompt_only, return_tensors="pt",
        truncation=True, max_length=max_seq_length
    ).to("cuda")
    prompt_len = enc_prompt["input_ids"].shape[1]

    input_ids = enc_full["input_ids"]
    labels    = input_ids.clone()
    labels[:, :prompt_len] = -100   # only score the response tokens

    loss = model(input_ids=input_ids, labels=labels).loss
    return -loss.item()             # higher = more confident in response


def reconstruct_prompt(mask: np.ndarray) -> str:
    words     = [w if mask[i] == 1 else MASK_TOKEN for i, w in enumerate(all_words)]
    sys_text  = " ".join(words[:n_sys])
    inst_text = " ".join(words[n_sys : n_sys + n_inst])
    inp_text  = " ".join(words[n_sys + n_inst :])
    return (
        f"{sys_text}\n\n"
        f"### Instruction:\n{inst_text}\n\n"
        f"### Input:\n{inp_text}\n\n"
        f"### Response:\n{RESPONSE_TEXT}"
    )


def shap_predict(mask_matrix: np.ndarray) -> np.ndarray:
    return np.array([
        compute_log_prob(reconstruct_prompt(mask))
        for mask in mask_matrix
    ])


# ── Step 3: SHAP KernelExplainer ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("RUNNING SHAP …")
print("=" * 60)

# FIX: background = all-zeros (all words masked = null reference state)
#      test input = all-ones  (full prompt)
# Without this, baseline == test and all SHAP values are trivially 0.
background  = np.zeros((1, n_features))   # ← was np.ones — that was the bug
test_input  = np.ones((1, n_features))

explainer   = shap.KernelExplainer(shap_predict, background)
shap_values = explainer.shap_values(test_input, nsamples=N_SHAP_SAMPLES, silent=False)
sv = shap_values[0]


# ── Step 4: visualise ─────────────────────────────────────────────────────────
def plot_shap_bar(shap_vals, labels, title, path):
    if len(shap_vals) == 0:
        return
    order = np.argsort(np.abs(shap_vals))[::-1]
    top_n = min(30, len(order))
    idx   = order[:top_n][::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.38)))
    
    # FIX: use threshold instead of bare > 0 so near-zero values get grey
    def bar_color(v):
        if v > 1e-4:   return "#d73027"   # red   = promotes
        if v < -1e-4:  return "#4575b4"   # blue  = suppresses
        return "#aaaaaa"                   # grey  = negligible

    colors = [bar_color(shap_vals[i]) for i in idx]
    ax.barh(range(top_n), shap_vals[idx], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([labels[i] for i in idx], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value  (red ↑ promotes · blue ↓ suppresses · grey ◦ negligible)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

dateTime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
path = f"shap_graphs/{dateTime}/"
os.makedirs(path, exist_ok=True)

plot_shap_bar(sv,                        all_labels, f'SHAP – all words → "{RESPONSE_TEXT[:50]}…"', f"{path}shap_overall.png")
plot_shap_bar(sv[:n_sys],                sys_labels,  "SHAP – System prompt words",                f"{path}shap_system.png")
plot_shap_bar(sv[n_sys:n_sys+n_inst],    inst_labels, "SHAP – Instruction words",                  f"{path}shap_instruction.png")
plot_shap_bar(sv[n_sys+n_inst:],         inp_labels,  "SHAP – Input words",                        f"{path}shap_input.png")

print("\nTop 15 most influential words:")
order = np.argsort(np.abs(sv))[::-1]
order = np.argsort(np.abs(sv))[::-1]
for rank, i in enumerate(order[:15], 1):
    v = sv[i]
    if v > 1e-4:
        direction = "↑ promotes"
    elif v < -1e-4:
        direction = "↓ suppresses"
    else:
        direction = "◦ negligible"
    print(f"  {rank:>2}. {all_labels[i]:<40s}  {v:+.6f}  {direction}")