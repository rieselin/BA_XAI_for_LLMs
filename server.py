"""
FastAPI backend for LLM Explainability Explorer
Wraps: (1) Chain-of-Thought JSON generation  (2) SHAP word-level attribution
"""
import os, json, base64, warnings
from io import BytesIO

os.environ["TORCHDYNAMO_DISABLE"] = "1"
warnings.filterwarnings("ignore")

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBase
from typing import Optional

# ── Fixed system prompts (not user-configurable) ──────────────────────────────
COT_SYSTEM_PROMPT = (
    "You are a bot that ONLY responds with an instance of JSON without any "
    "additional information. You have access to a JSON schema, which will "
    "determine how the JSON should be structured."
)

SHAP_SYSTEM_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request."
)

# ── Lazy model loader ─────────────────────────────────────────────────────────
_model = None
_tokenizer = None

def get_model(instruct: bool = True):
    global _model, _tokenizer
    model_name = (
        "unsloth/Llama-3.1-8B-Instruct-bnb-4bit" if instruct
        else "unsloth/Llama-3.1-8B-bnb-4bit"
    )
    if _model is None:
        from unsloth import FastLanguageModel
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
            max_seq_length=8192,
            device_map="cuda",
        )
        FastLanguageModel.for_inference(_model)
    return _model, _tokenizer


# ── Request / Response schemas ────────────────────────────────────────────────
class CoTRequest(PydanticBase):
    task: str
    instruction: str = "Think step by step and reason carefully."
    max_new_tokens: int = 512

class CoTStep(PydanticBase):
    explanation: str

class CoTResponse(PydanticBase):
    steps: list[CoTStep]
    final_answer: str
    raw: str

class ShapRequest(PydanticBase):
    instruction: str = "Answer like a human would in an engaging way"
    input_text: str
    n_shap_samples: int = 128
    max_new_tokens: int = 128

class WordAnnotation(PydanticBase):
    word: str
    shap_value: float
    section: str   # "system" | "instruction" | "input"

class ShapResponse(PydanticBase):
    response_text: str
    overall_image: str
    system_image: str
    instruction_image: str
    input_image: str
    top_words: list[dict]
    word_annotations: list[WordAnnotation]


# ── Helpers ───────────────────────────────────────────────────────────────────
def repair_json(s: str) -> str:
    opening = {"{": "}", "[": "]"}
    closing = set("}]")
    stack, in_string, escape = [], False, False
    for ch in s:
        if escape:
            escape = False; continue
        if ch == "\\" and in_string:
            escape = True; continue
        if ch == '"':
            in_string = not in_string; continue
        if not in_string:
            if ch in opening:
                stack.append(opening[ch])
            elif ch in closing and stack and stack[-1] == ch:
                stack.pop()
    return s + "".join(reversed(stack))


def fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def plot_shap_bar(shap_vals, labels, title) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(shap_vals) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig_to_b64(fig)

    order = np.argsort(np.abs(shap_vals))[::-1]
    top_n = min(30, len(order))
    idx   = order[:top_n][::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.38)), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    def bar_color(v):
        if v >  1e-4: return "#f97316"
        if v < -1e-4: return "#38bdf8"
        return "#6b7280"

    colors = [bar_color(shap_vals[i]) for i in idx]
    ax.barh(range(top_n), shap_vals[idx], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([labels[i] for i in idx], fontsize=8, color="#e2e8f0")
    ax.axvline(0, color="#4b5563", linewidth=0.8)
    ax.set_xlabel("SHAP value  (orange promotes · blue suppresses)", color="#9ca3af", fontsize=9)
    ax.set_title(title, color="#f1f5f9", fontsize=11, pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")
    ax.tick_params(colors="#9ca3af")
    plt.tight_layout()
    return fig_to_b64(fig)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="LLM Explainability API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/explain/cot", response_model=CoTResponse)
def explain_cot(req: CoTRequest):
    """Chain-of-Thought structured JSON generation."""
    from pydantic import BaseModel as _Base

    class Step(_Base):
        explanation: str

    class ChainOfThought(_Base):
        steps: list[Step]
        final_answer: str

    model, tokenizer = get_model(instruct=True)
    schema = json.dumps(ChainOfThought.model_json_schema(), indent=2)

    # Prime the assistant turn with the opening brace so the model continues
    # the JSON directly rather than re-introducing it as a string.
    PROMPT = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{COT_SYSTEM_PROMPT}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Instruction: {req.instruction}\n\n"
        "Return ONLY a valid JSON object matching the schema below. "
        "Do NOT include the schema itself, any explanation, or any text outside the JSON object.\n\n"
        f"JSON schema:\n{schema}\n\nTask: {req.task}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "{"   # prime: model continues from here, preventing double-wrapping
    )

    # Tokenize with the primed opening brace; we'll prepend it back to raw output
    inputs = tokenizer([PROMPT], return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        do_sample=False,
    )
    # The primed "{" is part of the prompt so we prepend it back
    raw = "{" + tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()

    def _unwrap_if_double_encoded(candidate: ChainOfThought) -> ChainOfThought:
        """If explanation[0] is itself valid JSON for our schema, use that instead."""
        if candidate.steps and candidate.steps[0].explanation.strip().startswith("{"):
            try:
                inner = ChainOfThought.model_validate_json(
                    repair_json(candidate.steps[0].explanation)
                )
                if inner.steps:
                    return inner
            except Exception:
                pass
        return candidate

    def _make_response(cot: ChainOfThought) -> CoTResponse:
        cot = _unwrap_if_double_encoded(cot)
        return CoTResponse(
            steps=[CoTStep(explanation=s.explanation) for s in cot.steps],
            final_answer=cot.final_answer,
            raw=raw,
        )

    # Stage 1: parse as-is (with brace repair)
    try:
        return _make_response(ChainOfThought.model_validate_json(repair_json(raw)))
    except Exception:
        pass

    # Stage 2: extract outermost {...} and retry
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        return _make_response(ChainOfThought.model_validate_json(repair_json(raw[start:end])))
    except Exception:
        pass

    # Stage 3: synthetic fallback — split raw text into steps so the frontend always works
    sentences = [s.strip() for s in raw.replace("\n", " ").split(".") if s.strip()]
    steps = [CoTStep(explanation=s + ".") for s in sentences[:-1]] if len(sentences) > 1 else [CoTStep(explanation=raw)]
    final = sentences[-1] if sentences else raw
    return CoTResponse(steps=steps, final_answer=final, raw=raw)


@app.post("/explain/shap", response_model=ShapResponse)
def explain_shap(req: ShapRequest):
    """SHAP KernelExplainer word-level attribution."""
    import shap as _shap

    model, tokenizer = get_model(instruct=False)
    MAX_SEQ = 8192

    PROMPT_TEMPLATE = (
        "{system_prompt}\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{response}"
    )

    # 1. Generate response
    # Strategy: generate up to 2× the requested budget so the model has room
    # to finish naturally, then trim to the last complete sentence that fits
    # within the token budget. This avoids mid-sentence cutoffs without relying
    # on the base model following word-count instructions (it won't reliably).
    hard_cap = req.max_new_tokens * 2

    prompt_for_gen = PROMPT_TEMPLATE.format(
        system_prompt=SHAP_SYSTEM_PROMPT,
        instruction=req.instruction,
        input=req.input_text,
        response="",
    )
    gen_inputs = tokenizer([prompt_for_gen], return_tensors="pt").to("cuda")
    out = model.generate(
        **gen_inputs,
        max_new_tokens=hard_cap,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.3,
        do_sample=False,
    )
    raw_resp = tokenizer.decode(out[0, gen_inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
    for marker in ["\n\n### Instruction:", "\n### Instruction:", "### Input:"]:
        if marker in raw_resp:
            raw_resp = raw_resp.split(marker)[0]
    raw_resp = raw_resp.strip()

    # Trim to the last complete sentence within req.max_new_tokens.
    # Re-tokenise the full response word-by-word and walk back from the last
    # sentence boundary that keeps us under budget.
    import re as _re
    def _trim_to_token_budget(text: str, budget: int) -> str:
        # Split on sentence-ending punctuation, keeping the delimiter
        sentences = _re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            return text
        kept = []
        for sent in sentences:
            candidate = " ".join(kept + [sent])
            n_tokens = tokenizer(candidate, return_tensors="pt")["input_ids"].shape[1]
            if n_tokens <= budget:
                kept.append(sent)
            else:
                break   # adding this sentence would exceed budget
        if kept:
            return " ".join(kept)
        # Edge case: even the first sentence is too long — truncate at last word
        # boundary within budget tokens instead of returning nothing
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0][:budget]
        return tokenizer.decode(tokens, skip_special_tokens=True).rsplit(" ", 1)[0]

    RESPONSE_TEXT = _trim_to_token_budget(raw_resp, req.max_new_tokens)

    # 2. Build word features (system prompt also attributed)
    def tokenize_section(text, label):
        words = text.split()
        return words, [f"[{label}] {w}" for w in words]

    sys_words,  sys_labels  = tokenize_section(SHAP_SYSTEM_PROMPT, "SYSTEM")
    inst_words, inst_labels = tokenize_section(req.instruction,    "INSTRUCTION")
    inp_words,  inp_labels  = tokenize_section(req.input_text,     "INPUT")
    all_words  = sys_words  + inst_words  + inp_words
    all_labels = sys_labels + inst_labels + inp_labels
    n_features = len(all_words)
    n_sys      = len(sys_words)
    n_inst     = len(inst_words)
    MASK_TOKEN = "[MASKED]"

    @torch.no_grad()
    def compute_log_prob(full_prompt: str) -> float:
        enc_full    = tokenizer(full_prompt, return_tensors="pt",
                                truncation=True, max_length=MAX_SEQ).to("cuda")
        prompt_only = full_prompt.split("### Response:\n")[0] + "### Response:\n"
        enc_prompt  = tokenizer(prompt_only, return_tensors="pt",
                                truncation=True, max_length=MAX_SEQ).to("cuda")
        prompt_len  = enc_prompt["input_ids"].shape[1]
        input_ids   = enc_full["input_ids"]
        labels      = input_ids.clone()
        labels[:, :prompt_len] = -100
        return -model(input_ids=input_ids, labels=labels).loss.item()

    def reconstruct_prompt(mask: np.ndarray) -> str:
        words  = [w if mask[i] == 1 else MASK_TOKEN for i, w in enumerate(all_words)]
        sys_t  = " ".join(words[:n_sys])
        inst_t = " ".join(words[n_sys : n_sys + n_inst])
        inp_t  = " ".join(words[n_sys + n_inst :])
        return (f"{sys_t}\n\n### Instruction:\n{inst_t}\n\n"
                f"### Input:\n{inp_t}\n\n### Response:\n{RESPONSE_TEXT}")

    def shap_predict(mask_matrix: np.ndarray) -> np.ndarray:
        return np.array([compute_log_prob(reconstruct_prompt(m)) for m in mask_matrix])

    # 3. SHAP
    background = np.zeros((1, n_features))
    test_input = np.ones((1, n_features))
    explainer  = _shap.KernelExplainer(shap_predict, background)
    sv         = explainer.shap_values(test_input, nsamples=req.n_shap_samples, silent=True)[0]

    # 4. Top words
    order     = np.argsort(np.abs(sv))[::-1]
    top_words = []
    for i in order[:15]:
        v = sv[i]
        direction = "promotes" if v > 1e-4 else ("suppresses" if v < -1e-4 else "negligible")
        top_words.append({"label": all_labels[i], "value": float(v), "direction": direction})

    # 5. Plots
    overall_img     = plot_shap_bar(sv,                     all_labels,  f'All words → "{RESPONSE_TEXT[:50]}…"')
    system_img      = plot_shap_bar(sv[:n_sys],             sys_labels,  "System prompt words")
    instruction_img = plot_shap_bar(sv[n_sys:n_sys+n_inst], inst_labels, "Instruction words")
    input_img       = plot_shap_bar(sv[n_sys+n_inst:],      inp_labels,  "Input words")

    # 6. Word annotations for inline rendering
    annotations = []
    for i, (w, _) in enumerate(zip(all_words, all_labels)):
        section = "system" if i < n_sys else ("instruction" if i < n_sys + n_inst else "input")
        annotations.append(WordAnnotation(word=w, shap_value=float(sv[i]), section=section))

    return ShapResponse(
        response_text=RESPONSE_TEXT,
        overall_image=overall_img,
        system_image=system_img,
        instruction_image=instruction_img,
        input_image=input_img,
        top_words=top_words,
        word_annotations=annotations,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)