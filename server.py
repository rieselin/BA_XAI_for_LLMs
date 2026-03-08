"""
FastAPI-Backend für den LLM-Erklärbarkeits-Explorer
Endpunkte:
  GET  /health
  POST /explain/cot         – Chain-of-Thought JSON-Generierung
  POST /explain/shap        – SHAP-Wortattribution + KI-Gegenanalyse
  POST /explain/confidence  – Token-Konfidenz-Bewertung
"""
import os
import json
import base64
import threading
import warnings
from io import BytesIO

os.environ["TORCHDYNAMO_DISABLE"] = "1"
warnings.filterwarnings("ignore")

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel as PydanticBase, field_validator

# ── Umgebungskonfiguration ────────────────────────────────────────────────────
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS: list[str] = (
    ["*"] if _raw_origins.strip() == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

# ── Feste System-Prompts ──────────────────────────────────────────────────────
COT_SYSTEM_PROMPT = (
    "You are a bot that ONLY responds with an instance of JSON without any "
    "additional information. You have access to a JSON schema, which will "
    "determine how the JSON should be structured."
)

DEFAULT_SHAP_SYSTEM_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request."
)

DEFAULT_CONFIDENCE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions clearly and concisely."
)

# ── Eingabelimits ─────────────────────────────────────────────────────────────
MAX_TEXT_CHARS   = 2_000
MAX_NEW_TOKENS   = 1_024
MAX_SHAP_SAMPLES = 512

# ── Thread-sicherer Lazy-Model-Loader ─────────────────────────────────────────
_model      = None
_tokenizer  = None
_model_lock = threading.Lock()


def get_model(instruct: bool = True):
    """Modell genau einmal laden, thread-sicher."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    with _model_lock:
        if _model is not None:       # Double-Checked Locking
            return _model, _tokenizer
        model_name = (
            "unsloth/Llama-3.1-8B-Instruct-bnb-4bit" if instruct
            else "unsloth/Llama-3.1-8B-bnb-4bit"
        )
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


# ── Gemeinsamer Validator ─────────────────────────────────────────────────────
def _text_validator(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("Das Feld darf nicht leer sein.")
    if len(v) > MAX_TEXT_CHARS:
        raise ValueError(f"Das Feld überschreitet das Limit von {MAX_TEXT_CHARS} Zeichen.")
    return v


# ── Schemas ───────────────────────────────────────────────────────────────────

class CoTRequest(PydanticBase):
    task:           str
    instruction:    str = "Denke Schritt für Schritt und argumentiere sorgfältig."
    max_new_tokens: int = 512

    @field_validator("task", "instruction")
    @classmethod
    def _chk(cls, v): return _text_validator(v)

    @field_validator("max_new_tokens")
    @classmethod
    def _cap(cls, v): return max(64, min(v, MAX_NEW_TOKENS))

class CoTStep(PydanticBase):
    explanation: str

class CoTResponse(PydanticBase):
    steps:        list[CoTStep]
    final_answer: str
    raw:          str


class ShapRequest(PydanticBase):
    system_prompt:  str = DEFAULT_SHAP_SYSTEM_PROMPT
    instruction:    str = "Antworte so, wie ein Mensch es auf ansprechende Weise tun würde"
    input_text:     str
    n_shap_samples: int = 128
    max_new_tokens: int = 128

    @field_validator("system_prompt", "instruction", "input_text")
    @classmethod
    def _chk(cls, v): return _text_validator(v)

    @field_validator("n_shap_samples")
    @classmethod
    def _cap_s(cls, v): return max(16, min(v, MAX_SHAP_SAMPLES))

    @field_validator("max_new_tokens")
    @classmethod
    def _cap_t(cls, v): return max(32, min(v, MAX_NEW_TOKENS))

class WordAnnotation(PydanticBase):
    word:       str
    shap_value: float
    section:    str   # "system" | "instruction" | "input"

class ShapResponse(PydanticBase):
    response_text:     str
    overall_image:     str
    system_image:      str
    instruction_image: str
    input_image:       str
    top_words:         list[dict]
    word_annotations:  list[WordAnnotation]
    reasoning:         str   # KI-Gegenanalyse der wichtigsten Attributionen


class ConfidenceRequest(PydanticBase):
    system_prompt:  str = DEFAULT_CONFIDENCE_SYSTEM_PROMPT  # jetzt vom Nutzer editierbar
    instruction:    str = "Antworte klar und prägnant."
    input_text:     str
    max_new_tokens: int = 256

    @field_validator("system_prompt", "instruction", "input_text")
    @classmethod
    def _chk(cls, v): return _text_validator(v)

    @field_validator("max_new_tokens")
    @classmethod
    def _cap(cls, v): return max(32, min(v, MAX_NEW_TOKENS))

class TokenConfidence(PydanticBase):
    token:       str
    probability: float   # 0–1

class ConfidenceResponse(PydanticBase):
    response_text:     str
    mean_confidence:   float
    min_confidence:    float
    token_confidences: list[TokenConfidence]


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
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


def plot_shap_bar(shap_vals: np.ndarray, labels: list[str], title: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(shap_vals) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Keine Daten", ha="center")
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
    ax.set_xlabel("SHAP-Wert  (orange = fördert · blau = hemmt)", color="#9ca3af", fontsize=9)
    ax.set_title(title, color="#f1f5f9", fontsize=11, pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")
    ax.tick_params(colors="#9ca3af")
    plt.tight_layout()
    return fig_to_b64(fig)


def _instruct_generate(model, tokenizer, system_prompt: str, user_message: str, max_new_tokens: int = 400) -> str:
    """Einzelner Instruct-Generierungsaufruf – genutzt für Konfidenz & SHAP-Gegenanalyse."""
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_message}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    enc = tokenizer([prompt], return_tensors="pt").to("cuda")
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        do_sample=False,
    )
    return tokenizer.decode(
        out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# ── FastAPI-App ───────────────────────────────────────────────────────────────
app = FastAPI(title="LLM-Erklärbarkeits-API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Chain-of-Thought ──────────────────────────────────────────────────────────
@app.post("/explain/cot", response_model=CoTResponse)
def explain_cot(req: CoTRequest):
    from pydantic import BaseModel as _Base

    class Step(_Base):
        explanation: str

    class ChainOfThought(_Base):
        steps:        list[Step]
        final_answer: str

    model, tokenizer = get_model(instruct=True)
    schema = json.dumps(ChainOfThought.model_json_schema(), indent=2)

    PROMPT = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{COT_SYSTEM_PROMPT}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Instruction: {req.instruction}\n\n"
        "Return ONLY a valid JSON object matching the schema below. "
        "Do NOT include the schema itself, any explanation, or any text outside the JSON object.\n\n"
        f"JSON schema:\n{schema}\n\nTask: {req.task}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "{"
    )

    inputs = tokenizer([PROMPT], return_tensors="pt").to("cuda")
    out    = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        do_sample=False,
    )
    raw = "{" + tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    def _unwrap(candidate: ChainOfThought) -> ChainOfThought:
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

    def _make(cot: ChainOfThought) -> CoTResponse:
        cot = _unwrap(cot)
        return CoTResponse(
            steps=[CoTStep(explanation=s.explanation) for s in cot.steps],
            final_answer=cot.final_answer,
            raw=raw,
        )

    # Versuch 1: direkte JSON-Validierung
    try:
        return _make(ChainOfThought.model_validate_json(repair_json(raw)))
    except Exception:
        pass

    # Versuch 2: erstes vollständiges JSON-Objekt extrahieren
    try:
        start = raw.index("{"); end = raw.rindex("}") + 1
        return _make(ChainOfThought.model_validate_json(repair_json(raw[start:end])))
    except Exception:
        pass

    # Fallback: Sätze als Schritte verwenden
    sentences = [s.strip() for s in raw.replace("\n", " ").split(".") if s.strip()]
    steps     = (
        [CoTStep(explanation=s + ".") for s in sentences[:-1]]
        if len(sentences) > 1 else [CoTStep(explanation=raw)]
    )
    return CoTResponse(steps=steps, final_answer=sentences[-1] if sentences else raw, raw=raw)


# ── SHAP + Gegenanalyse ───────────────────────────────────────────────────────
@app.post("/explain/shap", response_model=ShapResponse)
def explain_shap(req: ShapRequest):
    import re as _re
    import shap as _shap

    base_model,     base_tok     = get_model(instruct=False)
    instruct_model, instruct_tok = get_model(instruct=True)
    MAX_SEQ = 8192

    PROMPT_TEMPLATE = (
        "{system_prompt}\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{response}"
    )

    # 1. Basisantwort generieren
    prompt_for_gen = PROMPT_TEMPLATE.format(
        system_prompt=req.system_prompt,
        instruction=req.instruction,
        input=req.input_text,
        response="",
    )
    gen_inputs = base_tok([prompt_for_gen], return_tensors="pt").to("cuda")
    out = base_model.generate(
        **gen_inputs,
        max_new_tokens=req.max_new_tokens * 2,
        eos_token_id=base_tok.eos_token_id,
        pad_token_id=base_tok.eos_token_id,
        repetition_penalty=1.3,
        do_sample=False,
    )
    raw_resp = base_tok.decode(
        out[0, gen_inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    for marker in ["\n\n### Instruction:", "\n### Instruction:", "### Input:"]:
        if marker in raw_resp:
            raw_resp = raw_resp.split(marker)[0]
    raw_resp = raw_resp.strip()

    def _trim_to_budget(text: str, budget: int) -> str:
        sentences = _re.split(r"(?<=[.!?])\s+", text)
        kept = []
        for sent in sentences:
            candidate = " ".join(kept + [sent])
            if base_tok(candidate, return_tensors="pt")["input_ids"].shape[1] <= budget:
                kept.append(sent)
            else:
                break
        if kept:
            return " ".join(kept)
        # Fallback: auf Token-Ebene kürzen
        tokens = base_tok(text, return_tensors="pt")["input_ids"][0][:budget]
        return base_tok.decode(tokens, skip_special_tokens=True).rsplit(" ", 1)[0]

    RESPONSE_TEXT = _trim_to_budget(raw_resp, req.max_new_tokens)

    # 2. Wort-Features aufbauen
    def tokenize_section(text, label):
        words = text.split()
        return words, [f"[{label}] {w}" for w in words]

    sys_words,  sys_labels  = tokenize_section(req.system_prompt, "SYSTEM")
    inst_words, inst_labels = tokenize_section(req.instruction,   "INSTRUCTION")
    inp_words,  inp_labels  = tokenize_section(req.input_text,    "INPUT")
    all_words  = sys_words  + inst_words  + inp_words
    all_labels = sys_labels + inst_labels + inp_labels
    n_features = len(all_words)
    n_sys  = len(sys_words)
    n_inst = len(inst_words)
    MASK_TOKEN = "[MASKED]"

    @torch.no_grad()
    def compute_log_prob(full_prompt: str) -> float:
        enc_full = base_tok(full_prompt, return_tensors="pt",
                            truncation=True, max_length=MAX_SEQ).to("cuda")
        ponly    = full_prompt.split("### Response:\n")[0] + "### Response:\n"
        enc_p    = base_tok(ponly, return_tensors="pt",
                            truncation=True, max_length=MAX_SEQ).to("cuda")
        ids      = enc_full["input_ids"]
        labels   = ids.clone()
        labels[:, :enc_p["input_ids"].shape[1]] = -100
        return -base_model(input_ids=ids, labels=labels).loss.item()

    def reconstruct_prompt(mask: np.ndarray) -> str:
        words  = [w if mask[i] == 1 else MASK_TOKEN for i, w in enumerate(all_words)]
        return (
            f"{' '.join(words[:n_sys])}\n\n"
            f"### Instruction:\n{' '.join(words[n_sys:n_sys+n_inst])}\n\n"
            f"### Input:\n{' '.join(words[n_sys+n_inst:])}\n\n"
            f"### Response:\n{RESPONSE_TEXT}"
        )

    def shap_predict(mask_matrix: np.ndarray) -> np.ndarray:
        return np.array([compute_log_prob(reconstruct_prompt(m)) for m in mask_matrix])

    # 3. SHAP berechnen
    explainer = _shap.KernelExplainer(shap_predict, np.zeros((1, n_features)))
    sv        = explainer.shap_values(
        np.ones((1, n_features)), nsamples=req.n_shap_samples, silent=True
    )[0]

    # 4. Wichtigste Wörter ermitteln
    order     = np.argsort(np.abs(sv))[::-1]
    top_words = []
    for i in order[:15]:
        v = sv[i]
        top_words.append({
            "label":     all_labels[i],
            "value":     float(v),
            "direction": "promotes" if v > 1e-4 else ("suppresses" if v < -1e-4 else "negligible"),
        })

    # 5. Diagramme erstellen
    overall_img     = plot_shap_bar(sv,                     all_labels,  f'Alle Wörter → „{RESPONSE_TEXT[:50]}…"')
    system_img      = plot_shap_bar(sv[:n_sys],             sys_labels,  "System-Prompt-Wörter")
    instruction_img = plot_shap_bar(sv[n_sys:n_sys+n_inst], inst_labels, "Anweisungswörter")
    input_img       = plot_shap_bar(sv[n_sys+n_inst:],      inp_labels,  "Eingabewörter")

    # 6. Annotationen aufbauen
    annotations = []
    for i, (w, _) in enumerate(zip(all_words, all_labels)):
        section = (
            "system"      if i < n_sys
            else "instruction" if i < n_sys + n_inst
            else "input"
        )
        annotations.append(WordAnnotation(word=w, shap_value=float(sv[i]), section=section))

    # 7. Gegenanalyse über das Instruct-Modell generieren
    promotes   = [tw for tw in top_words[:10] if tw["direction"] == "promotes"][:5]
    suppresses = [tw for tw in top_words[:10] if tw["direction"] == "suppresses"][:5]

    promote_str  = ", ".join(
        f'"{tw["label"].split("] ")[-1]}" (+{abs(tw["value"]):.4f})' for tw in promotes
    ) or "keine identifiziert"
    suppress_str = ", ".join(
        f'"{tw["label"].split("] ")[-1]}" (-{abs(tw["value"]):.4f})' for tw in suppresses
    ) or "keine identifiziert"

    reasoning_prompt = (
        f"Eine SHAP-Analyse wurde für den folgenden KI-Austausch durchgeführt:\n"
        f"  Eingabe: \"{req.input_text}\"\n"
        f"  Antwort: \"{RESPONSE_TEXT}\"\n\n"
        f"Wörter, die die Antwort GEFÖRDERT haben (das Modell zuversichtlicher gemacht haben, sie zu produzieren):\n"
        f"  {promote_str}\n\n"
        f"Wörter, die die Antwort GEHEMMT haben (das Modell weniger zuversichtlich gemacht haben):\n"
        f"  {suppress_str}\n\n"
        f"Erkläre in 3–5 Sätzen in verständlicher Sprache, WARUM diese spezifischen Wörter wahrscheinlich "
        f"diese Effekte hatten. Berücksichtige, was jedes Wort semantisch signalisiert, welche Muster das Modell "
        f"möglicherweise gelernt hat, und warum diese Signale die Konfidenz in diese oder jene Richtung lenken. "
        f"Sei aufschlussreich und lehrreich. Wiederhole nicht die numerischen Werte."
    )

    reasoning = _instruct_generate(
        instruct_model, instruct_tok,
        DEFAULT_CONFIDENCE_SYSTEM_PROMPT,  # Standard-System-Prompt für internen Analyseaufruf
        reasoning_prompt,
        max_new_tokens=300,
    )

    return ShapResponse(
        response_text=RESPONSE_TEXT,
        overall_image=overall_img,
        system_image=system_img,
        instruction_image=instruction_img,
        input_image=input_img,
        top_words=top_words,
        word_annotations=annotations,
        reasoning=reasoning,
    )


# ── Konfidenz-Bewertung ───────────────────────────────────────────────────────
@app.post("/explain/confidence", response_model=ConfidenceResponse)
def explain_confidence(req: ConfidenceRequest):
    """
    Generiert eine Antwort und gibt Softmax-Wahrscheinlichkeiten pro Token zurück.
    Hohe Wahrscheinlichkeit = das Modell war bei diesem Token „sicher";
    niedrige Wahrscheinlichkeit = es war unsicher, was oft mit Halluzinationsrisiko korreliert.
    """
    model, tokenizer = get_model(instruct=True)

    # Vom Nutzer bereitgestellten System-Prompt verwenden (Fallback auf Standard)
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{req.system_prompt}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Instruction: {req.instruction}\n\n"
        f"Question: {req.input_text}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # out.sequences[0] enthält den Prompt; nur generierte Tokens extrahieren
    generated_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
    # out.scores ist ein Tupel von (1, vocab_size)-Tensoren, einen pro Schritt
    scores = out.scores

    token_confidences: list[TokenConfidence] = []
    probs_list: list[float] = []

    for token_id, score in zip(generated_ids, scores):
        tid  = token_id.item()
        # Bei EOS-Token stoppen
        if tid == tokenizer.eos_token_id:
            break
        prob = torch.softmax(score[0], dim=-1)[tid].item()
        text = tokenizer.decode([tid], skip_special_tokens=True)
        # Leere / rein aus Leerzeichen bestehende Tokens ohne semantischen Inhalt überspringen
        if not text:
            continue
        token_confidences.append(TokenConfidence(token=text, probability=prob))
        probs_list.append(prob)

    if not probs_list:
        probs_list = [0.0]

    return ConfidenceResponse(
        response_text=tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
        mean_confidence=float(np.mean(probs_list)),
        min_confidence=float(np.min(probs_list)),
        token_confidences=token_confidences,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)