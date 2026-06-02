import pandas as pd
import numpy as np
import sqlite3
from scipy import stats
from itertools import combinations

# ─────────────────────────────────────────────
# DB CONNECTION & LOAD
# ─────────────────────────────────────────────
conn = sqlite3.connect("../survey/database.db")
df = pd.read_sql("SELECT * FROM data_project", conn)
conn.close()

# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────
df = df[(df['used_llm'] != 0) & (df['quota'] != 0)]

# ─────────────────────────────────────────────
# MAPPING
# ─────────────────────────────────────────────
mapping = {
    "Stimme überhaupt nicht zu": 1,
    "Stimme nicht zu":           2,
    "Weder noch":                3,
    "Stimme zu":                 4,
    "Stimme voll zu":            5
}

# ─────────────────────────────────────────────
# COLUMN GROUPS
# ─────────────────────────────────────────────
sets = {
    "LLM":  [
        "llm_trust_that_factual",
        "llm_trust_that_reliable",
        "llm_trust_that_competent",
        "llm_trust_that_tried_to_answer_correct",
        "llm_trust_that_on_answer"
    ],
    "CoT":  [
        "cot_trust_that_factual",
        "cot_trust_that_reliable",
        "cot_trust_that_competent",
        "cot_trust_that_tried_to_answer_correct",
        "cot_trust_that_on_answer"
    ],
    "SHAP": [
        "shap_trust_that_factual",
        "shap_trust_that_reliable",
        "shap_trust_that_competent",
        "shap_trust_that_tried_to_answer_correct",
        "shap_trust_that_on_answer"
    ],
    "CONF": [
        "conf_trust_that_factual",
        "conf_trust_that_reliable",
        "conf_trust_that_competent",
        "conf_trust_that_tried_to_answer_correct",
        "conf_trust_that_on_answer"
    ],
    "EXPL": [
        "explanation_trust_that_factual",
        "explanation_trust_that_reliable",
        "explanation_trust_that_competent",
        "explanation_trust_that_tried_to_answer_correct",
        "explanation_trust_that_on_answer"
    ]
}

# ─────────────────────────────────────────────
# CLEAN TRUST COLUMNS
# ─────────────────────────────────────────────
all_trust_cols = sum(sets.values(), [])
for col in all_trust_cols:
    df[col] = df[col].replace('NULL', np.nan).map(mapping)
    df[col] = pd.to_numeric(df[col], errors='coerce').clip(1, 5)

# ─────────────────────────────────────────────
# COMPUTE MEAN TRUST PER METHOD
# ─────────────────────────────────────────────
for method, cols in sets.items():
    df[f'trust_{method}'] = df[cols].mean(axis=1)

methods      = list(sets.keys())
trust_cols   = [f'trust_{m}' for m in methods]

# XAI methods = everything except the LLM baseline
baseline_col = 'trust_LLM'
xai_methods  = [m for m in methods if m != 'LLM']
xai_cols     = [f'trust_{m}' for m in xai_methods]

# ─────────────────────────────────────────────
# HELPER: Cronbach's Alpha
# ─────────────────────────────────────────────
def cronbach_alpha(data: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for a DataFrame of items (rows = subjects)."""
    data   = data.dropna()
    k      = data.shape[1]
    item_variances = data.var(axis=0, ddof=1).sum()
    total_variance = data.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_variances / total_variance)

# ─────────────────────────────────────────────
# HELPER: Holm-corrected p-values
# ─────────────────────────────────────────────
def holm_correction(p_values: list) -> list:
    """Return Holm-corrected p-values (same order as input)."""
    n   = len(p_values)
    idx = np.argsort(p_values)
    corrected = np.empty(n)
    running_max = 0.0
    for rank, i in enumerate(idx):
        adjusted     = p_values[i] * (n - rank)
        running_max  = max(running_max, adjusted)
        corrected[i] = min(running_max, 1.0)
    return corrected.tolist()

# ═════════════════════════════════════════════
# TEIL 1: CRONBACH'S ALPHA — Skalenzulässigkeit
# ═════════════════════════════════════════════
print("=" * 60)
print("TEIL 1: CRONBACH'S ALPHA — Ist die Trust-Skala zulässig?")
print("=" * 60)

alpha_results = {}
for method, cols in sets.items():
    alpha = cronbach_alpha(df[cols])
    alpha_results[method] = alpha

# Overall alpha across all 25 items
alpha_overall = cronbach_alpha(df[all_trust_cols])
alpha_results['OVERALL (all 25 items)'] = alpha_overall

for label, alpha in alpha_results.items():
    verdict = (
        "ausgezeichnet (≥ .90)" if alpha >= 0.90 else
        "gut           (≥ .80)" if alpha >= 0.80 else
        "akzeptabel    (≥ .70)" if alpha >= 0.70 else
        "fragwürdig    (≥ .60)" if alpha >= 0.60 else
        "schlecht      (< .60)"
    )
    print(f"  {label:<28} α = {alpha:.3f}  →  {verdict}")

print()

# ═════════════════════════════════════════════
# TEIL 2: FRIEDMAN-TEST + WILCOXON POST-HOC (HOLM)
# Baseline (LLM) vs. XAI-Methoden
# ═════════════════════════════════════════════
print("=" * 60)
print("TEIL 2: FRIEDMAN-TEST — Trust: Baseline vs. XAI-Methoden")
print("=" * 60)

# Keep only rows complete across all five methods
complete = df[trust_cols].dropna()
n_complete = len(complete)
print(f"  Vollständige Fälle (alle 5 Methoden): N = {n_complete}\n")

friedman_data = [complete[c].values for c in trust_cols]
stat_f, p_f = stats.friedmanchisquare(*friedman_data)
print(f"  Friedman χ²({len(methods) - 1}) = {stat_f:.3f},  p = {p_f:.4f}  "
      f"({'** signifikant' if p_f < 0.05 else 'nicht signifikant'} bei α = .05)\n")

# Descriptive: means & medians per method
print("  Deskriptive Statistiken:")
print(f"  {'Methode':<8} {'M':>6} {'Mdn':>6} {'SD':>6}")
for m in methods:
    col = f'trust_{m}'
    sub = complete[col]
    print(f"  {m:<8} {sub.mean():>6.3f} {sub.median():>6.3f} {sub.std():>6.3f}")
print()

# ── Post-hoc: Wilcoxon signed-rank (pairwise) with Holm correction ──────────
print("  Wilcoxon Post-hoc-Tests (paarweise, Holm-korrigiert):")
pairs     = list(combinations(methods, 2))
raw_stats = []
raw_ps    = []
for (a, b) in pairs:
    s, p = stats.wilcoxon(complete[f'trust_{a}'], complete[f'trust_{b}'],
                          alternative='two-sided')
    raw_stats.append(s)
    raw_ps.append(p)

adj_ps = holm_correction(raw_ps)

header = f"  {'Paar':<18} {'W':>9} {'p (roh)':>10} {'p (Holm)':>10}  Signifikanz"
print(header)
print("  " + "-" * (len(header) - 2))
for (a, b), W, p_raw, p_adj in zip(pairs, raw_stats, raw_ps, adj_ps):
    sig = "**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "n.s.")
    print(f"  {a} vs {b:<12} {W:>9.1f} {p_raw:>10.4f} {p_adj:>10.4f}  {sig}")
print()
print("Fertig.")