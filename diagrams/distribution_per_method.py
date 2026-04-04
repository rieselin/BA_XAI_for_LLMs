import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np

# --- DB CONNECTION ---
conn = sqlite3.connect("database.db")

# --- LOAD DATA ---
df = pd.read_sql("SELECT * FROM data_project", conn)

# --- MAPPING ---
mapping = {
    "Stimme überhaupt nicht zu": 1,
    "Stimme nicht zu": 2,
    "Weder noch": 3,
    "Stimme zu": 4,
    "Stimme voll zu": 5
}

test_col = "llm_trust_that_factual"

print("Raw unique values:")
print(df[test_col].unique())

print("\nAfter map:")
print(df[test_col].map(mapping).value_counts(dropna=False))


# --- COLUMN GROUPS ---
sets = {
    "LLM": [
        "llm_trust_that_factual",
        "llm_trust_that_reliable",
        "llm_trust_that_competent",
        "llm_trust_that_tried_to_answer_correct",
        "llm_trust_that_on_answer"
    ],
    "CoT": [
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

aspects = ["I trust that these answers are factually accurate.", "I consider the information in these answers to be reliable.",
            "These answers come across as competent and knowledgeable.", "I believe the system has tried to provide accurate answers.",
              "I would rely on the information contained in these answers."]

# --- CLEAN DATA ---
all_cols = sum(sets.values(), [])

for col in all_cols:
    df[col] = df[col].replace('NULL', np.nan)  # <-- treat 'NULL' string as missing
    df[col] = df[col].map(mapping)


# --- PLOT: grouped box plot, one group per method, one box per aspect ---
methods = list(sets.keys())
n_methods = len(methods)
n_aspects = len(aspects)

fig, ax = plt.subplots(figsize=(16, 6))

group_width = n_aspects + 1.5
colors = plt.cm.Set2.colors

positions = []
data = []
colors_list = []

for i, method in enumerate(methods):
    base = i * group_width
    for j, col in enumerate(sets[method]):
        # Map and drop NaN per column individually
        col_data = df[col].dropna().values   # <-- per-column, not per-row
        pos = base + j
        positions.append(pos)
        data.append(col_data)
        colors_list.append(colors[j])
        print(f"{method} - {aspects[j]}: {len(col_data)} responses")

bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6,
                medianprops=dict(color="black", linewidth=1.5))

for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

method_centers = [i * group_width + (n_aspects - 1) / 2 for i in range(n_methods)]
ax.set_xticks(method_centers)
ax.set_xticklabels(methods, fontsize=12)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[j], alpha=0.8, label=aspects[j]) for j in range(n_aspects)]
ax.legend(handles=legend_elements, title="Aspect", bbox_to_anchor=(1.01, 1), loc='upper left')

ax.set_ylabel("Trust Level (1–5)")
ax.set_title("Trust per Method and Aspect")
ax.set_ylim(0.5, 5.5)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()