import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import seaborn as sns
from scipy import stats

# --- DB CONNECTION ---
conn = sqlite3.connect("../survey/database.db")

# --- LOAD DATA ---
df = pd.read_sql("SELECT * FROM data_project", conn)

# --- FILTER ---
df = df[(df['used_llm'] != 0) & (df['quota'] != 0)]

# --- MAPPING ---
mapping = {
    "Stimme überhaupt nicht zu": 1,
    "Stimme nicht zu": 2,
    "Weder noch": 3,
    "Stimme zu": 4,
    "Stimme voll zu": 5
}

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

# --- CLEAN TRUST COLUMNS ---
all_trust_cols = sum(sets.values(), [])
for col in all_trust_cols:
    df[col] = df[col].replace('NULL', np.nan).map(mapping)
    df[col] = pd.to_numeric(df[col], errors='coerce').clip(1, 5)

# --- CLEAN AGE ---
# Age is stored as German strings e.g. "25-34 Jahre", "65 Jahre oder älter"
df['age'] = df['age'].replace('NULL', np.nan).str.strip()
df = df.dropna(subset=['age'])

print("Age unique values:", df['age'].unique())

# Canonical order for display
age_order = [
    "18-24 Jahre",
    "25-34 Jahre",
    "35-44 Jahre",
    "45-54 Jahre",
    "55-64 Jahre",
    "65 Jahre oder älter"
]
# Only keep categories that actually appear in the data
age_order = [a for a in age_order if a in df['age'].values]
df['age_group'] = pd.Categorical(df['age'], categories=age_order, ordered=True)
df = df.dropna(subset=['age_group'])

# Short display labels for x-axis
short_labels = {
    "18-24 Jahre": "18–24",
    "25-34 Jahre": "25–34",
    "35-44 Jahre": "35–44",
    "45-54 Jahre": "45–54",
    "55-64 Jahre": "55–64",
    "65 Jahre oder älter": "65+"
}
display_labels = [short_labels.get(a, a) for a in age_order]

# Numeric midpoints for correlation / regression
midpoint_map = {
    "18-24 Jahre": 21,
    "25-34 Jahre": 30,
    "35-44 Jahre": 40,
    "45-54 Jahre": 50,
    "55-64 Jahre": 60,
    "65 Jahre oder älter": 70
}
df['age_numeric'] = df['age'].map(midpoint_map)

# --- COMPUTE MEAN TRUST PER METHOD ---
for method, cols in sets.items():
    df[f'trust_{method}'] = df[cols].mean(axis=1)

trust_mean_cols = [f'trust_{m}' for m in sets]

# ============================================================
# PLOT 1: Box plot — mean trust per method by age group
# ============================================================
methods = list(sets.keys())
colors = plt.cm.Set2.colors

# 2 rows × 3 columns
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()

for i, method in enumerate(methods):
    col = f'trust_{method}'

    data_by_age = [
        df[df['age_group'] == g][col].dropna().values
        for g in age_order
    ]

    bp = axes[i].boxplot(
        data_by_age,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color='black', linewidth=1.5)
    )

    for patch, c in zip(
        bp['boxes'],
        [colors[j % len(colors)] for j in range(len(age_order))]
    ):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)

    axes[i].set_xticks(range(1, len(age_order) + 1))
    axes[i].set_xticklabels(
        display_labels,
        rotation=30,
        ha='right',
        fontsize=9
    )

    axes[i].set_title(method, fontsize=12)
    axes[i].set_xlabel("Age Group")
    axes[i].set_ylim(0.5, 5.5)
    axes[i].yaxis.grid(True, linestyle='--', alpha=0.7)
    axes[i].set_axisbelow(True)

    # ylabel only on left column
    if i % 3 == 0:
        axes[i].set_ylabel("Mean Trust Score (1–5)")
    else:
        axes[i].set_ylabel("")

# Hide unused subplot
axes[-1].set_visible(False)

fig.suptitle("Mean Trust per Method by Age Group", fontsize=14, y=0.98)

plt.tight_layout()
plt.savefig("age_trust_boxplot.png", dpi=150, bbox_inches='tight')
plt.show()

print("Saved: age_trust_boxplot.png")

# ============================================================
# PLOT 2: Heatmap — Pearson r between age and trust per method
# ============================================================
corr_results = {}
for method in methods:
    col = f'trust_{method}'
    valid = df[['age_numeric', col]].dropna()
    r, p = stats.pearsonr(valid['age_numeric'], valid[col])
    corr_results[method] = {'r': round(r, 3), 'p': round(p, 4), 'n': len(valid)}
    print(f"{method}: r={r:.3f}, p={p:.4f}, n={len(valid)}")

r_values = np.array([[corr_results[m]['r'] for m in methods]])
p_values = np.array([[corr_results[m]['p'] for m in methods]])

fig, ax = plt.subplots(figsize=(10, 3))
sns.heatmap(
    r_values,
    annot=True,
    fmt=".3f",
    cmap='coolwarm',
    center=0,
    vmin=-1, vmax=1,
    xticklabels=methods,
    yticklabels=['Age'],
    ax=ax,
    linewidths=0.5,
    cbar_kws={'label': 'Pearson r'}
)

# Annotate significance
for j in range(len(methods)):
    p = p_values[0, j]
    stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    ax.text(j + 0.5, 0.85, stars, ha='center', va='center', fontsize=11,
            color='white' if abs(r_values[0, j]) > 0.3 else 'black')

ax.set_title("Pearson Correlation: Age × Mean Trust per Method\n(* p<0.05, ** p<0.01, *** p<0.001)", fontsize=12)
plt.tight_layout()
plt.savefig("age_trust_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: age_trust_heatmap.png")

# ============================================================
# PLOT 3: Scatter with regression line — age vs. mean overall trust
# ============================================================
df['trust_overall'] = df[trust_mean_cols].mean(axis=1)
valid_scatter = df[['age_numeric', 'trust_overall']].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(valid_scatter['age_numeric'], valid_scatter['trust_overall'],
           alpha=0.3, s=20, color='steelblue', label='Participant')

slope, intercept, r, p, se = stats.linregress(valid_scatter['age_numeric'], valid_scatter['trust_overall'])
x_range = np.linspace(valid_scatter['age_numeric'].min(), valid_scatter['age_numeric'].max(), 200)
ax.plot(x_range, intercept + slope * x_range, color='crimson', linewidth=2,
        label=f'Regression (r={r:.3f}, p={p:.4f})')

ax.set_xticks(list(midpoint_map.values()))
ax.set_xticklabels(list(short_labels.values()), rotation=20, ha='right')
ax.set_xlabel("Age Group")
ax.set_ylabel("Mean Overall Trust (1–5)")
ax.set_title("Age vs. Overall Mean Trust (all methods combined)")
ax.legend()
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("age_trust_scatter.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: age_trust_scatter.png")

conn.close()
