import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch

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

aspects = [
    "Factually accurate",
    "Reliable",
    "Competent & knowledgeable",
    "Tried to answer correctly",
    "Would rely on answers"
]

# --- CLEAN TRUST COLUMNS ---
all_trust_cols = sum(sets.values(), [])
for col in all_trust_cols:
    df[col] = df[col].replace('NULL', np.nan).map(mapping)
    df[col] = pd.to_numeric(df[col], errors='coerce').clip(1, 5)

# --- CLEAN GENDER ---
# Values are German strings: "männlich", "weiblich", (possibly "divers" etc.)
df['gender'] = df['gender'].replace('NULL', np.nan).str.strip().str.lower()
print("Gender unique values:", df['gender'].unique())

df = df.dropna(subset=['gender'])

# Canonical order — adjust if your DB has additional categories
gender_order = ['männlich', 'weiblich', 'divers', 'keine Angabe']
gender_labels = [g for g in gender_order if g in df['gender'].values]
df = df[df['gender'].isin(gender_labels)]

print("\nGender distribution:\n", df['gender'].value_counts())

# --- COMPUTE MEAN TRUST PER METHOD ---
for method, cols in sets.items():
    df[f'trust_{method}'] = df[cols].mean(axis=1)

methods = list(sets.keys())
trust_mean_cols = [f'trust_{m}' for m in methods]
df['trust_overall'] = df[trust_mean_cols].mean(axis=1)
n_genders = len(gender_labels)

# ============================================================
# PLOT 1: Grouped box plot — mean trust per method by gender
# ============================================================
colors = plt.cm.Set2.colors

# 2 rows × 3 columns
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()

for i, method in enumerate(methods):
    col = f'trust_{method}'

    data_by_gender = [
        df[df['gender'] == g][col].dropna().values
        for g in gender_labels
    ]

    bp = axes[i].boxplot(
        data_by_gender,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color='black', linewidth=1.5)
    )

    for patch, c in zip(
        bp['boxes'],
        [colors[j % len(colors)] for j in range(n_genders)]
    ):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)

    axes[i].set_xticks(range(1, n_genders + 1))
    axes[i].set_xticklabels(
        gender_labels,
        rotation=20,
        ha='right',
        fontsize=9
    )

    axes[i].set_title(method, fontsize=12)
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

fig.suptitle(
    "Mean Trust per Method by Gender",
    fontsize=14,
    y=0.98
)

plt.tight_layout()
plt.savefig("gender_trust_boxplot.png", dpi=150, bbox_inches='tight')
plt.show()

print("Saved: gender_trust_boxplot.png")

# ============================================================
# PLOT 2: Heatmap — mean trust score per gender × method
# ============================================================
heatmap_data = pd.DataFrame(index=gender_labels, columns=methods, dtype=float)
for g in gender_labels:
    for method in methods:
        heatmap_data.loc[g, method] = df[df['gender'] == g][f'trust_{method}'].mean()

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(
    heatmap_data.astype(float),
    annot=True,
    fmt=".2f",
    cmap='YlOrRd',
    vmin=1, vmax=5,
    ax=ax,
    linewidths=0.5,
    cbar_kws={'label': 'Mean Trust (1–5)'}
)
ax.set_title("Mean Trust Score by Gender and Method", fontsize=12)
ax.set_xlabel("Method")
ax.set_ylabel("Gender")
plt.tight_layout()
plt.savefig("gender_trust_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: gender_trust_heatmap.png")

# ============================================================
# PLOT 3: Violin plot — overall trust distribution by gender
# ============================================================
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(
    x='gender',
    y='trust_overall',
    data=df,
    palette='Set2',
    inner='quartile',
    order=gender_labels,
    ax=ax
)
ax.set_title("Overall Trust Distribution by Gender (all methods combined)", fontsize=12)
ax.set_xlabel("Gender")
ax.set_ylabel("Mean Overall Trust (1–5)")
ax.set_ylim(0.5, 5.5)
plt.tight_layout()
plt.savefig("gender_trust_violin.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: gender_trust_violin.png")

# ============================================================
# PLOT 4: Bar chart per method — aspect-level breakdown by gender
# ============================================================
# 2 rows × 3 columns
fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
axes = axes.flatten()

bar_colors = [colors[j % len(colors)] for j in range(n_genders)]
x = np.arange(len(aspects))
bar_width = 0.8 / n_genders

for i, method in enumerate(methods):
    cols = sets[method]

    for j, g in enumerate(gender_labels):
        group = df[df['gender'] == g]
        means = [group[col].mean() for col in cols]

        offset = (j - n_genders / 2 + 0.5) * bar_width

        axes[i].bar(
            x + offset,
            means,
            width=bar_width,
            label=g,
            color=bar_colors[j],
            alpha=0.85
        )

    axes[i].set_xticks(x)
    axes[i].set_xticklabels(
        aspects,
        rotation=35,
        ha='right',
        fontsize=12
    )

    axes[i].set_title(method, fontsize=11)
    axes[i].set_ylim(1, 5)
    axes[i].yaxis.grid(True, linestyle='--', alpha=0.6)
    axes[i].set_axisbelow(True)

    # ylabel only on left column
    if i % 3 == 0:
        axes[i].set_ylabel("Mean Trust Score (1–5)")
    else:
        axes[i].set_ylabel("")

# Hide unused subplot
axes[-1].set_visible(False)

legend_elements = [
    Patch(facecolor=bar_colors[j], alpha=0.85, label=g)
    for j, g in enumerate(gender_labels)
]

fig.legend(
    handles=legend_elements,
    title="Gender",
    bbox_to_anchor=(1.01, 0.7),
    loc='upper left'
)

fig.suptitle(
    "Mean Trust per Aspect, Method, and Gender",
    fontsize=13,
    y=0.98
)

plt.tight_layout()
plt.savefig("gender_trust_aspect_bars.png", dpi=150, bbox_inches='tight')
plt.show()

print("Saved: gender_trust_aspect_bars.png")
# ============================================================
# STATISTICAL TEST: Mann-Whitney U (or Kruskal-Wallis for 3+)
# ============================================================
print("\n--- Statistical Tests (trust_overall by gender) ---")
if n_genders == 2:
    g1 = df[df['gender'] == gender_labels[0]]['trust_overall'].dropna()
    g2 = df[df['gender'] == gender_labels[1]]['trust_overall'].dropna()
    stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    print(f"Mann-Whitney U: U={stat:.1f}, p={p:.4f} "
          f"({'significant' if p < 0.05 else 'not significant'} at α=0.05)")
else:
    groups = [df[df['gender'] == g]['trust_overall'].dropna().values for g in gender_labels]
    stat, p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis: H={stat:.3f}, p={p:.4f} "
          f"({'significant' if p < 0.05 else 'not significant'} at α=0.05)")

print("\n--- Mean trust_overall per gender ---")
print(df.groupby('gender')['trust_overall'].agg(['mean', 'median', 'std', 'count']))

conn.close()
