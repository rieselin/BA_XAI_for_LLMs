import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import seaborn as sns

# --- DB CONNECTION ---
conn = sqlite3.connect("../survey/database.db")

# --- LOAD DATA ---
df = pd.read_sql("SELECT * FROM data_project", conn)

# --- FILTER ---
df = df[(df['used_llm'] != 0) & (df['quota'] != 0)]

# --- MAP TRUST SCORES ---
mapping = {
    "Stimme überhaupt nicht zu": 1,
    "Stimme nicht zu": 2,
    "Weder noch": 3,
    "Stimme zu": 4,
    "Stimme voll zu": 5
}

llm_trust_cols = [
    "llm_trust_that_factual",
    "llm_trust_that_reliable",
    "llm_trust_that_competent",
    "llm_trust_that_tried_to_answer_correct",
    "llm_trust_that_on_answer"
]

aspects = ["I trust that these answers are\n factually accurate.", "I consider the information in these\n  answers to be reliable.",
            "These answers come across as\n  competent and knowledgeable.", "I believe the system has tried\n  to provide accurate answers.",
              "I would rely on the information\n  contained in these answers."]

for col in llm_trust_cols:
    df[col] = df[col].replace('NULL', np.nan).map(mapping)
    # Drop rows where value is missing or invalid
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].clip(1,5)  # <-- ensures all values are between 1 and 5

# --- COUNT NUMBER OF LLMs USED ---
llm_cols = [col for col in df.columns if col.startswith('llm_used_')]

for col in llm_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

df['num_llms_used'] = df[llm_cols].sum(axis=1)

# --- FILTER OUT COLUMNS WITH NaN VALUES ---
llm_trust_cols_clean = [col for col in llm_trust_cols if df[col].notna().all()]
aspects_clean = [aspects[i] for i, col in enumerate(llm_trust_cols) if df[col].notna().all()]
sns.set(style="whitegrid")

fig, axes = plt.subplots(2, 3, figsize=(18,10), sharey=True)
axes = axes.flatten()  # makes indexing easier

for i, col in enumerate(llm_trust_cols_clean):
    sns.violinplot(
        x=df['num_llms_used'],
        y=df[col],
        ax=axes[i],
        palette="Set2",
        inner="quartile",
        bw=0.1
    )

    axes[i].set_title(aspects_clean[i], fontsize=10)
    axes[i].set_xlabel("Number of LLMs Used", fontsize=9)

    if i % 3 == 0:
        axes[i].set_ylabel("Trust Level (1-5)", fontsize=10)
    else:
        axes[i].set_ylabel("")

    axes[i].set_ylim(0, 6)

# Hide the unused 6th subplot
axes[-1].set_visible(False)

plt.suptitle(
    "Distribution of Trust Levels per Number of LLMs Used",
    fontsize=14
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()