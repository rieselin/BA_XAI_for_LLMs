import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

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

# --- CLEAN DATA ---
all_cols = sum(sets.values(), [])
df = df.dropna(subset=all_cols)

for col in all_cols:
    df[col] = df[col].map(mapping)

# --- PLOT ---
plt.figure()

for i in df.index:
    values = []
    for set_name, cols in sets.items():
        values.append(df.loc[i, cols].mean())
    
    plt.plot(list(sets.keys()), values, alpha=0.2)

plt.ylabel("Trust Level")
plt.title("Individual Trust Trajectories")

plt.show()