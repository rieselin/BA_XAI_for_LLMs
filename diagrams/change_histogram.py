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

# --- COMPUTE CHANGE ---
df["LLM_mean"] = df[sets["LLM"]].mean(axis=1)
df["EXPL_mean"] = df[sets["EXPL"]].mean(axis=1)

df["change"] = df["EXPL_mean"] - df["LLM_mean"]

# --- PLOT ---
plt.figure()

plt.hist(df["change"], bins=10)

plt.xlabel("Change in Trust (EXPL - LLM)")
plt.ylabel("Number of People")
plt.title("How Trust Changed After Explanation")

plt.show()