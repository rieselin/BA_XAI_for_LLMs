import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns

# --- DB CONNECTION ---
conn = sqlite3.connect("database.db")  # <-- change this

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

# --- FILTER COMPLETE ROWS ---
all_cols = sum(sets.values(), [])
df = df.dropna(subset=all_cols)

# --- MAP TEXT TO NUMERIC ---
for col in all_cols:
    df[col] = df[col].map(mapping)

# --- BUILD RESULT TABLE ---
aspects = ["I trust that these answers are\n factually accurate.", "I consider the information in these\n  answers to be reliable.",
            "These answers come across as\n  competent and knowledgeable.", "I believe the system has tried\n  to provide accurate answers.",
              "I would rely on the information\n  contained in these answers."]
result = pd.DataFrame(index=aspects, columns=sets.keys())

for set_name, cols in sets.items():
    result[set_name] = df[cols].mean().values

# --- HEATMAP ---
plt.figure()

sns.heatmap(
    result.astype(float),
    annot=True,
    vmin=1,
    vmax=5,
    cmap="coolwarm"
)

plt.title("Trust Across Explanation Types")
plt.xlabel("Method")
plt.ylabel("Aspect")

plt.show()