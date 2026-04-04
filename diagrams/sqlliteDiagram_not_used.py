import sqlite3
import matplotlib.pyplot as plt
import textwrap


# ---- CONFIG ----
DB_FILE = "database.db"

columns = [
    "llm_used_bard", "llm_used_chatGpt", "llm_used_claud", "llm_used_command", "llm_used_copilot",
 "llm_used_deepSeek", "llm_used_ernie", "llm_used_falcon", "llm_used_gemini", "llm_used_gemma",
 "llm_used_granite", "llm_used_grok", "llm_used_llama", "llm_used_mistral", "llm_used_nemotron", 
 "llm_used_nova", "llm_used_qwen", "llm_used_palm", "llm_used_perplexity","llm_used_other"
]

# ---- CONNECT ----
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# ---- BUILD QUERY ----
query = f"""
SELECT {", ".join(columns)}
FROM data_project
WHERE quota != 0 and know_llm = 1 and used_llm = 1
"""

cursor.execute(query)
rows = cursor.fetchall()

# ---- COUNT "1"s PER COLUMN ----
counts = {col: 0 for col in columns}

for row in rows:
    for col, value in zip(columns, row):
        if value == 1:
            counts[col] += 1

conn.close()

# ---- PREPARE DATA FOR PLOT ----
values = list(counts.values())

# Make labels shorter (optional, nicer chart)
labels = [
    "Bard (Google)", "ChatGPT (OpenAI)", "Claude (Anthropic)", "Command (Cohere)", "Copilot (Microsoft)",
    "DeepSeek (DeepSeek)", "Ernie (Baidu)", "Falcon (Falcon)", "Gemini (Google DeepMind)", "Gemma (Google)",
    "Granite (IBM)", "Grok (xAI)", "LLaMA (Meta AI)", "Mistral (Mistral AI)", "Nemotron (NVIDIA)",
    "Nova (Amazon)", "Qwen (Alibaba)", "PaLM (Google AI)", "Perplexity (Perplexity)", "Other"
]

wrapped_labels = [textwrap.fill(label, 20) for label in labels]
# ---- PLOT ----
plt.figure(figsize=(12, 6))
plt.bar(wrapped_labels, values)

plt.title("Which of these LLMs have you tried so far?")
plt.xlabel("LLM")
plt.ylabel("Count")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("not_used_llm_reasons.png")
plt.show()