import sqlite3
import matplotlib.pyplot as plt
from collections import Counter
import textwrap

# ---- CONFIG ----
DB_FILE = "database.db"
COLUMN_NAME = "used_llm"   # column you want to analyze

# ---- CONNECT TO DB ----
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# ---- GET TABLES ----
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]

print("Tables found:", tables)

results = {}

# ---- SCAN TABLES FOR COLUMN ----
for table in tables:
    cursor.execute(f"PRAGMA table_info({table});")
    columns = [col[1] for col in cursor.fetchall()]
    
    if COLUMN_NAME in columns:
        print(f"Found '{COLUMN_NAME}' in table '{table}'")

        cursor.execute(f"SELECT {COLUMN_NAME} FROM {table} where {COLUMN_NAME} IS NOT 'NULL';")
        values = [row[0] for row in cursor.fetchall()]

        counter = Counter(values)
        results[table] = counter

# ---- CLOSE DB ----
conn.close()

# ---- PLOT RESULTS ----
for table, counter in results.items():
    labels = list(counter.keys())
    counts = list(counter.values())
    labels = [str(x) if x is not None else "NULL" for x in labels]

    plt.figure()
    plt.pie(counts, labels=["Yes", "No"], autopct='%1.1f%%')
    title = f"Have you used a generative AI tool (e.g., ChatGPT, Gemini, Copilot, Claude, LLaMA, Bard) at least once?"
    plt.title(textwrap.fill(title, width=40), wrap=True)

    # Save diagram
    filename = f"{table}_{COLUMN_NAME}_distribution.png"
    plt.savefig(filename)
    print(f"Saved diagram: {filename}")

    # Optional: show plot
    plt.show()