import sqlite3
import matplotlib.pyplot as plt
from collections import Counter
import textwrap

# ---- CONFIG ----
DB_FILE = "../survey/database.db"
FIGSIZE = (5, 4)  # Both charts will use this exact size
DPI = 150

CHARTS = [
    {
        "column": "used_llm",
        "title": "Have you used a generative AI tool (e.g., ChatGPT, Gemini, Copilot, Claude, LLaMA, Bard) at least once?",
        "output": "used_llm_distribution.png",
    },
    {
        "column": "know_llm",
        "title": "Are you familiar with generative AI tools such as ChatGPT, Gemini, Copilot, Claude, LLaMA, and Bard?",
        "output": "know_llm_distribution.png",
    },
]

# ---- CONNECT TO DB ----
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]
print("Tables found:", tables)

def get_counts(cursor, tables, column):
    """Aggregate counts for a column across all tables that contain it."""
    total = Counter()
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [col[1] for col in cursor.fetchall()]
        if column in columns:
            print(f"  Found '{column}' in table '{table}'")
            cursor.execute(
                f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL AND {column} != 'NULL';"
            )
            values = [row[0] for row in cursor.fetchall()]
            total.update(Counter(values))
    return total

# ---- GENERATE CHARTS ----
for chart in CHARTS:
    print(f"\nProcessing column: {chart['column']}")
    counter = get_counts(cursor, tables, chart["column"])

    if not counter:
        print(f"  No data found for '{chart['column']}', skipping.")
        continue

    print(f"  Raw values in DB: {dict(counter)}")

    # Map whatever the DB stores to Yes/No display labels
    def to_label(val):
        if str(val).strip().lower() in ("1", "true", "yes"):
            return "Yes"
        if str(val).strip().lower() in ("0", "false", "no"):
            return "No"
        return str(val)

    normalized = Counter()
    for val, count in counter.items():
        normalized[to_label(val)] += count

    labels_order = ["Yes", "No"]
    counts = [normalized.get(label, 0) for label in labels_order]
    # Drop labels with zero count
    pairs = [(l, c) for l, c in zip(labels_order, counts) if c > 0]
    labels, counts = zip(*pairs) if pairs else ([], [])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(textwrap.fill(chart["title"], width=45), fontsize=9, pad=12)

    plt.tight_layout()
    plt.savefig(chart["output"], dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {chart['output']}")

conn.close()
print("\nDone! Both images are saved and ready to tile side by side.")
