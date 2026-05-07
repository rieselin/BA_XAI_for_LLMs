import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np

# --- DB CONNECTION ---
conn = sqlite3.connect("../survey/database.db")

# --- LOAD DATA ---
df = pd.read_sql("SELECT * FROM data_project", conn)

# --- FILTER ---
df = df[(df['used_llm'] != 0) & (df['quota'] != 0)]

# --- CLEAN trust_changed ---
df['trust_changed'] = df['trust_changed'].replace('NULL', np.nan).str.strip()

# Normalize to canonical values
value_map = {
    '0': 'No',
    '1': 'Yes',
    'Unsicher': 'Unsure',
    0: 'No',
    1: 'Yes',
}
df['trust_changed'] = df['trust_changed'].map(value_map)

print("Value counts:\n", df['trust_changed'].value_counts(dropna=False))

n_total = len(df)
n_valid = df['trust_changed'].notna().sum()

# --- COUNT ---
order = ['Yes', 'Unsure', 'No']
colors = {
    'Yes': '#4C9F70',     # green
    'Unsure': '#F0A500', # amber
    'No': '#D94F3D'      # red
}

counts = [ (df['trust_changed'] == cat).sum() for cat in order ]

# --- PLOT PIE CHART ---
fig, ax = plt.subplots(figsize=(7, 6))

wedges, texts, autotexts = ax.pie(
    counts,
    labels=order,
    colors=[colors[c] for c in order],
    autopct=lambda pct: f"{pct:.1f}%",
    startangle=140,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1},
    textprops={'fontsize': 12, 'color': '#333333'}
)

ax.set_title(
    "Has this survey changed your confidence in LLMs?",
    fontsize=14,
    pad=16
)

ax.axis('equal')  # Keep circle shape

plt.tight_layout()
plt.savefig("trust_changed_piechart.png", dpi=150, bbox_inches='tight')
plt.show()

print("Saved: trust_changed_piechart.png")

conn.close()