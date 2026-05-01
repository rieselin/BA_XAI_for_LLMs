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

# --- TYPEOFAI COLUMNS ---
typeofai_cols = [
    "typeOfAI_gpt",
    "typeOfAI_alexa",
    "typeOfAI_stock_market",
    "typeOfAI_chessComputer",
    "typeOfAI_googleMaps",
    "typeOfAI_netflix_recommendations",
    "typeOfAI_selfDrivingCars",
    "typeOfAI_face_recognition",
    "typeOfAI_clock",
    "typeOfAI_gps_device",
    "typeOfAI_alarm_clock"
]

# Display labels (English, readable)
labels = [
    "ChatGPT",
    "Amazon Alexa / Voice Assistants",
    "Stock Market Prediction Algorithm",
    "Chess Computer",
    "Google Maps / Navigation Systems",
    "Netflix Series and Film Recommendations",
    "Self-Driving Cars / Autonomous Vehicles",
    "Face Recognition to Unlock Phone",
    "Analog Clock",
    "Classical GPS Device",
    "Simple Alarm Clock App"
]

# --- CLEAN & COUNT ---
for col in typeofai_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

n_total = len(df)
counts = [df[col].sum() for col in typeofai_cols]
percentages = [c / n_total * 100 for c in counts]

# --- SORT by percentage descending ---
sorted_pairs = sorted(zip(percentages, counts, labels), reverse=True)
percentages_sorted, counts_sorted, labels_sorted = zip(*sorted_pairs)

# --- COLOR: shade by percentage (darker = more selected) ---
cmap = plt.cm.Blues
norm_vals = np.array(percentages_sorted) / 100
bar_colors = [cmap(0.35 + 0.55 * v) for v in norm_vals]

# --- PLOT ---
fig, ax = plt.subplots(figsize=(10, 7))

bars = ax.barh(labels_sorted, percentages_sorted, color=bar_colors,
               edgecolor='white', linewidth=0.8, height=0.65)

# Annotate bars with count + percentage
for bar, pct, cnt in zip(bars, percentages_sorted, counts_sorted):
    ax.text(
        bar.get_width() + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{pct:.1f}%  (n={cnt})",
        va='center', ha='left', fontsize=9.5, color='#333333'
    )

ax.set_xlabel("% of Participants who selected this as AI", fontsize=11)
ax.set_title(f"\"Which of these do you consider a type of AI?\"\n(N={n_total} participants)",
             fontsize=13, pad=14)
ax.set_xlim(0, max(percentages_sorted) * 1.25)
ax.xaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.invert_yaxis()  # highest bar on top

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("typeofai_barplot.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: typeofai_barplot.png  (N={n_total})")

conn.close()