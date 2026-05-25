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

# --- PLOT: LLM usage per schooling ---
school_counts = df.groupby('schooling')['used_llm'].sum()

schooling_order = ["Kein Abschluss", "Hauptschulabschluss", "Realschulabschluss", "Fachhochschulreife", "Abitur", "Bachelor-Abschluss", "Master-Abschluss"]

school_counts = school_counts.reindex(schooling_order)
school_counts.plot(kind='bar', figsize=(10,6), color='lightgreen')
plt.xlabel("Education Level")
plt.ylabel("Number of LLM Users")
plt.title("LLM Usage per Education Level")
plt.xticks(rotation=45)
plt.show()

# --- LABEL MAPPINGS ---
employment_role_labels = {
    "employment_role_ceo": "CEO / Executive",
    "employment_role_management": "Management / Executive",
    "employment_role_expert": "Specialist / Expert",
    "employment_role_project_manager": "Project Manager",
    "employment_role_ingenieur": "Technical Staff / Engineer",
    "employment_role_sale": "Sales",
    "employment_role_marketing": "Marketing / Communications",
    "employment_role_administration": "Administration",
    "employment_role_research": "Research & Development",
    "employment_role_production": "Production / Operations",
    "employment_role_software": "IT / Software / Data",
    "employment_role_consulting": "Consulting",
    "employment_role_education": "Education / Training",
    "employment_role_self_employed": "Self-Employed / Entrepreneur",
    "employment_role_freelancing": "Freelancer",
    "employment_role_student": "Student / In Training",
    "employment_role_searching_for_employment": "Job Seeker",
    "employment_role_retired": "Retired",
    "employment_role_not_employed": "Not employed",
    "employment_role_employed_not_declared": "Employed (not declared)",
    "employment_role_other": "Other"
}

sector_labels = {
    "sector_software": "Information Technology / Software",
    "sector_telecommunication": "Telecommunications",
    "sector_finance": "Financial Services / Banking / Insurance",
    "sector_consulting": "Consulting",
    "sector_production": "Industry / Manufacturing",
    "sector_engineering": "Mechanical Engineering / Engineering",
    "sector_construction": "Construction / Architecture / Real Estate",
    "sector_commerce": "Retail / E-Commerce",
    "sector_marketing": "Marketing / Media / Advertising",
    "sector_health": "Healthcare / Pharmaceuticals / Biotechnology",
    "sector_education": "Education / Research",
    "sector_public_sector": "Public Sector / Administration",
    "sector_environment": "Energy / Environment / Sustainability",
    "sector_transportation": "Transport / Logistics",
    "sector_tourism": "Tourism / Hospitality",
    "sector_agriculture": "Agriculture / Food",
    "sector_design": "Creative Industries / Design",
    "sector_non_profit": "Non-Profit / NGO",
    "sector_na": "Not Specified / Not Applicable",
    "sector_other": "Other"
}

# --- GET EMPLOYMENT ROLE COLUMNS ---
roles = [col for col in df.columns if col.startswith('employment_role_')]
for col in roles:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

role_counts = df[roles].sum().sort_values(ascending=False)
role_counts.index = role_counts.index.map(lambda x: employment_role_labels.get(x, x))

role_counts.plot(kind='barh', figsize=(12,8), color='coral')
plt.xlabel("Number of LLM Users", fontsize=14)
plt.title("LLM Usage by Employment Role", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()
plt.show()

# --- GET SECTOR COLUMNS ---
sectors = [col for col in df.columns if col.startswith('sector_')]
for col in sectors:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

sector_counts = df[sectors].sum().sort_values(ascending=True)
sector_counts.index = sector_counts.index.map(lambda x: sector_labels.get(x, x))

sector_counts.plot(kind='barh', figsize=(12,8), color='orchid')
plt.xlabel("Number of LLM Users")
plt.title("LLM Usage by Sector")
plt.show()