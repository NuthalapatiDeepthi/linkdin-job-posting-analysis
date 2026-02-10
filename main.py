import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

DATA_FILE = "jobs.csv"

SKILLS = [
    "python",
    "sql",
    "excel",
    "power bi",
    "tableau",
    "machine learning",
    "aws",
    "deep learning",
    "pandas",
    "numpy"
]

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("\nDataset Loaded Successfully!")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    return df

def clean_data(df):
    df.columns = df.columns.str.lower().str.strip()
    df.drop_duplicates(inplace=True)

    if "job_description" not in df.columns:
        raise ValueError("Column 'job_description' not found in dataset.")

    df.dropna(subset=["job_description"], inplace=True)
    return df

def extract_skills(text):
    found = []
    text = str(text).lower()

    for skill in SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found.append(skill)

    return found

def analyze_skills(df):
    df["skills"] = df["job_description"].apply(extract_skills)

    all_skills = []
    for skill_list in df["skills"]:
        all_skills.extend(skill_list)

    skill_counts = Counter(all_skills)

    skill_df = (
        pd.DataFrame(skill_counts.items(), columns=["Skill", "Count"])
        .sort_values(by="Count", ascending=False)
    )

    return skill_df

def plot_top_skills(skill_df, top_n=10):
    top = skill_df.head(top_n)

    plt.figure()
    plt.bar(top["Skill"], top["Count"])
    plt.xticks(rotation=45)
    plt.title("Top In-Demand Skills")
    plt.tight_layout()
    plt.show()

def create_wordcloud(df):
    text = " ".join(df["job_description"].astype(str))

    wc = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Job Description WordCloud")
    plt.show()

def main():
    df = load_data(DATA_FILE)
    df = clean_data(df)
    skill_df = analyze_skills(df)

    print("\nTop 10 In-Demand Skills:")
    print(skill_df.head(10))

    plot_top_skills(skill_df)
    create_wordcloud(df)

if __name__ == "__main__":
    main()

