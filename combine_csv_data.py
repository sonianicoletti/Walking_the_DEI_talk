# pip install pandas
import os
import pandas as pd
import re

DATASETS = [
    "data_csv/CEO_Data_400_500.csv",
    "data_csv/CEO_Dataa_1_98.csv"
]

COMPANIES_FOLDER = "data_csv/companies"
FINAL_OUTPUT = "combined_csv_tweets.csv"
ID_COLUMN = "id"

all_tweets = []

# Load stand-alone datasets
for file in DATASETS:
    if not os.path.isfile(file):
        print(f"Warning: {file} not found, skipping...")
        continue

    df = pd.read_csv(file)
    all_tweets.append(df)

# Traverse companies folder
if os.path.isdir(COMPANIES_FOLDER):
    for company in os.listdir(COMPANIES_FOLDER):
        company_path = os.path.join(COMPANIES_FOLDER, company)

        if not os.path.isdir(company_path):
            continue  # skip files

        # read every CSV inside each company folder
        for file in os.listdir(company_path):
            if not file.lower().endswith(".csv"):
                continue

            csv_path = os.path.join(company_path, file)
            df = pd.read_csv(csv_path)

            # Add missing fields
            df["Company"] = company
            if "CEO" not in df.columns:
                df["CEO"] = pd.NA

            all_tweets.append(df)
else:
    print(f"Warning: {COMPANIES_FOLDER} folder not found.")

# Combine all tweets into one DataFrame
if not all_tweets:
    print("No valid datasets found. Exiting.")
    exit()

combined = pd.concat(all_tweets, ignore_index=True)

# Deduplicate by tweet ID
if ID_COLUMN not in combined.columns:
    print(f"Error: ID column '{ID_COLUMN}' not found in data_csv. Check dataset structure.")
    exit()

before_dedup = len(combined)
combined = combined.drop_duplicates(subset=[ID_COLUMN])
after_dedup = len(combined)

# Replace HTML entity
combined["text"] = combined["text"].str.replace("&amp;", "&", regex=False)

# Extract URL at the end
url_pattern = re.compile(r"(https?:.*?t\.co.\S+)$", re.MULTILINE)
def extract_end_of_tweet_url(text):
    if pd.isna(text):
        return "", text
    match = url_pattern.search(text)
    if match:
        return match.group(1), url_pattern.sub("", text).strip()
    return "", text
combined["end_of_tweet_url"], combined["text"] = zip(*combined["text"].map(extract_end_of_tweet_url))

# Save final CSV
combined.to_csv(FINAL_OUTPUT, index=False)

# Compute stats
total_tweets = after_dedup
unique_companies = combined["Company"].nunique(dropna=True)
unique_CEOs = combined["CEO"].nunique(dropna=True)

print(f"Combined CSV saved as: {FINAL_OUTPUT}")
print(f"Duplicate rows removed: {before_dedup - after_dedup}")
print(f"Total tweets: {total_tweets}")
print(f"Unique companies: {unique_companies}")
print(f"Unique CEOs: {unique_CEOs}")
