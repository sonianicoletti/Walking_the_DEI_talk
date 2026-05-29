import os
import pandas as pd
import pickle
import re

ROOT = "data_pickle"
COMPANIES_BATCH_DIR = os.path.join(ROOT, "companies")
OTHER_TABLE_DIR = os.path.join(ROOT, "other")
FINAL_OUTPUT = "combined_pickle_tweets.csv"

tweets = []

# -------- Helper functions -------- #

def extract_from_batch_dict(obj, company=None, CEO=None, path=None):
    user_map = {}
    if isinstance(obj, dict) and "includes" in obj and "users" in obj["includes"]:
        for u in obj["includes"]["users"]:
            user_map[u.get("id")] = u.get("username")

    # Determine the data key
    data_list = None
    if isinstance(obj, dict):
        if "data_csv" in obj:
            data_list = obj["data_csv"]
        elif "data" in obj:
            data_list = obj["data"]
        else:
            # Handle single-tweet dict
            data_list = [obj]

    if data_list and isinstance(data_list, list):
        for row in data_list:
            t_id = row.get("id") or row.get("id_str")
            if not t_id:
                continue
            username = user_map.get(row.get("author_id"))

            text = row.get("text", "")
            text = text.replace("&amp;", "&")
            url_match = re.search(r"(https?:.*?t\.co.\S+)$", text)
            end_of_tweet_url = url_match.group(1) if url_match else ""
            if end_of_tweet_url:
                text = text[: -len(end_of_tweet_url)].strip()

            tweets.append({
                "id": t_id,
                "text": text,
                "created_at": row.get("created_at"),
                "author_id": row.get("author_id"),
                "username": username,
                "Company": company if company else "",
                "CEO": CEO if CEO else "",
                "path": path,
                "end_of_tweet_url": end_of_tweet_url
            })

def extract_from_table_df(df, path, company=None, CEO=None):
    if not isinstance(df, pd.DataFrame):
        return

    for _, row in df.iterrows():
        if "id" not in row:
            continue

        text = row.get("text", "")
        text = text.replace("&amp;", "&")
        url_match = re.search(r"(https?:.*?t\.co.\S+)$", text)
        end_of_tweet_url = url_match.group(1) if url_match else ""
        if end_of_tweet_url:
            text = text[: -len(end_of_tweet_url)].strip()

        tweets.append({
            "id": str(row["id"]),
            "text": text,
            "created_at": row.get("created_at"),
            "year": int(row.get("year")) if pd.notna(row.get("year")) else None,
            "month": int(row.get("month")) if pd.notna(row.get("month")) else None,
            "day": int(row.get("day")) if pd.notna(row.get("day")) else None,
            "Company": company if company else "",
            "CEO": CEO if CEO else "",
            "author_id": row.get("author_id") if "author_id" in row else None,
            "username": row.get("username") if "username" in row else None,
            "path": path,
            "end_of_tweet_url": end_of_tweet_url
        })

# -------- 1. Companies batch subfolders -------- #

for company in os.listdir(COMPANIES_BATCH_DIR):
    company_path = os.path.join(COMPANIES_BATCH_DIR, company)
    if not os.path.isdir(company_path):
        continue

    for file in os.listdir(company_path):
        if not file.lower().endswith((".pickle", ".pkl")):
            continue
        pickle_path = os.path.join(company_path, file)
        try:
            with open(pickle_path, "rb") as f:
                obj = pickle.load(f)
            extract_from_batch_dict(obj, path=pickle_path, company=company, CEO=None)
        except Exception:
            try:
                df = pd.read_pickle(pickle_path)
                extract_from_table_df(df, path=pickle_path, company=company, CEO=None)
            except Exception as e:
                print(f"Skipping (unreadable): {pickle_path} → {e}")

# -------- 2. Miscellaneous pickles in root -------- #

for file in os.listdir(ROOT):
    file_path = os.path.join(ROOT, file)
    if os.path.isfile(file_path) and file.lower().endswith((".pickle", ".pkl")):
        try:
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
            extract_from_batch_dict(obj, path=file_path)
        except Exception:
            try:
                df = pd.read_pickle(file_path)
                extract_from_table_df(df, path=file_path)
            except Exception:
                pass


# -------- 3. Other folder containing pandas DataFrame tables -------- #

for subdir in os.listdir(OTHER_TABLE_DIR):
    subdir_path = os.path.join(OTHER_TABLE_DIR, subdir)
    if not os.path.isdir(subdir_path):
        continue

    for file in os.listdir(subdir_path):
        if not file.lower().endswith((".pickle", ".pkl")):
            continue
        pickle_path = os.path.join(subdir_path, file)
        try:
            df = pd.read_pickle(pickle_path)
            extract_from_table_df(df, path=pickle_path, company=subdir, CEO=subdir)
        except Exception as e:
            print(f"Skipping table pickle: {pickle_path} → {e}")

# -------- 4. Save combined CSV after deduplication -------- #

if not tweets:
    print("No tweets extracted. Exiting.")
    exit()

combined = pd.DataFrame(tweets)

# Deduplicate by tweet id
combined = combined.drop_duplicates(subset=["id"])

# Remove tweets from Twitter
combined = combined[combined["username"].fillna("") != "Twitter"].copy()

# Ensure all columns are present
expected_cols = ["id", "text", "created_at", "year", "month", "day",
                 "Company", "CEO", "author_id", "username", "path", "end_of_tweet_url"]
for col in expected_cols:
    if col not in combined.columns:
        combined[col] = pd.NA

combined = combined[expected_cols]

combined.to_csv(FINAL_OUTPUT, index=False)
print(f"Saved: {FINAL_OUTPUT}")
print(f"Total unique tweets: {len(combined)}")

unique_companies = combined["Company"].nunique(dropna=True)
unique_CEOs = combined["CEO"].nunique(dropna=True)

print(f"Total unique companies: {unique_companies}")
print(f"Total unique CEOs: {unique_CEOs}")
