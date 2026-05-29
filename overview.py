import pandas as pd

FILE = "tweets.csv"
df = pd.read_csv(FILE)

print("="*80)
print("TWEETS DATASET OVERVIEW")
print("="*80 + "\n")

# 1. Total number of tweets
total_tweets = len(df)
print(f"Total number of tweets: {total_tweets}\n")

# 2. Unique CEOs and their company
ceo_company_table = df[["CEO", "company"]].drop_duplicates().sort_values("company").reset_index(drop=True)
print("="*80)
print("CEO ↔ Company Table")
print("="*80)
print(ceo_company_table.to_string(index=False))

# 3. Tweets per CEO with first and last year of tweets
ceo_summary = df.groupby("CEO").agg(
    total_tweets=("id", "count"),
    first_year=("year", "min"),
    last_year=("year", "max")
).reset_index().sort_values("total_tweets", ascending=False)

print("="*80)
print("CEO Tweet Summary Table")
print("="*80)
print(ceo_summary.to_string(index=False))