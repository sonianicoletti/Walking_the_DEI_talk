import pandas as pd

FILE1 = "combined_csv_tweets.csv"
FILE2 = "combined_pickle_tweets.csv"
FINAL_OUTPUT = "tweets.csv"
ID_COLUMN = "id"

df1 = pd.read_csv(FILE1)
df2 = pd.read_csv(FILE2)

# Ensure ID column exists
if ID_COLUMN not in df1.columns or ID_COLUMN not in df2.columns:
    print(f"Error: '{ID_COLUMN}' column must exist in both files.")
    exit()

# Convert IDs to string (safety)
df1[ID_COLUMN] = df1[ID_COLUMN].astype(str)
df2[ID_COLUMN] = df2[ID_COLUMN].astype(str)
ids1 = set(df1[ID_COLUMN])
ids2 = set(df2[ID_COLUMN])

# Compare differences
only_1 = ids1 - ids2
only_2 = ids2 - ids1

print("="*50)
print(f"Tweets in {FILE1} but NOT in {FILE2}: {len(only_1)}")
print(f"Tweets in {FILE2} but NOT in {FILE1}: {len(only_2)}")
print("="*50)

# Combine both and deduplicate
combined = pd.concat([df1, df2], ignore_index=True)
before_dedup = len(combined)
combined = combined.drop_duplicates(subset=[ID_COLUMN])
after_dedup = len(combined)

# -------- Post-merge processing -------- #

# Drop unwanted columns if they exist
cols_to_drop = ["Unnamed: 0", "index", "Unnamed: 0.1", "level_0", "author_id", "username", "path"]
combined = combined.drop(columns=[c for c in cols_to_drop if c in combined.columns], errors="ignore")

# Convert year, month and day to integers
for col in ["year", "month", "day"]:
    if col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Int64")
        combined[col] = combined[col].fillna(0).astype(int)

# ---- Normalise Tesla / Apple naming with period ---- #

# For Tesla Inc.
tesla_rows = combined["Company"].str.lower().str.startswith("tesla inc").fillna(False)
unique_tesla_ceos = combined.loc[tesla_rows, "CEO"].dropna().unique()

if len(unique_tesla_ceos) > 1:
    raise ValueError(f"Error: 'Tesla Inc.' has multiple CEOs: {unique_tesla_ceos.tolist()}")
elif len(unique_tesla_ceos) == 1:
    tesla_ceo = unique_tesla_ceos[0]
else:
    raise ValueError("Error: No CEO found for 'Tesla Inc.' to back-fill.")

# Rename all Tesla Inc. rows to "Tesla Inc."
combined.loc[tesla_rows, "Company"] = "Tesla Inc."
combined.loc[combined["Company"] == "Tesla Inc.", "CEO"] = tesla_ceo

# For Apple Inc.
apple_rows = combined["Company"].str.lower().str.startswith("apple inc").fillna(False)
unique_apple_ceos = combined.loc[apple_rows, "CEO"].dropna().unique()

if len(unique_apple_ceos) > 1:
    raise ValueError(f"Error: 'Apple Inc.' has multiple CEOs: {unique_apple_ceos.tolist()}")
elif len(unique_apple_ceos) == 1:
    apple_ceo = unique_apple_ceos[0]
else:
    raise ValueError("Error: No CEO found for 'Apple Inc.' to back-fill.")

# Rename all Apple Inc. rows to "Apple Inc."
combined.loc[apple_rows, "Company"] = "Apple Inc."
combined.loc[combined["Company"] == "Apple Inc.", "CEO"] = apple_ceo

# Rename company "Nasdaq Inc" to ensure consistent spelling
combined["Company"] = combined["Company"].replace({"Nasdaq Inc": "Nasdaq Inc."})

# -------- Manual CEO fills -------- #

manual_ceos = {
    "NRG Energy": "Mauricio Gutierrez",
    "PTC": "James E. Heppelmann",
    "Qualcomm": "Cristiano Amon",
    "Philip Morris International": "Jacek Olczak",
    "PayPal": "Dan Schulman",
    "Pioneer Natural Resources": "Scott D. Sheffield",
    "AES Corporation": "Andres Gluski",
    "Nasdaq Inc.": "Adena Friedman",
    "Otis Worldwide": "Judy Marks",
    "PepsiCo": "Ramon Laguarta",
    "Pfizer": "Albert Bourla",
}

# Fill for listed companies
for company, ceo in manual_ceos.items():
    combined.loc[combined["Company"] == company, "CEO"] = ceo

# ---- Back-fill CEO by company, enforcing one CEO per company ---- #

CEO_COLUMN = "CEO"
COMPANY_COLUMN = "Company"

missing_before = combined[CEO_COLUMN].isna().sum()

# build CEO-per-company map
ceo_map = {}
for company, group in combined.groupby(COMPANY_COLUMN):
    unique_ceos = group[CEO_COLUMN].dropna().unique()
    if len(unique_ceos) > 1:
        # raise ValueError(f"Error: Company '{company}' has multiple CEOs: {unique_ceos.tolist()}")
        ceo_map[company] = "multiple CEOs"
    if len(unique_ceos) == 1:
        ceo_map[company] = unique_ceos[0]

# apply map to fill missing CEOs
combined[CEO_COLUMN] = combined.apply(
    lambda row: ceo_map.get(row[COMPANY_COLUMN], row[CEO_COLUMN])
    if pd.isna(row[CEO_COLUMN]) else row[CEO_COLUMN],
    axis=1
)

# Print how many were filled
filled_count = missing_before - combined[CEO_COLUMN].isna().sum()
missing_after = combined[CEO_COLUMN].isna().sum()
print(f"CEO values back-filled from Company groups: {filled_count}")
print(f"Rows still missing CEO after back-fill: {missing_after}")

# ------ Remove rows where 'text' or 'created_at' is NaN or empty string ------ #

total_before = len(combined)
combined = combined[combined["text"].notna() & (combined["text"].str.strip() != "")]
total_after = len(combined)
removed_rows = total_before - total_after
print(f"Number of rows removed because of empty text: {removed_rows}")

total_before2 = len(combined)
combined = combined[combined["created_at"].notna() & (combined["created_at"].str.strip() != "")]
total_after2 = len(combined)
removed_rows2 = total_before2 - total_after2
print(f"Number of rows removed because of empty created_at: {removed_rows2}")

# ------ Fill year, month, day from created_at ------ #

combined["created_at"] = pd.to_datetime(combined["created_at"], errors="coerce")
mask = combined["year"] == 0
combined.loc[mask, "year"] = combined.loc[mask, "created_at"].dt.year
combined.loc[mask, "month"] = combined.loc[mask, "created_at"].dt.month
combined.loc[mask, "day"] = combined.loc[mask, "created_at"].dt.day
filled_count = mask.sum()
print(f"Number of rows filled from 'created_at': {filled_count}")

# ------ Rename Company column to company ------ #

combined = combined.rename(columns={"Company": "company"})

# ------ Save updated combined CSV again ------ #
combined.to_csv(FINAL_OUTPUT, index=False)

# -------- Print summary -------- #

total_tweets = len(combined)
unique_companies = combined["company"].nunique(dropna=True)
unique_CEOs = combined["CEO"].nunique(dropna=True)

print("="*80)
print("="*80)
print(f"Total unique tweet rows: {total_tweets}")
print(f"Total unique companies: {unique_companies}")
print(f"Total unique CEOs: {unique_CEOs}")
print("="*80)

print(f"\nSaved merged dataset as: {FINAL_OUTPUT}")
