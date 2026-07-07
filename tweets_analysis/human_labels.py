import pandas as pd
import shutil

# Copy the original CSV
shutil.copy("dei_tweets_merged.csv", "dei_tweets_merged_labeled.csv")

# Load the copy
df = pd.read_csv("dei_tweets_merged_labeled.csv")

# Add the empty focus column
df["focus"] = None

# Save it
df.to_csv("dei_tweets_merged_labeled.csv", index=False)

print("Created dei_tweets_merged_labeled.csv with an empty 'focus' column.")

df = pd.read_csv("dei_tweets_merged_labeled.csv")

# Select 100 tweets with missing labels
tweets_to_label = df[df["focus"].isna()].head(100)

for idx, row in tweets_to_label.iterrows():
    print("\n-----------------------------")
    print(f"TWEET {idx}:")
    print(row["text"])
    print("-----------------------------")
    choice = input("Press 1 = Promotion, 0 = Prevention, s = skip: ").strip()

    if choice == "1":
        df.at[idx, "focus"] = "Promotion"
    elif choice == "0":
        df.at[idx, "focus"] = "Prevention"
    else:
        print("Skipped.")

# Save updates
df.to_csv("dei_tweets_merged_labeled.csv", index=False)

print("\nSaved labels to dei_tweets_merged_labeled.csv")
