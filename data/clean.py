import pandas as pd
from utils import clean

df = pd.read_csv("data/Tweets.csv")[["text", "airline_sentiment"]]
print("Before:", df.shape, df.isna().sum().to_dict())

df = df.dropna().drop_duplicates()
df["text"] = df["text"].apply(clean)
df = df[df["text"].str.len() > 0]

df.to_csv("data/clean.csv", index=False)
print("Saved:", df.shape)
print(df["airline_sentiment"].value_counts())
