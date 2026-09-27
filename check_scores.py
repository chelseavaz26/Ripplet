import pandas as pd
df = pd.read_parquet('pipeline/data.parquet')
print(df[['sentiment_score','emotion_score','irony_score']].isna().sum())
print(df[['sentiment_score','emotion_score','irony_score']].describe())
