import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/workspace/data/uae/uae_iteration_5/after_cleaning_full.csv")

df['region'] = df['region'].str.replace('\n', '')

print(df.region.value_counts())

print(df.head())
train_df, test_df = train_test_split(
    df,
    test_size=0.2,  # 20% for test set, adjust as needed
    stratify=df['region'],
    random_state=42  # for reproducibility
)


train_df.to_csv("/workspace/data/uae/uae_iteration_5/after_cleaning_train.csv", index=False)
test_df.to_csv("/workspace/data/uae/uae_iteration_5/after_cleaning_test.csv", index=False)