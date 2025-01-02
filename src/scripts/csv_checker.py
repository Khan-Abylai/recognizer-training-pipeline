import pandas as pd

df = pd.read_csv("/mnt/ssd/data/uae/train.csv")

df["region"] = df["region"].replace("ras-al-khaiman", "ras-al-khaimah", regex=True)
df.to_csv("/mnt/ssd/data/uae/train.csv", index_label=False, index=False)