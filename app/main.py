import pandas as pd
import os

folder = "matches_dataset"

files = [
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if f.endswith(".parquet")
]

df = pd.concat([pd.read_parquet(f) for f in files])

print(df.head())

print("Rows:", len(df))

df.to_parquet("full_dataset.parquet", index=False)

print("DONE")