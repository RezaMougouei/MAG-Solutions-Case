import pandas as pd
import glob

files = glob.glob('/workspace/data/**/*.parquet', recursive=True)

first = True
for f in files:
    df = pd.read_parquet(f)
    df.to_csv('/workspace/output.csv', mode='a', index=False, header=first)
    first = False
