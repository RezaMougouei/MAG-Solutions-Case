import pandas as pd
import glob

files = glob.glob('/workspace/data/**/*.parquet', recursive=True)

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

df.to_csv('/workspace/output.csv', index=False)
