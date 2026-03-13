import duckdb
import glob
import os

input_folder = "/workspace/data"
output_folder = "/workspace/csv"

os.makedirs(output_folder, exist_ok=True)

files = glob.glob(f"{input_folder}/**/*.parquet", recursive=True)

con = duckdb.connect()

for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    out = f"{output_folder}/{name}.csv"

    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{f}')
        ) TO '{out}' (HEADER, DELIMITER ',');
    """)

    print(f"Converted {f} -> {out}")
