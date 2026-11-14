import pandas as pd
import gzip

df = pd.read_pickle("results/reframe/analysis_cache.pkl.gz")
print(f"Loaded: {type(df)}, shape: {df.shape if hasattr(df, 'shape') else len(df)}")
print(f"Columns: {list(df.columns) if hasattr(df, 'columns') else 'N/A'}")

out_path = "results/reframe/analysis_cache.jsonl.gz"
if isinstance(df, pd.DataFrame):
    df.to_json(out_path, orient="records", lines=True)
else:
    import json
    with gzip.open(out_path, "wt") as f:
        for record in df:
            f.write(json.dumps(record) + "\n")

print(f"Written to {out_path}")
