import pandas as pd

metadata_dir = "pf-bula-castillo-garcia/annotationsv2.csv"
annotations = pd.read_csv(metadata_dir)

print(annotations.head())