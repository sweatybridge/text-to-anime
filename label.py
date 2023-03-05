from io import StringIO

import pandas as pd


def parse_data(fp):
    with open(fp, "r") as f:
        contents = f.read()
    parts = contents.split("\n\n")

    meta = {}
    for line in parts[0].split("\n"):
        key, value = line.split(":")
        meta[key] = value.strip()

    bbox = pd.read_csv(StringIO(parts[1]), sep="\s+")
    text = (
        pd.read_csv(StringIO(parts[2]), sep="\s+") if len(parts) > 2 else pd.DataFrame()
    )
    return meta, bbox, text
