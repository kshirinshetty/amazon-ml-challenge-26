"""For records whose candidates tie on the normalized name, does raw-name similarity pick the true S1?"""
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

W = "work"
f = pl.read_parquet(f"{W}/train_feats.parquet", columns=["q", "s1", "name_eq", "len_aq", "label"])
rec = pl.read_parquet(f"{W}/train.parquet", columns=["entity_id", "business_name", "business_address"])
low = lambda c: pl.col(c).str.to_lowercase()
d = (f.filter(pl.col("name_eq") == 1)
     .join(rec.select(q="entity_id", qn=low("business_name"), qa=low("business_address")), on="q")
     .join(rec.select(s1="entity_id", sn=low("business_name"), sa=low("business_address")), on="s1"))
d = d.filter(pl.len().over("q") >= 2).filter(pl.col("label").sum().over("q") == 1)  # ties with exactly one true S1
d = d.with_columns(raw=cpdist(d["qn"].to_list(), d["sn"].to_list(), scorer=fuzz.ratio, workers=-1),
                   raw_addr=cpdist(d["qa"].fill_null("").to_list(), d["sa"].to_list(), scorer=fuzz.ratio, workers=-1))
for seg, df in (("no address", d.filter(pl.col("len_aq") == 0)), ("with address", d.filter(pl.col("len_aq") > 0))):
    k = df.group_by("q").len()["len"]
    top = df.sort("raw", descending=True).group_by("q", maintain_order=True).first()
    strict = df.with_columns(mx=pl.col("raw").max().over("q")).filter(pl.col("raw") == pl.col("mx")).group_by("q").agg(
        n_max=pl.len(), hit=pl.col("label").max())
    print(f"{seg}: {df['q'].n_unique()} tied records, mean candidates {k.mean():.2f}, chance {(1 / k).mean():.3f}")
    print(f"   raw-name best == true S1: {top['label'].mean():.3f};  unique best & true: "
          f"{strict.filter(pl.col('n_max') == 1)['hit'].mean():.3f} on {strict.filter(pl.col('n_max') == 1).height} records")
    if seg == "with address":
        topa = df.sort("raw_addr", descending=True).group_by("q", maintain_order=True).first()
        print(f"   raw-address best == true S1: {topa['label'].mean():.3f}")
