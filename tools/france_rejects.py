"""Show France best candidates the model rejected although the house numbers agree (likely missed matches)."""
import polars as pl

W = "work"
rec = pl.read_parquet(f"{W}/test.parquet", columns=["entity_id", "src", "country", "business_name", "business_address", "nm", "ad"]).with_columns(
    num1=pl.col("ad").str.extract(r"\b(\d+)\b"))
p = pl.read_parquet(f"{W}/test_probs.parquet")
best = p.sort("p", descending=True).unique("q", keep="first")
q = rec.filter(pl.col("src") != 1).select(q="entity_id", country="country", q_name="business_name", q_addr="business_address", q_nm="nm", num_q="num1")
s = rec.filter(pl.col("src") == 1).select(s1="entity_id", s_name="business_name", s_addr="business_address", s_nm="nm", num_s="num1")
b = best.join(q, on="q").join(s, on="s1").filter((pl.col("p") < 0.7) & (pl.col("num_q") == pl.col("num_s")))
print(b.group_by("country").agg(n=pl.len(), p_median=pl.col("p").median()).sort("country"))
print(b.filter(pl.col("country") == "France").select(pl.col("p").qcut([0.1, 0.25, 0.5, 0.75, 0.9], include_breaks=True)
      .struct.field("breakpoint")).unique().sort("breakpoint"))
for country in ["France", "US"]:
    print(f"\n=== {country}: rejected best candidates with agreeing house number")
    for r in b.filter(pl.col("country") == country).sample(18, seed=3).iter_rows(named=True):
        print(f"  p={r['p']:.2f}\n    Q  {r['q_name']} | {r['q_addr']}   [{r['q_nm']}]\n    S1 {r['s_name']} | {r['s_addr']}   [{r['s_nm']}]")
