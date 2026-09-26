"""Text normalization + prep: raw TSVs -> work/{split}.parquet (and work/train_gt.parquet).

Three layers, all country-agnostic:
  1. rules: alias stripping ("X formerly Y" -> "Y"), transliteration, legal forms, abbreviations;
  2. spelling maps learned from matched training pairs (praivet -> private, sixth -> 6th, ciy -> city);
  3. injected filler words, detected per split and country as tokens over-represented in S2/S3
     relative to S1 (catches France's "participations", "et fils" with no French training data).

Usage: uv run python src/normalize.py
"""
import json
import os
import re
from multiprocessing import Pool

import polars as pl
from unidecode import unidecode

DATA = "data/student_resource/dataset"
WORK = "work"
THREADS = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())
MAP_MIN, MAP_SHARE = 20, 0.5  # a learned replacement needs >= 20 pairs and >= 50% of the token's mismatches
NOISE_LIFT = {"nm": 3.0, "ad": 10.0}  # filler: over-represented this much in S2/S3 vs S1 (addresses stricter:
NOISE_MIN = 0.0003                    # city names reach ~3-5x, real filler 10-10000x), in >= 0.03% of S2/S3

# Legal forms / filler words dropped from names (any country; France's forms included).
NAME_STOP = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "plc", "lp", "llp", "pc", "pllc", "sas", "sasu", "sarl", "sa", "eurl",
    "sci", "snc", "the", "and", "of", "dba",
}
ALIAS = re.compile(r"^.*\b(?:formerly|also known as|known as|doing business as|trading as|fka|f/k/a|aka|a/k/a"
                   r"|dba|d/b/a)\b[\s:.\-]*", re.I)

ADDR_ABBR = {
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd", "drive": "dr",
    "lane": "ln", "boulevard": "blvd", "bd": "blvd", "highway": "hwy", "parkway": "pkwy",
    "court": "ct", "place": "pl", "suite": "ste", "apartment": "apt", "floor": "flr",
    "north": "n", "south": "s", "east": "e", "west": "w", "near": "nr", "opposite": "opp",
    "number": "no", "building": "bldg", "sector": "sec", "circle": "cir", "square": "sq",
    "terrace": "ter", "route": "rte", "r": "rue", "chemin": "ch", "impasse": "imp",
    # US states
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
    # Indian states (single-word ones; multi-word handled in ADDR_PHRASES)
    "karnataka": "ka", "maharashtra": "mh", "gujarat": "gj", "rajasthan": "rj",
    "telangana": "tg", "kerala": "kl", "punjab": "pb", "haryana": "hr", "bihar": "br",
    "odisha": "od", "orissa": "od", "assam": "as", "goa": "ga", "jharkhand": "jh",
    "uttarakhand": "uk", "chhattisgarh": "cg", "delhi": "dl",
}
ADDR_PHRASES = {
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "west virginia": "wv",
    "district of columbia": "dc", "tamil nadu": "tn", "andhra pradesh": "ap",
    "uttar pradesh": "up", "madhya pradesh": "mp", "west bengal": "wb",
    "himachal pradesh": "hp", "jammu and kashmir": "jk",
}
_PHRASE_RE = re.compile(r"\b(" + "|".join(ADDR_PHRASES) + r")\b")
_collapse = lambda t: re.sub(r"([a-z])\1+", r"\1", t)  # aa -> a: transliteration + typo noise
STOP = {_collapse(w) for w in NAME_STOP}

# Filled in by prep() before the worker pool forks: learned maps and per-country filler sets.
MAPS = {"name": {}, "addr": {}}
NOISE = {}


def _base(s):
    return (s if s.isascii() else unidecode(s)).lower()


def name_tokens(s):
    """Alias-strip, transliterate, drop domains/phones/punctuation, collapse repeats. No stopwords."""
    if not s:
        return ""
    m = ALIAS.match(s)
    if m and s[m.end():].strip():
        s = s[m.end():]
    s = _base(s)
    s = re.sub(r"\bwww\.|\.(com|net|org|in|co|fr|us|biz|info)\b|@", " ", s)
    s = re.sub(r"\d{6,}", " ", s)  # phone numbers glued onto names
    s = s.replace("&", " and ")
    s = re.sub(r"[.'`]", "", s)  # l.l.c. -> llc
    return " ".join(_collapse(t) for t in re.sub(r"[^a-z0-9]+", " ", s).split())


def addr_tokens(s):
    if not s or s in ("None", "N/A"):
        return ""
    s = _base(s)
    s = re.sub(r"\bn/a\b|\bndeg", " ", s)  # n/a; French "n°" (-> "ndeg" after unidecode)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b0+(\d)", r"\1", s)  # 00806 -> 806
    s = _PHRASE_RE.sub(lambda m: ADDR_PHRASES[m.group(1)], s)
    return " ".join(ADDR_ABBR.get(t, t) for t in s.split() if t not in ("none", "na"))


def finish(tokens, kind, country):
    """Apply the learned spelling map, then drop stopwords + filler (never leave a name empty)."""
    stop = STOP if kind == "name" else set()
    noise = NOISE.get(country, {}).get(kind, set())
    toks = [MAPS[kind].get(t, t) for t in tokens.split()]
    return " ".join([t for t in toks if t not in stop and t not in noise]
                    or [t for t in toks if t not in stop] or toks)


def finish_row(row):
    nm, ad, country = row
    return finish(nm, "name", country), finish(ad, "addr", country)


def learn_map(tq, ts):
    """Token replacements seen consistently in matched pairs: when a pair differs by 1-3 tokens on
    each side, count which S1 token stands in for each S2/S3 token. Only S2/S3-side spellings are
    remapped (S1 is the clean reference), so real words like "girl" or a city name never get rewritten."""
    toks = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    d = pl.DataFrame({"tq": tq, "ts": ts}).with_columns(toks("tq"), toks("ts"))
    count = lambda c: d.select(t=pl.col(c)).explode("t").group_by("t").len(c)
    s1_side = count("tq").join(count("ts"), on="t", how="left").fill_null(0).filter(pl.col("ts") > 0.2 * pl.col("tq"))
    d = (d
         .with_columns(uq=pl.col("tq").list.set_difference("ts"), us=pl.col("ts").list.set_difference("tq"))
         .filter(pl.col("uq").list.len().is_between(1, 3) & pl.col("us").list.len().is_between(1, 3)))
    n_t = d.select("uq").explode("uq").group_by("uq").len("n_t")
    best = (d.select("uq", "us").explode("uq").explode("us").group_by("uq", "us").len("n")
            .sort("n", descending=True).unique("uq", keep="first").join(n_t, on="uq")
            .filter((pl.col("n") >= MAP_MIN) & (pl.col("n") / pl.col("n_t") >= MAP_SHARE)
                    & ~pl.col("uq").str.contains(r"\d") & ~pl.col("uq").is_in(s1_side["t"].implode())
                    # skip lossy maps: run-together words keep their full letters ("oncologymedicine" already
                    # matches "oncology medicine" on the space-free key), single letters, bare function words
                    & ~((pl.col("uq").str.len_chars() >= 12) & pl.col("uq").str.contains(pl.col("us"), literal=True))
                    & (pl.col("us").str.len_chars() > 1) & ~pl.col("us").is_in(["the", "and", "of", "dba"])))
    return dict(best.select("uq", "us").iter_rows())


def filler(df, col):
    """Per-country tokens over-represented in S2/S3 vs S1 (injected noise), digits excluded."""
    n = df.group_by("country").agg(n1=(pl.col("src") == 1).sum(), n23=(pl.col("src") != 1).sum())
    t = (df.select("country", "src", t=pl.col(col).str.split(" ").list.unique()).explode("t")
         .filter((pl.col("t") != "") & ~pl.col("t").str.contains(r"^\d+$"))
         .group_by("country", "t").agg(s1=(pl.col("src") == 1).sum(), s23=(pl.col("src") != 1).sum())
         .join(n, on="country")
         .filter((pl.col("s23") / pl.col("n23") >= NOISE_LIFT[col] * (pl.col("s1") + 1) / pl.col("n1"))
                 & (pl.col("s23") >= NOISE_MIN * pl.col("n23"))))
    return {c: set(g["t"]) for (c,), g in t.group_by("country")}


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def prep(split):
    df = pl.concat([
        read_tsv(f"{DATA}/{split}/{split}_source{n}.tsv").with_columns(src=pl.lit(n, pl.Int8))
        for n in (1, 2, 3)
    ])
    with Pool(THREADS) as p:
        nm = p.map(name_tokens, df["business_name"].to_list(), chunksize=20_000)
        ad = p.map(addr_tokens, df["business_address"].to_list(), chunksize=20_000)
    df = df.with_columns(nm=pl.Series(nm), ad=pl.Series(ad))
    if split == "train":  # learn spelling maps from matched pairs; test reuses them
        gt = pl.read_parquet(f"{WORK}/train_gt.parquet")
        side = lambda k: df.select(**{k: "entity_id", f"nm_{k}": "nm", f"ad_{k}": "ad"})
        pairs = gt.join(side("q"), on="q").join(side("s1"), on="s1")
        MAPS.update(name=learn_map(pairs["nm_q"], pairs["nm_s1"]), addr=learn_map(pairs["ad_q"], pairs["ad_s1"]))
        json.dump(MAPS, open(f"{WORK}/maps.json", "w"), indent=0, sort_keys=True)
    else:
        MAPS.update(json.load(open(f"{WORK}/maps.json")))
    mapped = df.select("country", "src", nm=pl.col("nm").str.split(" ").list.eval(pl.element().replace(MAPS["name"])).list.join(" "),
                       ad=pl.col("ad").str.split(" ").list.eval(pl.element().replace(MAPS["addr"])).list.join(" "))
    names, addrs = filler(mapped, "nm"), filler(mapped, "ad")
    NOISE.clear()
    NOISE.update({c: {"name": names.get(c, set()) - {""}, "addr": addrs.get(c, set())} for c in df["country"].unique()})
    json.dump({c: {k: sorted(v) for k, v in d.items()} for c, d in NOISE.items()},
              open(f"{WORK}/{split}_noise.json", "w"), indent=0)
    with Pool(THREADS) as p:  # forked after MAPS/NOISE are set, so workers see them
        out = p.map(finish_row, zip(df["nm"], df["ad"], df["country"]), chunksize=20_000)
    df = df.with_columns(nm=pl.Series([a for a, _ in out]), ad=pl.Series([b for _, b in out]))
    df.write_parquet(f"{WORK}/{split}.parquet")
    print(split, {k: len(v) for k, v in MAPS.items()}, "maps;",
          {c: {k: len(v) for k, v in d.items()} for c, d in NOISE.items()}, "filler tokens")


if __name__ == "__main__":
    assert name_tokens("SOLOVA FÁCT L.L.C.") == "solova fact lc"
    assert finish(name_tokens("SOLOVA FÁCT L.L.C."), "name", "US") == "solova fact"
    assert name_tokens("Xylozeta Co formerly known as Cervantes Select Mountain LLC") == "cervantes select mountain lc"
    assert name_tokens("Viozeta formerly: New Delhi Communications") == "new delhi comunications"
    assert name_tokens("internationalforteanimation.com") == "internationalforteanimation"
    assert finish(name_tokens("Fire Master (India) Pvt (Ltd) - 9832661323"), "name", "India") == "fire master india"
    assert finish("lc", "name", "US") == "lc"  # a name made only of stopwords is kept as-is
    assert addr_tokens("##1824 Brittany Lane, N/A, Edmond, Oklahoma") == "1824 brittany ln edmond ok"
    assert addr_tokens("00806-807 Trinity Orion, GJ, Surat") == "806 807 trinity orion gj surat"
    assert addr_tokens("12 MG Road, Chennai, Tamil Nadu") == "12 mg rd chennai tn"
    assert addr_tokens("N°3 Rue Kant") == "3 rue kant"
    assert addr_tokens("None") == ""
    os.makedirs(WORK, exist_ok=True)
    gt = read_tsv(f"{DATA}/train/train_ground_truth.tsv")
    gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode(
        "matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename(
        {"source1_entity_id": "s1", "matched_entity_ids": "q"}).write_parquet(f"{WORK}/train_gt.parquet")
    for split in ("train", "test"):
        prep(split)
