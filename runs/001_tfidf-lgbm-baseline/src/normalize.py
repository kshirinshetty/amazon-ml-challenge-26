"""Text normalization + prep: raw TSVs -> work/{split}.parquet (and work/train_gt.parquet).

Usage: uv run python src/normalize.py
"""
import os
import re
from multiprocessing import Pool

import polars as pl
from unidecode import unidecode

DATA = "data/student_resource/dataset"
WORK = "work"

# Legal forms / filler words dropped from names (any country; France's forms included).
NAME_STOP = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "plc", "lp", "llp", "pc", "pllc", "sas", "sasu", "sarl", "sa", "eurl",
    "sci", "snc", "the", "and", "of", "dba",
}

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


def _base(s):
    return (s if s.isascii() else unidecode(s)).lower()


def norm_name(s):
    if not s:
        return ""
    s = _base(s)
    s = re.sub(r"\bwww\.|\.(com|net|org|in|co|fr|us|biz|info)\b|@", " ", s)
    s = re.sub(r"\d{6,}", " ", s)  # phone numbers glued onto names
    s = s.replace("&", " and ")
    s = re.sub(r"[.'`]", "", s)  # l.l.c. -> llc
    out = []
    for t in re.sub(r"[^a-z0-9]+", " ", s).split():
        c = re.sub(r"([a-z])\1+", r"\1", t)  # aa -> a: transliteration + typo noise
        if t not in NAME_STOP and c not in NAME_STOP:
            out.append(c)
    return " ".join(out)


def norm_addr(s):
    if not s or s in ("None", "N/A"):
        return ""
    s = _base(s)
    s = re.sub(r"\bn/a\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b0+(\d)", r"\1", s)  # 00806 -> 806
    s = _PHRASE_RE.sub(lambda m: ADDR_PHRASES[m.group(1)], s)
    return " ".join(ADDR_ABBR.get(t, t) for t in s.split() if t not in ("none", "na"))


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def prep(split):
    df = pl.concat([
        read_tsv(f"{DATA}/{split}/{split}_source{n}.tsv").with_columns(src=pl.lit(n, pl.Int8))
        for n in (1, 2, 3)
    ])
    with Pool(int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count())) as p:
        nm = p.map(norm_name, df["business_name"].to_list(), chunksize=20_000)
        ad = p.map(norm_addr, df["business_address"].to_list(), chunksize=20_000)
    df = df.with_columns(nm=pl.Series(nm), ad=pl.Series(ad))
    df.write_parquet(f"{WORK}/{split}.parquet")
    print(split, df.group_by("src", "country").len().sort("src", "country"))


if __name__ == "__main__":
    assert norm_name("SOLOVA FÁCT L.L.C.") == "solova fact"
    assert norm_name("internationalforteanimation.com") == "internationalforteanimation"
    assert norm_name("Fire Master (India) Pvt (Ltd) - 9832661323") == "fire master india"
    assert norm_name("राम मार्केटिंग प्राइवेट लिमिटेड") == "ram marketing praivet"
    assert norm_addr("##1824 Brittany Lane, N/A, Edmond, Oklahoma") == "1824 brittany ln edmond ok"
    assert norm_addr("00806-807 Trinity Orion, GJ, Surat") == "806 807 trinity orion gj surat"
    assert norm_addr("12 MG Road, Chennai, Tamil Nadu") == "12 mg rd chennai tn"
    assert norm_addr("None") == ""
    os.makedirs(WORK, exist_ok=True)
    for split in ("train", "test"):
        prep(split)
    gt = read_tsv(f"{DATA}/train/train_ground_truth.tsv")
    gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode(
        "matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename(
        {"source1_entity_id": "s1", "matched_entity_ids": "q"}).write_parquet(f"{WORK}/train_gt.parquet")
