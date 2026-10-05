"""Stage 2: normalization -> work/state/norm_<src>.csv (+ taxonomy coverage / plan)."""
import os, re, json, collections
import pandas as pd
from common import (canon_platform, plat_key, parse_year, parse_num, canon_esrb, strip_disambig, norm_title,
                    title_numbers)

W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
SRCS = ["dbpedia", "metacritic", "sales"]
BAD_TITLES = {"unchanged", ""}


def clean_company(v):
    v = str(v).strip()
    v = re.sub(r"\s*\((?:video game )?(?:company|developer|studio|publisher)\)\s*$", "", v, flags=re.I)
    return v


def company_key(v):
    k = norm_title(clean_company(v))
    k = re.sub(r"\b(inc|ltd|llc|co|corp|corporation|limited|gmbh|s a|sa|the)\b", " ", k)
    return " ".join(k.split())


cov = []
for src in SRCS:
    df = pd.read_csv(f"{W}/state/translated_{src}.csv", dtype=str, keep_default_na=False)
    # platform: per-source prior from unambiguous alias hits (tie-break only)
    vc = df["platform"].value_counts()
    prior = collections.Counter()
    for v, c in vc.items():
        cp, m = canon_platform(v)
        if m.startswith("alias"):
            prior[cp] += c
    pmap = {v: canon_platform(v, prior) for v in vc.index}
    out = pd.DataFrame({"id": df["id"], "source": src})
    out["name_raw"] = df["name"]
    nm = df["name"].map(lambda s: strip_disambig(s) if src == "dbpedia" else s.strip())
    nm = nm.where(~nm.str.lower().isin(BAD_TITLES), "")
    out["name"] = nm
    out["ntitle"] = nm.map(norm_title)
    out["dis_year"] = df["name"].map(lambda s: (re.findall(r"\((\d{4}) [^)]*\)\s*$", s) or [""])[0]) if src == "dbpedia" else ""
    out["tnums"] = out["ntitle"].map(lambda t: " ".join(title_numbers(t)))
    out["year"] = df["releaseYear"].map(parse_year)
    out["platform_raw"] = df["platform"]
    out["platform"] = df["platform"].map(lambda v: pmap[v][0] or "")
    out["plat_method"] = df["platform"].map(lambda v: pmap[v][1])
    # unmapped / distinct real platforms keep a source-supported cleaned label
    keep_raw = out["plat_method"].isin(["unmapped", "distinct"]) & (df["platform"].str.strip() != "")
    out["platform_label"] = out["platform"].where(~keep_raw, df["platform"].str.strip())
    out["pkey"] = out["platform"].where(~keep_raw, "raw:" + df["platform"].map(plat_key))
    out["developer"] = df["developer"].map(clean_company)
    out["dev_key"] = df["developer"].map(company_key)
    if "publisher" in df:
        out["publisher"] = df["publisher"].str.strip()
    else:
        out["publisher"] = ""
    out["genres_raw"] = df["genres"]
    out["series"] = df.get("series", pd.Series([""] * len(df))).fillna("")
    if "criticScore" in df:
        out["critic"] = df["criticScore"].map(lambda v: parse_num(v, 0, 100, integer=True))
        out["user"] = df["userScore"].map(lambda v: parse_num(v, 0, 10))
        es = df["ESRB"].map(canon_esrb)
        out["esrb"] = es.map(lambda x: x[0] or "")
        out["esrb_method"] = es.map(lambda x: x[1])
    else:
        out["critic"] = None; out["user"] = None; out["esrb"] = ""; out["esrb_method"] = "absent"
    out.to_csv(f"{W}/state/norm_{src}.csv", index=False)
    n = len(df)
    def rate(mask):
        return round(float(mask.mean()), 4)
    nn = df["platform"].str.strip() != ""
    cov.append(dict(source=src, attribute="platform", non_null=int(nn.sum()),
                    canonical=int((out["platform"] != "").sum()),
                    unmapped=int((nn & (out["platform"] == "")).sum()),
                    canonical_rate=round((out["platform"] != "").sum() / max(1, nn.sum()), 4)))
    if "ESRB" in df:
        nn = df["ESRB"].str.strip() != ""
        cov.append(dict(source=src, attribute="ESRB", non_null=int(nn.sum()), canonical=int((out["esrb"] != "").sum()),
                        unmapped=int((nn & (out["esrb"] == "")).sum()),
                        canonical_rate=round((out["esrb"] != "").sum() / max(1, nn.sum()), 4)))
    yy = df["releaseYear"].str.strip() != ""
    cov.append(dict(source=src, attribute="releaseYear(parse)", non_null=int(yy.sum()),
                    canonical=int(out["year"].notna().sum()), unmapped=int((yy & out["year"].isna()).sum()),
                    canonical_rate=round(out["year"].notna().sum() / max(1, yy.sum()), 4)))
    print(src, "platform methods", out["plat_method"].value_counts().to_dict())

pd.DataFrame(cov).to_csv(f"{W}/taxonomy_coverage.csv", index=False)
print(pd.DataFrame(cov))
json.dump({
    "platform": {"path": "task/input/schemamatching/Gaming_Platforms_Taxonomy.csv", "column": "Platform Name",
                 "exhaustive": False,
                 "policy": "alias table + sorted-token + prefix + keyboard/OCR-weighted fuzzy (common.py); "
                           "unmapped real platforms keep cleaned source label; uninformative strings -> unknown"},
    "ESRB": {"path": "task/input/schemamatching/ESRB_Rating_Taxonomy.csv", "column": "Rating Code",
             "exhaustive": True, "aliases": {"K-A": "E", "Teen": "T", "Everyone": "E", "E10 Plus": "E10+",
                                             "Mature 17+": "M"},
             "policy": "closed set; single-char keyboard-neighbour noise resolved only when unique; else null"},
    "genres": {"path": "task/input/schemamatching/Video_Game_Genres_Taxonomy.csv", "column": "Genre Name",
               "exhaustive": False, "serialization": "JSON list",
               "policy": "source genre strings repaired to frequent vocabulary; mapped to taxonomy top-level where clear; "
                         "union of both kept (non-exhaustive)"},
}, open(f"{W}/taxonomy_plan.json", "w"), indent=1)
