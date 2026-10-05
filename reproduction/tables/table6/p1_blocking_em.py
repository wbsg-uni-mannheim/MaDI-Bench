#!/usr/bin/env python
"""Table 6, P1 columns: pair completeness (PC), reduction ratio (RR) and matching F1 of the human-written
pipelines, per source pair, on the shipped entity-matching test split of each base task.

Table protocol: PC, RR and F1 are unweighted means over a task's source pairs; F1 counts a false positive
only for a predicted pair that is a labeled negative of the test split (the benchmark convention).

Input: a P1 run folder as written by reproduction/p1/run_p1.sh, <run_root>/<domain>/ with run.log (the
notebook's printed output), base/output/ (the files the notebook writes) and dump/. Where a notebook evaluates
on the test split itself, its logged counts are used; where it evaluates on another split, the persisted
candidate and correspondence files are re-scored on the test split after reproducing the notebook's own
logged numbers from the same files:

  games      notebook counts on *_test.csv (blocking and the rule-based matcher's correspondences);
  companies  the notebook logs validation counts only; the candidates of forbes-dbpedia are regenerated with
             the notebook's TokenBlocker (the file on disk holds the last evaluated pair only), the links are
             the greedy 1:1 links P1 fuses (forbes-fullcontact from matching_detailed_results.csv,
             forbes-dbpedia from the fused table's _fusion_sources); all re-scored on the test split;
  music      blocking candidates re-scored on the test split; matching = the raw rule-based correspondences;
  papers     notebook counts on *_test.csv (blocking summaries; final greedy 1:1 links);
  products   the notebook evaluates on its own gold; candidates and the greedy 1:1 links P1 fuses
             (debug_results_entity_matching/p1_p*_greedy) are re-scored on the shipped test split.

Run from the repository root:

    python reproduction/tables/table6/p1_blocking_em.py --p1-run /path/to/p1_runs --out p1_blocking_em.csv
    python reproduction/tables/table6/p1_blocking_em.py --p1-run /path/to/p1_runs \
        --domain-run products=/path/to/p1_products_run --out p1_blocking_em.csv

The output has the P1 rows of results/paper_tables/table6/blocking_em_per_pair.csv (pipeline, domain,
source_pair, metric, numerator, denominator, value, split) and the per-task means are printed. Counts only;
no API calls.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path

for _k in list(os.environ):
    if _k.endswith("API_KEY"):
        os.environ.pop(_k)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "use cases"
DOMAINS = ("games", "companies", "music", "products", "papers")
_TRUE = {"true", "1", "1.0"}
_FALSE = {"false", "0", "0.0"}


# ----------------------------------------------------------------------------- gold helpers

def _canon(a, b) -> tuple[str, str]:
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


def load_em_pairs(path: Path) -> pd.DataFrame:
    """Any EM pair file -> id1, id2, label (bool). A file has a header iff the last field of its first row is
    not a label; headerless rows are anchored on both ends (ids may contain commas)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not (len(r) == 1 and not r[0].strip())]
    header = rows[0][-1].strip().lower() not in _TRUE | _FALSE
    out = []
    if header:
        names = [c.strip().lower() for c in rows[0]]
        lab = names.index("label")
        ids = [i for i in range(len(names)) if i != lab][:2]
        out = [(r[ids[0]], r[ids[1]], r[lab]) for r in rows[1:]]
    else:
        out = [(r[0], ",".join(r[1:-1]), r[-1]) for r in rows]
    frame = pd.DataFrame(out, columns=["id1", "id2", "label"], dtype=str)
    frame["label"] = frame["label"].map(lambda x: x.strip().lower() in _TRUE)
    return frame


def test_gold(domain: str) -> dict[str, pd.DataFrame]:
    """The shipped test split per source pair (base task: the plain *_test.csv files)."""
    em_dir = TASKS / domain / "base" / "input" / "entitymatching"
    out = {}
    for p in sorted(em_dir.glob("*_test.csv")):
        if p.stem == "test_gt" or "_baseline_pruned" in p.stem or "_corner_filled" in p.stem:
            continue
        out[p.stem.removesuffix("_test")] = load_em_pairs(p)
    return out


def load_split_csv(path: Path) -> pd.DataFrame:
    """An EM split read the way the notebooks read it (header row tolerated)."""
    df = pd.read_csv(path, header=None, dtype=str)
    if str(df.iloc[0, 0]).strip().lower() in ("id1", "id_dblp", "id_left"):
        df = df.iloc[1:]
    df = df.iloc[:, :3]
    df.columns = ["id1", "id2", "label"]
    df["label"] = df["label"].map(lambda x: str(x).strip().lower() in _TRUE)
    return df.reset_index(drop=True)


def gold_sets(df: pd.DataFrame):
    pos = {_canon(a, b) for a, b, lab in zip(df.id1, df.id2, df.label) if lab}
    neg = {_canon(a, b) for a, b, lab in zip(df.id1, df.id2, df.label) if not lab}
    return pos, neg


def pc_counts(cands: pd.DataFrame, gdf: pd.DataFrame) -> tuple[int, int]:
    pos, _ = gold_sets(gdf)
    cset = {_canon(a, b) for a, b in zip(cands.id1.astype(str), cands.id2.astype(str))}
    return len(pos & cset), len(pos)


def em_counts(corr: pd.DataFrame, gdf: pd.DataFrame) -> tuple[int, int, int]:
    """TP, FP, FN; a false positive only on a labeled negative pair."""
    pos, neg = gold_sets(gdf)
    pred = {_canon(a, b) for a, b in zip(corr.id1.astype(str), corr.id2.astype(str))}
    return len(pred & pos), len(pred & neg), len(pos - pred)


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


LOG = {
    "found": re.compile(r"True Matches Found: (\d+)/(\d+)"),
    "rr": re.compile(r"Matching (\d+) x (\d+) elements .*?; (\d+) blocked pairs \(reduction ratio: ([0-9.]+)\)"),
    "tp": re.compile(r"True Positives:\s+(\d+)"),
    "fp": re.compile(r"False Positives:\s+(\d+)"),
    "fn": re.compile(r"False Negatives:\s+(\d+)"),
}


def parse_log(run: Path) -> dict:
    text = (run / "run.log").read_text(encoding="utf-8", errors="replace")
    return {"found": [(int(a), int(b)) for a, b in LOG["found"].findall(text)],
            "rr": [(int(a), int(b), int(c), float(d)) for a, b, c, d in LOG["rr"].findall(text)],
            "cms": list(zip(*(map(int, LOG[k].findall(text)) for k in ("tp", "fp", "fn"))))}


# ----------------------------------------------------------------------------- per domain

def p1_games(run: Path) -> dict:
    log, g = parse_log(run), test_gold("games")
    (f_m, t_m), (f_s, t_s) = log["found"][:2]
    assert (t_m, t_s) == (int(g["dbpedia_2_metacritic"].label.sum()), int(g["dbpedia_2_sales"].label.sum()))
    raw_m, _mbm_m, raw_s, _mbm_s = log["cms"][:4]
    cands = pd.read_csv(run / "base/output/blocking-evaluation/blocking_detailed_results.csv",
                        usecols=["id1", "id2"], dtype=str)
    assert pc_counts(cands, g["dbpedia_2_sales"]) == (f_s, t_s), "last evaluated pair (dbpedia-sales) differs"
    return {"pc": {"dbpedia_2_metacritic": (f_m, t_m), "dbpedia_2_sales": (f_s, t_s)},
            "rr": {"dbpedia_2_metacritic": log["rr"][0][3], "dbpedia_2_sales": log["rr"][1][3]},
            "em": {"dbpedia_2_metacritic": raw_m, "dbpedia_2_sales": raw_s}}


def p1_companies(run: Path, scratch: Path) -> dict:
    from PyDI.entitymatching import TokenBlocker

    log, g = parse_log(run), test_gold("companies")
    base = run / "base"
    em_dir = TASKS / "companies/base/input/entitymatching"
    val = {"forbes_2_dbpedia": load_split_csv(em_dir / "forbes_2_dbpedia_val.csv"),
           "forbes_2_fullcontact": load_split_csv(em_dir / "forbes_2_fullcontact_val.csv")}
    (fv_d, tv_d), (fv_fc, tv_fc) = log["found"][:2]
    (_, _, c_d, rr_d), (_, _, c_fc, rr_fc) = log["rr"][:2]
    _raw_d, greedy_d, _raw_fc, greedy_fc = log["cms"][:4]
    cand_fc = pd.read_csv(base / "output/blocking-evaluation/blocking_detailed_results.csv",
                          usecols=["id1", "id2"], dtype=str)
    assert len(cand_fc) == c_fc and pc_counts(cand_fc, val["forbes_2_fullcontact"]) == (fv_fc, tv_fc)
    # the forbes-dbpedia candidates were overwritten on disk: regenerate the notebook's TokenBlocker
    forbes = pd.read_csv(base / "output/normalization/forbes.csv", dtype={"id": str}, keep_default_na=False,
                         na_values=[""])
    dbpedia = pd.read_csv(base / "output/normalization/dbpedia.csv", dtype={"id": str}, keep_default_na=False,
                          na_values=[""])
    blk = TokenBlocker(forbes, dbpedia, column="name", batch_size=1000,
                       output_dir=str(scratch / "companies_f2d_tokenblocker"), id_column="id")
    cand_d = blk.materialize()[["id1", "id2"]].astype(str).drop_duplicates()
    assert len(cand_d) == c_d and pc_counts(cand_d, val["forbes_2_dbpedia"]) == (fv_d, tv_d), \
        "regenerated forbes-dbpedia candidates do not reproduce the logged count / validation PC"
    corr_fc = pd.read_csv(base / "output/debug_results_entity_matching/matching_detailed_results.csv",
                          usecols=["id1", "id2", "score"], dtype={"id1": str, "id2": str})
    assert em_counts(corr_fc, val["forbes_2_fullcontact"]) == greedy_fc
    fused = pd.read_csv(base / "output/data_fusion/fused.csv", usecols=["_fusion_sources"], dtype=str)
    pairs = []
    for s in fused["_fusion_sources"]:
        ids = ast.literal_eval(s)
        fb = [i for i in ids if i.startswith("http://www.forbes.com/")]
        db = [i for i in ids if i.startswith("http://dbpedia.org/")]
        if fb and db:
            pairs.append((fb[0], db[0]))
    corr_d = pd.DataFrame(pairs, columns=["id1", "id2"]).drop_duplicates()
    assert em_counts(corr_d, val["forbes_2_dbpedia"]) == greedy_d
    return {"pc": {"forbes_2_dbpedia": pc_counts(cand_d, g["forbes_2_dbpedia"]),
                   "forbes_2_fullcontact": pc_counts(cand_fc, g["forbes_2_fullcontact"])},
            "rr": {"forbes_2_dbpedia": rr_d, "forbes_2_fullcontact": rr_fc},
            "em": {"forbes_2_dbpedia": em_counts(corr_d, g["forbes_2_dbpedia"]),
                   "forbes_2_fullcontact": em_counts(corr_fc, g["forbes_2_fullcontact"])}}


def p1_music(run: Path) -> dict:
    base, g = run / "base/output", test_gold("music")
    out = {"pc": {}, "rr": {}, "em": {}}
    for short, pair in (("m2d", "musicbrainz_2_discogs"), ("m2l", "musicbrainz_2_lastfm")):
        summ = json.loads((base / f"blocking-evaluation/{short}/blocking_evaluation_summary.json").read_text())
        cands = pd.read_csv(base / f"blocking-evaluation/{short}/blocking_detailed_results.csv",
                            usecols=["id1", "id2"], dtype=str)
        assert len(cands) == summ["total_candidates"]
        out["pc"][pair] = pc_counts(cands, g[pair])
        out["rr"][pair] = summ["reduction_ratio"]
        d = base / f"debug_results_entity_matching/{short}_raw"
        corr = pd.read_csv(d / "matching_detailed_results.csv", usecols=["id1", "id2"], dtype=str)
        es = json.loads((d / "matching_evaluation_summary.json").read_text())
        c = em_counts(corr, g[pair])
        assert c == (es["true_positives"], es["false_positives"], es["false_negatives"])
        out["em"][pair] = c
    return out


def p1_papers(run: Path) -> dict:
    base, g = run / "base/output/entity_matching/blocking", test_gold("papers")
    out = {"pc": {}, "rr": {}, "em": {}}
    for pair in ("dblp_crossref", "dblp_openalex"):
        s = json.loads((base / pair / "blocking_evaluation_summary.json").read_text())
        assert s["total_true_pairs"] == int(g[pair].label.sum())
        out["pc"][pair] = (s["true_positives_found"], s["total_true_pairs"])
        out["rr"][pair] = s["reduction_ratio"]
    _val_c, _val_o, test_c, test_o = parse_log(run)["cms"][:4]      # final greedy 1:1 links on the test split
    out["em"] = {"dblp_crossref": test_c, "dblp_openalex": test_o}
    return out


def p1_products(run: Path) -> dict:
    base, g = run / "base/output", test_gold("products")
    out = {"pc": {}, "rr": {}, "em": {}}
    for short, pair, bdir in (("p1_p2", "prod1_to_prod2", "blocking_eval_prod1_prod2"),
                              ("p1_p3", "prod1_to_prod3", "blocking_eval_prod1_prod3"),
                              ("p1_p4", "prod1_to_prod4", "blocking_eval_prod1_prod4")):
        s = json.loads((base / "Blocking" / bdir / "blocking_evaluation_summary.json").read_text())
        cands = pd.read_csv(base / "Blocking" / bdir / "blocking_detailed_results.csv", usecols=["id1", "id2"],
                            dtype=str)
        assert len(cands) == s["total_candidates"]
        out["pc"][pair] = pc_counts(cands, g[pair])
        out["rr"][pair] = s["reduction_ratio"]
        corr = pd.read_csv(base / f"debug_results_entity_matching/{short}_greedy/matching_detailed_results.csv",
                           usecols=["id1", "id2"], dtype=str)
        out["em"][pair] = em_counts(corr, g[pair])
    return out


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--p1-run", type=Path, required=True, help="run root with <domain>/ folders (run_p1.sh)")
    ap.add_argument("--domain-run", action="append", default=[], metavar="DOMAIN=ROOT",
                    help="a different run root for one domain (repeatable)")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--out", type=Path, required=True, help="output CSV")
    a = ap.parse_args(argv)
    roots = {d: a.p1_run for d in DOMAINS}
    roots.update({k: Path(v) for k, v in (s.split("=", 1) for s in a.domain_run)})
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for d in [x for x in a.domains.split(",") if x]:
            run = roots[d] / d
            res = {"games": p1_games, "music": p1_music, "papers": p1_papers, "products": p1_products}[d](run) \
                if d != "companies" else p1_companies(run, Path(scratch))
            for pair, (found, pos) in res["pc"].items():
                rows.append(["P1", d, pair, "pc", found, pos, found / pos, "test"])
            for pair, rr in res["rr"].items():
                rows.append(["P1", d, pair, "rr", "", "", rr, ""])
            for pair, (tp, fp, fn) in res["em"].items():
                rows.append(["P1", d, pair, "f1", f"tp={tp} fp={fp} fn={fn}", "", f1(tp, fp, fn), "test"])
            pc = sum(f / p for f, p in res["pc"].values()) / len(res["pc"])
            rr = sum(res["rr"].values()) / len(res["rr"])
            fm = sum(f1(*c) for c in res["em"].values()) / len(res["em"])
            print(f"{d}: PC {100 * pc:.2f}  RR {100 * rr:.2f}  F1 {100 * fm:.2f}", flush=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pipeline", "domain", "source_pair", "metric", "numerator", "denominator", "value", "split"])
        w.writerows(rows)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
