"""The evaluation package against the shipped task files.

Pinned facts: the graded (kept) gold cells of every base test split, the
anchor rule, gold fed back as a submission scoring 1.0 with full coverage
on every tier, and the comparator behaviours the rule tables rely on.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from madi_bench.evaluation import fusion_rules as R
from madi_bench.evaluation import load_gold, score_fusion
from madi_bench.evaluation.fusion_gold import ANCHOR_ID, ANCHOR_PRIORITY, anchors_by_cluster, pick_anchor

ROOT = Path(__file__).resolve().parents[1]
# MADI_BENCH_USECASES points the tests at another task tree
# (default: the tasks shipped next to the package)
_env = os.environ.get("MADI_BENCH_USECASES")
USECASES = Path(_env) if _env else next((p for p in (ROOT / "use cases", ROOT / "usecases") if p.is_dir()), None)
DOMAINS = ("companies", "games", "music", "products", "papers")
TIERS = ("base", "easy", "medium", "hard")
KEPT_TEST_CELLS = {"companies": 610, "games": 793, "music": 726, "products": 836, "papers": 922}

needs_tasks = pytest.mark.skipif(USECASES is None, reason="task folders not found next to the package")


def task_dir(domain: str, tier: str) -> Path:
    return USECASES / domain / tier


@needs_tasks
@pytest.mark.parametrize("domain", DOMAINS)
def test_base_gold_has_100_records_with_primary_anchors_and_the_pinned_kept_cells(domain):
    gold = load_gold(task_dir(domain, "base"))
    assert len(gold) == 100
    assert gold[ANCHOR_ID].astype(str).str.startswith(ANCHOR_PRIORITY[domain][0]).all()
    from madi_bench.evaluation.fusion_gold import graded_cells, target_schema_attributes
    attrs = [a for a in target_schema_attributes(task_dir(domain, "base")) if a in gold.columns]
    assert graded_cells(gold, attrs) == KEPT_TEST_CELLS[domain]


@needs_tasks
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_gold_as_submission_scores_one_with_full_coverage(domain, tier):
    tdir = task_dir(domain, tier)
    if not (tdir / "input" / "fusion").is_dir():
        pytest.skip(f"{domain}/{tier} not present")
    gold = load_gold(tdir)
    fused = gold.copy()
    fused["_id"] = fused[ANCHOR_ID]
    result = score_fusion(tdir, fused)
    assert result["n_gold"] == 100
    assert result["gold_coverage"] == 1.0 and result["attribute_cell_coverage"] == 1.0
    assert result["overall_accuracy_all_gold"] == pytest.approx(1.0)
    assert result["total_evaluations"] == result["gold_cells_total"]


@needs_tasks
def test_membership_based_anchor_and_a_missing_entity_counts_as_wrong():
    tdir = task_dir("games", "base")
    gold = load_gold(tdir)
    fused = gold.drop(columns=[ANCHOR_ID]).copy()
    fused["_id"] = [f"c{i}" for i in range(len(fused))]
    membership = pd.DataFrame({"record_id": gold[ANCHOR_ID].astype(str), "source": "metacritic",
                               "cluster_id": fused["_id"]})
    full = score_fusion(tdir, fused, membership)
    assert full["overall_accuracy_all_gold"] == pytest.approx(1.0)
    half = score_fusion(tdir, fused.iloc[:50], membership)
    assert half["gold_coverage"] == pytest.approx(0.5)
    assert half["overall_accuracy_evaluated"] == pytest.approx(1.0)
    assert half["overall_accuracy_all_gold"] == pytest.approx(0.5)
    # dropping a column costs exactly the cells it held
    no_publisher = score_fusion(tdir, fused.drop(columns=["publisher"]), membership)
    assert no_publisher["missing_schema_attributes"] == ["publisher"]
    assert no_publisher["attribute_cell_coverage"] < 1.0
    assert no_publisher["overall_accuracy_all_gold"] == pytest.approx(no_publisher["attribute_cell_coverage"])


def test_anchor_rule_prefers_the_primary_source_then_the_smallest_fallback_id():
    assert pick_anchor(["sales_3", "dbpedia_3", "metacritic_3"], "games") == "metacritic_3"
    assert pick_anchor(["sales_3", "dbpedia_9", "dbpedia_3"], "games") == "dbpedia_3"
    assert pick_anchor(["other_3"], "games") is None
    membership = pd.DataFrame([
        {"record_id": "sales_1", "cluster_id": "c1"}, {"record_id": "metacritic_1", "cluster_id": "c1"},
        {"record_id": "dbpedia_50", "cluster_id": "c2"}, {"record_id": "dbpedia_7", "cluster_id": "c2"},
    ])
    assert anchors_by_cluster(membership, "games").to_dict() == {"c1": "metacritic_1", "c2": "dbpedia_50"}


def test_comparators_tolerate_representation_not_disagreement():
    assert R.tokenized_match("The Witcher 3: Wild Hunt", "the witcher 3 wild hunt")
    assert not R.tokenized_match("The Witcher 3", "The Witcher 2")
    assert R.relative_tolerance_match(1000, 1019, tolerance=0.02) and not R.relative_tolerance_match(1000, 1030, tolerance=0.02)
    assert R.epsilon_tolerance_match(7.8, 8.0, tolerance=0.2) and not R.epsilon_tolerance_match(7.5, 8.0, tolerance=0.2)
    assert R.year_only_match("2018-01-01", "2018") and not R.year_only_match("2018-01-01", "2019-01-01")
    assert R.full_date_match("2002/05/01", "2002-05-01 00:00:00") and not R.full_date_match("2002", "2002-05-01")
    assert R.gold_elements_present('["Action", "Adventure", "RPG"]', "Action|Adventure")
    assert not R.gold_elements_present("Action", "Action|Adventure")
    assert R.case_folded_set_equality("a|b", '["B", "A"]') and not R.case_folded_set_equality("a|b|c", "a|b")
    assert R.folded_person_set_equality('["Rosalía Mera"]', "Rosalia Mera")
    assert R.authors_tokenset_equality("adam j struck|mary a wood", "adam j. struck|mary a. wood")
    assert R.usb_if_hardware_strict("USB 3.0", "USB 3.2 Gen 1") and not R.usb_if_hardware_strict("USB 3.0", "USB 3.2 Gen 2")
    assert R.punct_collapse_match("Ace - A80", "Ace A80") and R.casefold_match("blue", "Blue")
    assert R.unicode_fold_tokens("Top‐k queries", "top-k queries") and not R.unicode_fold_tokens("Deep Learning", "Deep Learning: A Survey")
    assert R.count_drift_match(9, 11) and R.count_drift_match(1000, 1090) and not R.count_drift_match(1000, 1200)
    assert R.hardware_strict_spec_match("GDDR6", "gddr6") and not R.hardware_strict_spec_match("GDDR6", "GDDR5")


def test_rule_tables_cover_every_domain():
    for domain in DOMAINS:
        table = R.describe(domain)
        assert table and all("comparator" in row for row in table)


# ----------------------------------------------------------- the submission contract

def _gold_as_submission(tdir: Path):
    gold = load_gold(tdir)
    fused = gold.drop(columns=[ANCHOR_ID]).copy()
    fused["_id"] = [f"c{i}" for i in range(len(fused))]
    anchors = gold[ANCHOR_ID].astype(str).tolist()
    membership = pd.DataFrame({"record_id": anchors, "source": "x", "cluster_id": fused["_id"]})
    return gold, fused, membership, anchors


@needs_tasks
def test_membership_fusion_sources_and_numeric_ids_all_anchor_the_same():
    tdir = task_dir("games", "base")
    gold, fused, membership, anchors = _gold_as_submission(tdir)
    by_membership = score_fusion(tdir, fused, membership)
    with_sources = fused.copy()
    with_sources["_fusion_sources"] = [[a] for a in anchors]
    by_sources = score_fusion(tdir, with_sources)
    numeric = fused.copy()
    numeric["_id"] = range(len(numeric))
    numeric_membership = membership.assign(cluster_id=range(len(membership)))
    by_numeric = score_fusion(tdir, numeric, numeric_membership)
    for result in (by_membership, by_sources, by_numeric):
        assert result["overall_accuracy_all_gold"] == pytest.approx(1.0)
        assert result["n_gold_evaluated"] == 100 and result["n_gold_aligned"] == 100
    # duplicate anchors: the first row counts, the rest are reported
    doubled = score_fusion(tdir, pd.concat([fused, fused], ignore_index=True),
                           pd.concat([membership, membership], ignore_index=True))
    assert doubled["n_duplicate_anchor_rows"] == 100 and doubled["overall_accuracy_all_gold"] == pytest.approx(1.0)
    twice = fused.copy()
    twice["_id"] = [f"d{i}" for i in range(len(twice))]
    twice_membership = membership.assign(cluster_id=twice["_id"])
    both = score_fusion(tdir, pd.concat([fused, twice], ignore_index=True),
                        pd.concat([membership, twice_membership], ignore_index=True))
    assert both["n_duplicate_anchor_rows"] == 100
    assert both["overall_accuracy_all_gold"] == pytest.approx(1.0)
    # extra rows with foreign anchors do not change the score
    extra = fused.iloc[:3].copy()
    extra["_id"] = ["x1", "x2", "x3"]
    extra_membership = pd.DataFrame({"record_id": ["metacritic_9999991", "metacritic_9999992", "metacritic_9999993"],
                                     "source": "x", "cluster_id": ["x1", "x2", "x3"]})
    padded = score_fusion(tdir, pd.concat([fused, extra], ignore_index=True),
                          pd.concat([membership, extra_membership], ignore_index=True))
    assert padded["overall_accuracy_all_gold"] == pytest.approx(1.0) and padded["n_submitted_aligned"] == 103


@needs_tasks
def test_products_base_accepts_the_visible_currency():
    """dataset_N source names and raw ids (what a system reads from the
    visible files) score like the gold's products_N_ ids."""
    tdir = task_dir("products", "base")
    gold, fused, membership, anchors = _gold_as_submission(tdir)
    visible = membership.copy()
    visible["source"] = [f"dataset_{a.split('_')[1]}" for a in anchors]
    visible["record_id"] = [a.split("_", 2)[2] for a in anchors]
    result = score_fusion(tdir, fused, visible)
    assert result["overall_accuracy_all_gold"] == pytest.approx(1.0)
    with_sources = fused.copy()
    with_sources["_fusion_sources"] = [[a.split("_", 2)[2]] for a in anchors]
    with_sources["_fusion_source_datasets"] = [[f"prod{a.split('_')[1]}"] for a in anchors]
    assert score_fusion(tdir, with_sources)["overall_accuracy_all_gold"] == pytest.approx(1.0)


@needs_tasks
@pytest.mark.parametrize("domain", ["games", "papers", "products"])
def test_validation_split_scores_like_the_test_split(domain):
    tdir = task_dir(domain, "base")
    gold = load_gold(tdir, split="validation")
    fused = gold.copy()
    fused["_id"] = fused[ANCHOR_ID]
    result = score_fusion(tdir, fused, split="validation")
    assert result["split"] == "validation" and result["n_gold"] == 100
    assert result["overall_accuracy_all_gold"] == pytest.approx(1.0)


@needs_tasks
def test_unanchorable_submission_is_an_error_not_a_zero():
    from madi_bench.evaluation import SubmissionError
    tdir = task_dir("games", "base")
    gold, fused, membership, anchors = _gold_as_submission(tdir)
    wrong = membership.assign(record_id=["nowhere_" + a for a in anchors])
    with pytest.raises(SubmissionError):
        score_fusion(tdir, fused, wrong)
    with pytest.raises(SubmissionError):
        score_fusion(tdir, fused.drop(columns=["_id"]), membership)


FROZEN_RULES = {
    "companies": [("name", "tokenized_match", {}), ("revenue", "relative_tolerance_match", {"tolerance": 0.02}),
                  ("assets", "relative_tolerance_match", {"tolerance": 0.02}), ("keypeople", "folded_person_set_equality", {}),
                  ("founded", "year_only_match", {}), ("country", "tokenized_match", {}), ("city", "tokenized_match", {}),
                  ("industry", "exact_match", {})],
    "music": [("name", "tokenized_match", {}), ("artist", "tokenized_match", {}),
              ("duration", "numeric_tolerance_match", {"tolerance": 5}), ("release-date", "full_date_match", {}),
              ("release-country", "tokenized_match", {}), ("label", "gold_elements_present", {}),
              ("tracks", "case_folded_set_equality", {}), ("genre", "exact_match", {})],
    "games": [("name", "tokenized_match", {}), ("platform", "exact_match", {}), ("developer", "exact_match", {}),
              ("releaseYear", "year_only_match", {}), ("ESRB", "exact_match", {}),
              ("criticScore", "epsilon_tolerance_match", {"tolerance": 1}),
              ("userScore", "epsilon_tolerance_match", {"tolerance": 0.1}), ("genres", "gold_elements_present", {}),
              ("publisher", "tokenized_match", {})],
    "products": [("brand", "casefold_match", {}), ("product_type", "casefold_match", {})]
                + [(a, "relative_tolerance_match", {"tolerance": 0.02}) for a in
                   ("vram_gb", "storage_gb", "read_speed_mb_s", "write_speed_mb_s", "width_mm", "length_mm", "height_mm", "weight_g")]
                + [("chipset_name", "hardware_strict_spec_match", {}), ("bus_type", "usb_if_hardware_strict", {}),
                   ("interface_type", "hardware_strict_spec_match", {}), ("memory_type", "hardware_strict_spec_match", {}),
                   ("model", "punct_collapse_match", {}), ("model_number", "punct_collapse_match", {}),
                   ("storage_connection_type", "tokenized_match", {}), ("color", "casefold_match", {}),
                   ("form_factor", "exact_match", {})],
    "papers": [("type", "exact_match", {}), ("title", "unicode_fold_tokens", {}), ("authors", "authors_tokenset_equality", {}),
               ("publication_year", "year_only_match", {}), ("journal", "tokenized_match", {}), ("volume", "exact_match", {}),
               ("issue", "exact_match", {}), ("first_page", "exact_match", {}), ("last_page", "exact_match", {}),
               ("referenced_works_count", "exact_match", {}), ("cited_by_count", "count_drift_match", {"floor": 2, "rate": 0.10})],
}


@pytest.mark.parametrize("domain", DOMAINS)
def test_rule_tables_are_frozen(domain):
    """The strict tables, attribute by attribute, in order."""
    got = [(row["attribute"], row["comparator"], {k: v for k, v in row.items() if k not in ("attribute", "comparator")})
           for row in R.describe(domain)]
    assert got == FROZEN_RULES[domain]


def test_cell_accounting_rules():
    """A gold blank is skipped, a fused blank is wrong, a comparator exception is wrong."""
    from madi_bench.evaluation.fusion_score import _evaluate
    gold = pd.DataFrame({ANCHOR_ID: ["a", "b", "c", "d"], "platform": ["PC", None, "PC", "PC"]})
    fused = pd.DataFrame({ANCHOR_ID: ["a", "b", "c", "d"], "platform": ["PC", "PC", None, ["not", "a", "string"]]})
    scores = _evaluate(fused, gold, "games")
    assert scores["total_evaluations"] == 3 and scores["total_correct"] == 1


# ------------------------------------------------------------- normalization

@needs_tasks
@pytest.mark.parametrize("domain", DOMAINS)
def test_normalization_test_set_fed_back_scores_one(domain, tmp_path):
    import csv
    from collections import defaultdict
    from madi_bench.evaluation import score_normalization
    from madi_bench.evaluation.normalization_score import test_rows
    tdir = task_dir(domain, "base")
    rows = test_rows(tdir)
    tables = defaultdict(dict)
    for r in rows:
        tables[r["source"]].setdefault(r["source_id"], {"id": r["source_id"]})[r["attribute"]] = r["expected_value"]
    for src, recs in tables.items():
        cols = ["id"] + sorted({a for rec in recs.values() for a in rec if a != "id"})
        with open(tmp_path / f"{src}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(recs.values())
    (tmp_path / "sm_mapping.csv").write_text("source_dataset,source_column\nx,y\n")   # not a source table: skipped
    result = score_normalization(tdir, tmp_path)
    assert result["accuracy"] == 1.0 and result["record_coverage"] == 1.0 and result["rows"] == len(rows)
    assert result["skipped_files"] == ["sm_mapping.csv"]
    assert set(result["metrics"]) >= {"all", "category:identity"}


def test_normalization_comparator_and_lookup():
    from madi_bench.evaluation.normalization_score import NormalizedTables, equal
    assert equal("games", "platform", "PC", "PC") and not equal("games", "platform", "pc", "PC")
    assert equal("games", "genres", '["Action", "RPG"]', "Action|RPG") and not equal("games", "genres", "RPG|Action", "Action|RPG")
    assert equal("companies", "keypeople", "B|A", '["A", "B"]')          # keypeople is a set
    assert not equal("companies", "founded", "1888", "1888.0")            # numbers compare as text
    import tempfile, csv
    d = Path(tempfile.mkdtemp())
    with open(d / "prod1.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "brand"]); w.writerow(["123", "ADATA"])
    t = NormalizedTables(d)
    assert t.lookup("products_1", "products_1_123")["brand"] == "ADATA"
    assert t.lookup("dataset_1", "123")["brand"] == "ADATA"
    assert t.lookup("products_2", "products_2_123") is None


def test_cli_reports_bad_input_as_a_message(tmp_path, capsys):
    from madi_bench.evaluation.cli import score_fusion_main
    if USECASES is None:
        pytest.skip("task folders not found")
    bad = tmp_path / "fused.csv"
    bad.write_text("name,platform\nA,PC\n")
    assert score_fusion_main(["--task", str(task_dir("games", "base")), "--fused", str(bad)]) == 2
    assert "_id" in capsys.readouterr().err


# ------------------------------------------------------ variant-only records

def test_anchor_rule_skips_excluded_ids():
    """A variant's own records carry their source's usual id format; the
    submission side never lets one take the anchor slot."""
    own = {"metacritic_9000001", "http://dbpedia.org/resource/1288241"}
    assert pick_anchor(["metacritic_9000001", "metacritic_3"], "games") == "metacritic_9000001"
    assert pick_anchor(["metacritic_9000001", "metacritic_3"], "games", own) == "metacritic_3"
    assert pick_anchor(["metacritic_9000001", "sales_3"], "games", own) == "sales_3"
    companies = ["http://dbpedia.org/resource/Acme", "http://dbpedia.org/resource/1288241"]
    assert pick_anchor(companies, "companies") == "http://dbpedia.org/resource/1288241"
    assert pick_anchor(companies, "companies", own) == "http://dbpedia.org/resource/Acme"
    membership = pd.DataFrame([
        {"record_id": "metacritic_9000001", "cluster_id": "c1"}, {"record_id": "metacritic_1", "cluster_id": "c1"},
        {"record_id": "metacritic_9000001", "cluster_id": "c2"},
    ])
    assert anchors_by_cluster(membership, "games").to_dict() == {"c1": "metacritic_9000001", "c2": "metacritic_9000001"}
    assert anchors_by_cluster(membership, "games", own).to_dict() == {"c1": "metacritic_1"}


def _variant_copy(tmp_path: Path, domain: str = "games", tier: str = "hard") -> Path:
    import shutil
    root = tmp_path / domain / tier
    shutil.copytree(task_dir(domain, tier) / "input", root / "input", ignore=shutil.ignore_patterns("data_backups"))
    shutil.copytree(task_dir(domain, "base") / "input" / "data", tmp_path / domain / "base" / "input" / "data")
    return root


@needs_tasks
def test_variant_records_never_anchor_and_the_base_folder_is_required(tmp_path):
    """games hard on a copy: every submitted cluster lists, first, a record
    the variant added with the anchor-source prefix; the score stays perfect.
    Without the sibling base folder a variant cannot be scored."""
    from madi_bench.evaluation.cli import score_fusion_main
    from madi_bench.evaluation.fusion_gold import variant_only_record_ids
    from madi_bench.evaluation.fusion_score import derive_submission_anchor

    if not (task_dir("games", "hard") / "input" / "fusion").is_dir():
        pytest.skip("games/hard not present")
    tdir = _variant_copy(tmp_path)
    before = variant_only_record_ids(tdir)
    assert variant_only_record_ids(task_dir("games", "base")) == frozenset()
    gold, fused, membership, anchors = _gold_as_submission(tdir)
    own = [f"metacritic_{9_000_000 + i}" for i in range(len(anchors))]
    with open(tdir / "input" / "data" / "metacritic.csv", "a", encoding="utf-8") as f:
        f.writelines(f"{rid},a copy the variant added\n" for rid in own)
    assert variant_only_record_ids(tdir) == before | set(own)
    first = pd.DataFrame({"record_id": own, "source": "metacritic", "cluster_id": fused["_id"]})
    submission = pd.concat([first, membership], ignore_index=True)
    result = score_fusion(tdir, fused, submission)
    assert result["gold_coverage"] == 1.0 and result["overall_accuracy_all_gold"] == pytest.approx(1.0)
    with_sources = fused.copy()
    with_sources["_fusion_sources"] = [[o, a] for o, a in zip(own, anchors)]
    assert score_fusion(tdir, with_sources)["overall_accuracy_all_gold"] == pytest.approx(1.0)
    naive = derive_submission_anchor(fused, "games", submission)
    assert set(naive[ANCHOR_ID]) == set(own)              # the rule without the exclusion

    import shutil
    shutil.rmtree(tmp_path / "games" / "base")
    with pytest.raises(FileNotFoundError, match="base task folder"):
        score_fusion(tdir, fused, submission)
    fused_csv = tmp_path / "fused.csv"
    fused.to_csv(fused_csv, index=False)
    submission.to_csv(tmp_path / "membership.csv", index=False)
    assert score_fusion_main(["--task", str(tdir), "--fused", str(fused_csv),
                              "--membership", str(tmp_path / "membership.csv")]) == 2
