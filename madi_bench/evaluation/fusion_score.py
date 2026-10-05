"""Score a fused table against a task's fusion gold set.

Input: the fused table (one row per fused entity) with an ``_id`` column and
either a membership table (``record_id, source, cluster_id``: which source
records went into which fused row -- the benchmark's submission contract) or
a ``_fusion_sources`` column listing the member record ids of each row (what
PyDI's fuser writes). Ids are compared as strings. A submitted row is
anchored on a real record: in a variant task the records the variant added
(its source ids that are no base source id; the base task folder must sit
next to the variant, ``<domain>/base``) never anchor.

Headline (all-gold accuracy)::

    overall_accuracy_all_gold = overall_accuracy_evaluated
                                * gold_coverage * attribute_cell_coverage

* ``overall_accuracy_evaluated`` is the accuracy over the gold cells of the
  gold records the submission reached, cell-weighted over all attributes; a
  gold record is reached by a fused row carrying its anchor, or (as PyDI's
  evaluator aligns) by a fused row whose ``_fusion_sources`` list any member
  of the gold record -- ``n_gold_aligned`` counts the reached records;
* ``gold_coverage`` is the share of gold records whose ANCHOR a fused row
  carries -- an entity the system did not produce is a wrong answer, not an
  absent one (``n_gold_evaluated``; with ``_fusion_sources`` present this can
  be smaller than ``n_gold_aligned``, exactly as in the reference numbers);
* ``attribute_cell_coverage`` is the share of graded gold cells whose
  attribute the submission carries at all -- dropping a weak column costs
  exactly the cells it held.

A gold cell is graded when it holds a value; cells blanked in the gold
(``v2_removed``) are skipped, so the denominator is the kept cells (610 /
793 / 726 / 836 / 922 on the companies / games / music / products / papers
base test splits). A graded cell is correct when the domain's comparator for
that attribute (``fusion_rules.STRICT_RULES``) accepts the fused value; a
missing fused value is wrong.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pandas as pd

from . import fusion_rules as rules
from .fusion_gold import (
    ANCHOR_ID,
    anchors_by_cluster,
    domain_of,
    graded_cells,
    is_present,
    load_gold,
    parse_id_list,
    pick_anchor,
    symmetric_prep,
    target_schema_attributes,
    tier_of,
    variant_only_record_ids,
)

# every spelling of the four products sources in circulation: the visible
# data files (dataset_N), the gold (products_N), the LLM pipeline (prodN)
_PRODUCTS_BASE_SOURCES = {f"{spelling}{n}": f"products_{n}"
                          for n in range(1, 5) for spelling in ("dataset_", "products_", "prod")}
_PRODUCTS_PREFIX = re.compile(r"^products_[1-4]_")


class SubmissionError(ValueError):
    pass


# ------------------------------------------------- products base currency

def _prefix_record_id(source: str, record_id: str) -> str:
    """``products_<N>_<raw>``; an id that already carries a products prefix
    is left alone whatever source name came with it."""
    rid = str(record_id)
    if _PRODUCTS_PREFIX.match(rid):
        return rid
    return f"{source}_{rid}"


def _products_base_membership(membership: pd.DataFrame) -> pd.DataFrame:
    """products base ships raw ids in the source tables (dataset_1..4.json,
    id 12198483) but ``products_<N>_<id>`` ids in the gold; a submission in
    the visible currency is translated here."""
    out = membership.copy()
    out["source"] = out["source"].map(lambda s: _PRODUCTS_BASE_SOURCES.get(str(s), str(s)))
    out["record_id"] = [_prefix_record_id(s, r) for s, r in zip(out["source"], out["record_id"])]
    return out


def _products_base_fused_sources(fused: pd.DataFrame) -> pd.DataFrame:
    if "_fusion_sources" not in fused.columns:
        return fused
    out = fused.copy()
    datasets = out["_fusion_source_datasets"] if "_fusion_source_datasets" in out.columns else [None] * len(out)
    new_sources = []
    for sources, names in zip(out["_fusion_sources"], datasets):
        ids = parse_id_list(sources)
        canon = [_PRODUCTS_BASE_SOURCES.get(str(d), str(d)) for d in (parse_id_list(names) or [])]
        canon += [""] * (len(ids) - len(canon))
        new_sources.append([_prefix_record_id(n, r) if n else r for r, n in zip(ids, canon)])
    out["_fusion_sources"] = new_sources
    return out


# ------------------------------------------------------------ anchors

def derive_submission_anchor(fused: pd.DataFrame, domain: str,
                             membership: pd.DataFrame | None = None,
                             exclude=None) -> pd.DataFrame:
    """Add ``_anchor_id`` to a fused table: from the membership table when
    given, else from ``_fusion_sources``; rows without a member from any
    anchor-priority source are dropped. Record ids in ``exclude`` (a
    variant's own records, ``variant_only_record_ids``) never anchor."""
    out = fused.copy()
    if ANCHOR_ID in out.columns and out[ANCHOR_ID].notna().any():
        return out.dropna(subset=[ANCHOR_ID])
    if "_id" not in out.columns:
        raise SubmissionError("the fused table lacks the required '_id' column")
    if membership is not None and not membership.empty:
        missing = {"record_id", "source", "cluster_id"} - set(membership.columns)
        if missing:
            raise SubmissionError(f"the membership table lacks the columns {sorted(missing)}")
        membership = membership.assign(cluster_id=membership["cluster_id"].astype(str))
        out[ANCHOR_ID] = out["_id"].astype(str).map(anchors_by_cluster(membership, domain, exclude))
    elif "_fusion_sources" in out.columns:
        out[ANCHOR_ID] = out["_fusion_sources"].map(lambda v: pick_anchor(parse_id_list(v), domain, exclude))
    else:
        raise SubmissionError("cannot derive the fusion anchor: give a membership table or a "
                              "'_fusion_sources' column")
    out = out.dropna(subset=[ANCHOR_ID])
    if len(fused) and out.empty:
        raise SubmissionError(
            "no fused row could be anchored: no cluster of the membership table (or no _fusion_sources "
            "list) contains a record from an anchor-priority source -- check that cluster_id matches "
            "the fused _id and that record ids are the source tables' ids")
    return out


# ----------------------------------------------------------- evaluation

def _align(submission: pd.DataFrame, gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gold records reached by the submission, in gold order: a fused row is
    reachable by its anchor and by every id in its ``_fusion_sources``."""
    fused_clean = submission.dropna(subset=[ANCHOR_ID]).reset_index(drop=True)
    index: dict[str, Any] = {}
    for idx, row in fused_clean.iterrows():
        index[str(row[ANCHOR_ID])] = idx
        sources = row.get("_fusion_sources")
        if sources is None or (isinstance(sources, float) and pd.isna(sources)):
            continue
        if isinstance(sources, str):
            try:
                parsed = ast.literal_eval(sources)
            except (ValueError, SyntaxError):
                parsed = [sources]
        else:
            parsed = sources
        parsed = list(parsed) if isinstance(parsed, (list, tuple, set)) else []
        for source_id in parsed:
            if isinstance(source_id, str) and source_id:
                index[source_id] = idx
    fused_rows, gold_rows = [], []
    for _, gold_row in gold.iterrows():
        idx = index.get(str(gold_row[ANCHOR_ID]))
        if idx is None:
            continue
        fused_rows.append(fused_clean.loc[idx])
        gold_rows.append(gold_row)
    if not fused_rows:
        return pd.DataFrame(), pd.DataFrame()
    return (pd.DataFrame(fused_rows).reset_index(drop=True),
            pd.DataFrame(gold_rows).reset_index(drop=True))


def _evaluate(aligned_fused: pd.DataFrame, aligned_gold: pd.DataFrame, domain: str) -> dict[str, Any]:
    """Cell accounting exactly as PyDI's DataFusionEvaluator: a gold cell
    without a value is skipped, a missing fused value is wrong, a
    comparator exception is wrong."""
    attributes = [c for c in aligned_gold.columns
                  if c in aligned_fused.columns and c != ANCHOR_ID and not c.startswith("_fusion_")]
    total_correct = total_evaluated = 0
    per_attribute: dict[str, dict[str, Any]] = {}
    for attribute in attributes:
        compare = rules.comparator_for(domain, attribute)
        correct = total = 0
        for fused_value, expected_value in zip(aligned_fused[attribute], aligned_gold[attribute]):
            if rules.is_missing(expected_value):
                continue
            total += 1
            if rules.is_missing(fused_value):
                continue
            try:
                if compare(fused_value, expected_value):
                    correct += 1
            except Exception:  # noqa: BLE001 — a comparator that cannot read the value says wrong
                continue
        per_attribute[attribute] = {"accuracy": correct / total if total else 0.0,
                                    "correct": correct, "count": total}
        total_correct += correct
        total_evaluated += total
    accuracies = [r["accuracy"] for r in per_attribute.values() if r["count"] > 0]
    return {"overall_accuracy": total_correct / total_evaluated if total_evaluated else 0.0,
            "macro_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "total_evaluations": total_evaluated, "total_correct": total_correct,
            "num_evaluated_records": len(aligned_fused), "per_attribute": per_attribute}


def score_fusion(task_dir: Path, fused: pd.DataFrame, membership: pd.DataFrame | None = None,
                 split: str = "test") -> dict[str, Any]:
    """Score ``fused`` (with ``membership``) against the task's gold ``split``."""
    task_dir = Path(task_dir)
    domain, tier = domain_of(task_dir), tier_of(task_dir)
    if "_id" not in fused.columns and ANCHOR_ID not in fused.columns:
        raise SubmissionError("the fused table lacks the required '_id' column")
    if membership is not None and not membership.empty:
        missing = {"record_id", "source", "cluster_id"} - set(membership.columns)
        if missing:
            raise SubmissionError(f"the membership table lacks the columns {sorted(missing)}")
    if domain == "products" and tier == "base":
        fused = _products_base_fused_sources(fused)
        if membership is not None and not membership.empty:
            membership = _products_base_membership(membership)
    gold = load_gold(task_dir, split)
    schema_attributes = [a for a in target_schema_attributes(task_dir) if a in gold.columns]
    gold = gold[schema_attributes + [ANCHOR_ID]]
    # a variant's own records never anchor a submitted cluster (gold anchors are base records)
    exclude = variant_only_record_ids(task_dir)
    submission = derive_submission_anchor(symmetric_prep(fused, domain), domain, membership, exclude)
    n_duplicate_anchor_rows = int(submission.duplicated(ANCHOR_ID).sum())
    submission = submission.drop_duplicates(ANCHOR_ID, keep="first")

    aligned_fused, aligned_gold = _align(submission, gold)
    scores = _evaluate(aligned_fused, aligned_gold, domain) if len(aligned_gold) else {
        "overall_accuracy": 0.0, "macro_accuracy": 0.0, "total_evaluations": 0,
        "total_correct": 0, "num_evaluated_records": 0, "per_attribute": {}}

    n_gold = len(gold)
    gold_anchors = set(gold[ANCHOR_ID].astype(str))
    n_evaluated = len(gold_anchors & set(submission[ANCHOR_ID].astype(str)))
    gold_coverage = n_evaluated / n_gold if n_gold else 0.0
    rule_attributes = [a for a, _, _ in rules.rules_for(domain) if a in gold.columns]
    missing_attributes = [a for a in rule_attributes if a not in submission.columns]
    gold_cells_total = graded_cells(gold, schema_attributes)
    gold_cells_gradeable = graded_cells(gold, [a for a in schema_attributes if a in submission.columns])
    attribute_cell_coverage = gold_cells_gradeable / gold_cells_total if gold_cells_total else 1.0
    overall = scores["overall_accuracy"]
    all_gold = overall * gold_coverage * attribute_cell_coverage

    def dtype_family(series: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        return "object"

    dtype_mismatches = sorted(c for c in set(submission.columns) & set(gold.columns)
                              if c != ANCHOR_ID and dtype_family(submission[c]) != dtype_family(gold[c]))
    return {
        "task": f"{domain}_{tier}", "split": split, "gold_version": gold.attrs.get("gold_version", "v1"),
        "rules": "strict",
        "overall_accuracy_all_gold": all_gold,
        "overall_accuracy_evaluated": overall,
        "gold_coverage": gold_coverage,
        "attribute_cell_coverage": attribute_cell_coverage,
        "n_gold": n_gold, "n_gold_evaluated": n_evaluated, "n_gold_missing": n_gold - n_evaluated,
        "n_gold_aligned": int(scores["num_evaluated_records"]),
        "gold_cells_total": gold_cells_total, "gold_cells_gradeable": gold_cells_gradeable,
        "n_submitted_aligned": len(submission), "n_duplicate_anchor_rows": n_duplicate_anchor_rows,
        "missing_attributes": missing_attributes,
        "missing_schema_attributes": [a for a in schema_attributes if a not in submission.columns],
        "dtype_mismatches": dtype_mismatches,
        "per_attribute_accuracy_all_gold": {a: r["accuracy"] * gold_coverage
                                            for a, r in scores["per_attribute"].items()},
        "per_attribute_accuracy_evaluated": {a: r["accuracy"] for a, r in scores["per_attribute"].items()},
        "per_attribute_count": {a: r["count"] for a, r in scores["per_attribute"].items()},
        "macro_accuracy_evaluated": scores["macro_accuracy"],
        "total_evaluations": scores["total_evaluations"], "total_correct": scores["total_correct"],
    }


def load_table(path: Path) -> pd.DataFrame:
    """A fused or membership table from CSV (gzip accepted); ids stay strings."""
    return pd.read_csv(path, dtype={"_id": str, "record_id": str, "cluster_id": str, "source": str})
