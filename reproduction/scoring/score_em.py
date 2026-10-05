"""Entity-matching scorer: submitted correspondences against the task's
labeled test pairs.

Pair direction is normalized on BOTH sides before evaluation (each pair is
ordered by lexicographic id order): MaDI ids carry dataset prefixes, so this
makes scoring direction-agnostic (the games base files name the pair
metacritic_2_dbpedia in train and dbpedia_2_metacritic in test, and a system
may orient its pairs either way).

``id_coverage`` is a diagnostic: the share of submitted ids that exist in the
gold pair universe at all; it separates "bad pipeline" from "wrong id
convention" before a near-zero score is read.

Submission columns: ``id1, id2[, score]`` (``score`` is not used for
scoring: the evaluation takes every submitted pair as a predicted match).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import SubmissionContractError, gold
from .tasks import TaskSpec

CORRESPONDENCE_COLUMNS = ["id1", "id2", "score"]


def load_correspondences(path: Path) -> pd.DataFrame:
    try:
        # index_col=False: ragged rows must fail loudly, not silently shift
        # ids into the index.
        frame = pd.read_csv(path, dtype={"id1": str, "id2": str}, index_col=False)
    except pd.errors.EmptyDataError:
        raise SubmissionContractError(f"correspondences file is empty: {path}")
    except pd.errors.ParserError as exc:
        raise SubmissionContractError(f"malformed correspondences file {path}: {exc}")
    # `score` is part of the submission contract but never read by the
    # scorer (evaluate_matching is called with threshold=None), so a missing
    # score column is recorded, not fatal.
    missing = [c for c in CORRESPONDENCE_COLUMNS
               if c not in frame.columns and c != "score"]
    if missing:
        raise SubmissionContractError(
            f"correspondences.csv lacks required columns {missing}"
        )
    score_column_absent = "score" not in frame.columns
    if score_column_absent:
        frame["score"] = 1.0
    frame = frame[CORRESPONDENCE_COLUMNS].copy()
    frame.attrs["score_column_absent"] = score_column_absent
    # Non-numeric score cells must not reach the evaluator; threshold=None
    # never reads scores, so coerce and count.
    numeric = pd.to_numeric(frame["score"], errors="coerce")
    frame.attrs["n_nonnumeric_scores"] = int(
        (numeric.isna() & frame["score"].notna()).sum()
    )
    frame["score"] = numeric
    return frame


def canonicalize_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Order each (id1,id2) pair lexicographically -- direction-agnostic."""
    frame = frame.copy()
    swap = frame["id1"] > frame["id2"]
    frame.loc[swap, ["id1", "id2"]] = frame.loc[swap, ["id2", "id1"]].to_numpy()
    return frame


def _pair_id_universe(pairs: pd.DataFrame) -> tuple[set, set]:
    return set(pairs["id1"]), set(pairs["id2"])


def score_em(
    task: TaskSpec,
    correspondences_path: Path,
    variant_gold: str | None = None,
) -> dict:
    """Positive-class P/R/F1 via PyDI's EntityMatchingEvaluator per pair
    file. ``per_pair_mean`` is the paper's protocol (the UNWEIGHTED mean over
    the task's source-pair F1s); the ``micro`` block pools TP/FP/FN across
    the task's pair files.

    NOTE on 'precision': PyDI's evaluate_matching counts a false positive
    only when a prediction hits an explicitly LABELED negative pair --
    predictions outside the labeled test set are ignored (benchmark
    convention). ``n_predictions_outside_labeled_set`` shows how much of the
    submission the metric never saw."""
    from PyDI.entitymatching.evaluation import EntityMatchingEvaluator

    loaded = load_correspondences(correspondences_path)
    n_nonnumeric = loaded.attrs.get("n_nonnumeric_scores", 0)
    correspondences = canonicalize_pairs(loaded)
    per_pair: dict[str, dict] = {}
    tp = fp = fn = 0
    matched_any = pd.Series(False, index=correspondences.index)
    all_gold_ids: set = set()
    for pair_name, test_pairs in gold.load_em_test_pairs(task, variant_gold).items():
        test = canonicalize_pairs(test_pairs)
        side1, side2 = _pair_id_universe(test)
        # Side-pure filter: after canonicalization each side of a pair file
        # is dataset-pure, so require id1 in side1 AND id2 in side2 --
        # correspondences of other source pairs must not dilute this pair.
        in_pair = correspondences["id1"].isin(side1) & correspondences["id2"].isin(side2)
        matched_any |= in_pair
        relevant = correspondences[in_pair]
        all_gold_ids |= side1 | side2
        result = EntityMatchingEvaluator.evaluate_matching(
            relevant[CORRESPONDENCE_COLUMNS], test, threshold=None,
            max_logged_instances=0,  # skip PyDI's per-instance debug loop
        )
        labeled = set(zip(test["id1"], test["id2"]))
        submitted = set(zip(relevant["id1"], relevant["id2"]))
        per_pair[pair_name] = {
            "precision": result.get("precision"),
            "recall": result.get("recall"),
            "f1": result.get("f1"),
            "n_test_pairs": len(test),
            "n_relevant_correspondences": len(relevant),
            "n_predictions_outside_labeled_set": len(submitted - labeled),
        }
        tp += result.get("true_positives", 0)
        fp += result.get("false_positives", 0)
        fn += result.get("false_negatives", 0)

    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    submitted_ids = set(correspondences["id1"].dropna()) | set(
        correspondences["id2"].dropna()
    )
    id_coverage = (
        len(submitted_ids & all_gold_ids) / len(submitted_ids)
        if submitted_ids else 0.0
    )
    return {
        "stage": "entity_matching",
        "task_id": task.task_id,
        "score_column_absent": bool(
            getattr(correspondences, "attrs", {}).get("score_column_absent")),
        "n_correspondences": len(correspondences),
        "n_correspondences_unmatched_to_any_pair": int((~matched_any).sum()),
        "n_nonnumeric_scores": n_nonnumeric,
        "id_coverage": id_coverage,
        "per_pair": per_pair,
        # Paper protocol: the UNWEIGHTED MEAN over the task's source-pair
        # scores. The pooled micro block weights pairs by their test-pair counts.
        "per_pair_mean": {
            key: (round(sum(vals) / len(vals), 6) if vals else None)
            for key, vals in (
                (metric, [pp[metric] for pp in per_pair.values()
                          if pp.get(metric) is not None])
                for metric in ("precision", "recall", "f1")
            )
        },
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
    }
