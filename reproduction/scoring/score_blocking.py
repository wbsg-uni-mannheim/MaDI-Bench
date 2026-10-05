"""Blocking scorer: pair completeness / pair quality (and reduction ratio
when the cross-product size is given) of a submitted candidate set against
the task's labeled test pairs.

Pair completeness is computed over exactly the candidates given (a sample
would read as a poor blocker). Direction is canonicalized on both sides like
the EM scorer.

Submission columns: ``id1, id2`` (one file with the candidates of every
source pair).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import SubmissionContractError, gold
from .score_em import canonicalize_pairs
from .tasks import TaskSpec


def load_candidates(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype=str, index_col=False)
    except pd.errors.EmptyDataError:
        raise SubmissionContractError(f"candidates file is empty: {path}")
    except pd.errors.ParserError as exc:
        raise SubmissionContractError(f"malformed candidates file {path}: {exc}")
    if not {"id1", "id2"} <= set(frame.columns):
        raise SubmissionContractError("candidates file needs columns id1,id2")
    return frame[["id1", "id2"]]


def score_blocking(
    task: TaskSpec,
    candidates: pd.DataFrame,
    *,
    total_possible_pairs: int | None = None,
    variant_gold: str | None = None,
) -> dict:
    from PyDI.entitymatching.evaluation import EntityMatchingEvaluator

    candidates = canonicalize_pairs(candidates.copy())
    per_pair: dict[str, dict] = {}
    for pair_name, test_pairs in gold.load_em_test_pairs(task, variant_gold).items():
        test = canonicalize_pairs(test_pairs)
        # Side-pure filter (same as score_em): one combined candidates file
        # is submitted, so per-pair pair quality is computed over this pair's
        # candidates only.
        side1, side2 = set(test["id1"]), set(test["id2"])
        relevant = candidates[
            candidates["id1"].isin(side1) & candidates["id2"].isin(side2)
        ]
        # PC and PQ do not depend on total_possible_pairs, but the PyDI
        # evaluator requires it; pass a placeholder when unknown and report
        # reduction_ratio only for a real total.
        result = EntityMatchingEvaluator.evaluate_blocking(
            relevant[["id1", "id2"]],
            test,
            total_possible_pairs=total_possible_pairs or 1,
        )
        per_pair[pair_name] = {
            k: result[k]
            for k in ("pair_completeness", "pair_quality")
            if k in result
        }
        per_pair[pair_name]["reduction_ratio"] = (
            result.get("reduction_ratio") if total_possible_pairs else None
        )
        per_pair[pair_name]["n_test_pairs"] = len(test)
        per_pair[pair_name]["n_relevant_candidates"] = len(relevant)
    return {
        "stage": "blocking",
        "task_id": task.task_id,
        "n_candidates": len(candidates),
        "per_pair": per_pair,
    }
