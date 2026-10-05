"""Schema-matching scorer: a submitted ``sm_mapping.csv`` against the task's
gold schema mapping.

Uses PyDI's ``SchemaMappingEvaluator`` (complete mode: precision and recall
over the gold correspondences).

Submission columns: ``source_dataset, source_column, target_dataset,
target_column, score``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import SubmissionContractError, aliases, gold
from .tasks import TaskSpec


def load_sm_submission(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, index_col=False)
    except pd.errors.EmptyDataError:
        raise SubmissionContractError(f"sm_mapping file is empty: {path}")
    except pd.errors.ParserError as exc:
        raise SubmissionContractError(f"malformed sm_mapping file {path}: {exc}")
    missing = [c for c in gold.SM_COLUMNS if c not in frame.columns]
    if missing:
        raise SubmissionContractError(
            f"sm_mapping.csv lacks required columns {missing} "
            f"(contract schema: {gold.SM_COLUMNS})"
        )
    return frame[gold.SM_COLUMNS]


def _canonicalize(task: TaskSpec, frame: pd.DataFrame) -> pd.DataFrame:
    """SchemaMappingEvaluator matches exact 4-tuples, but the gold's
    target_dataset label is an arbitrary string that differs between tiers
    ('target_schema' on base, '<domain>' on variants) -- there is exactly one
    target schema per task, so it is pinned to a constant on both sides.
    Source dataset names go through the alias table (products base:
    dataset_N <-> products_N), and all fields are whitespace-stripped."""
    out = frame.copy()
    for col in gold.SM_COLUMNS[:4]:
        out[col] = out[col].astype(str).str.strip()
    out["source_dataset"] = out["source_dataset"].map(
        lambda name: aliases.canonical_source_name(task, name)
    )
    out["target_dataset"] = "__target__"
    return out


def score_sm(task: TaskSpec, submission_path: Path) -> dict:
    from PyDI.schemamatching.evaluation import SchemaMappingEvaluator

    submission = load_sm_submission(submission_path)
    gold_mapping = gold.load_sm_gold(task)
    submitted_targets = set(submission["target_dataset"].astype(str).str.strip())
    gold_targets = set(gold_mapping["target_dataset"].astype(str).str.strip())
    result = SchemaMappingEvaluator.evaluate(
        _canonicalize(task, submission),
        _canonicalize(task, gold_mapping[gold.SM_COLUMNS]).assign(
            label=gold_mapping["label"].to_numpy()
        ),
        complete=True,
        label_column="label",
    )
    return {
        "stage": "schema_matching",
        "task_id": task.task_id,
        "n_submitted": len(submission),
        "n_gold": int(gold_mapping["label"].sum()),
        "target_dataset_mismatch": (
            sorted(submitted_targets) if submitted_targets != gold_targets else None
        ),
        **{k: result[k] for k in ("precision", "recall", "f1") if k in result},
    }
