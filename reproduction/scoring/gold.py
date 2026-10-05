"""Gold locators and parsers for schema matching and entity matching.

The fusion gold is read by ``madi_bench.evaluation`` (scoring) and by
``panel_gold.py`` (end-to-end panel).

Schema-matching gold
  base tiers:    input/schemamatching/sm_mapping_gold.json -- {"mappings": [
                 {source_dataset, source_column, target_dataset,
                  target_column, score, label}, ...]}
  variant tiers: input/schemamatching/sm_mapping.csv -- same fields as CSV
                 (no label column; every row is a true correspondence).

Entity-matching test pairs -- five spellings across domains:
  companies/music: headerless CSV, labels True/False
  games:           headerless CSV, labels TRUE/FALSE
  products:        header id1,id2,label -- labels 1/0
  papers:          header id_dblp,id_crossref,label -- labels 1/0
  In the variant tiers the evaluation split is ``*_test_corner_filled.csv``
  (the full variant gold); ``*_test_baseline_pruned.csv`` is the subset
  comparable with the base task, and the plain ``*_test.csv`` there is a copy
  of the base gold kept for reference.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from .tasks import TaskSpec

SM_COLUMNS = ["source_dataset", "source_column", "target_dataset", "target_column", "score"]

_TRUE = {"true", "1", "1.0"}
_FALSE = {"false", "0", "0.0"}
_COMPANION_MARKERS = ("_baseline_pruned", "_corner_filled")


# --- schema matching ---

def load_sm_gold(task: TaskSpec) -> pd.DataFrame:
    """The gold schema mapping as a DataFrame in the canonical CSV schema
    (+ ``label`` column, True for every gold correspondence)."""
    sm_dir = task.root / "input" / "schemamatching"
    gold_json = sm_dir / "sm_mapping_gold.json"
    gold_csv = sm_dir / "sm_mapping.csv"
    if gold_json.is_file():
        with open(gold_json) as f:
            mappings = json.load(f)["mappings"]
        frame = pd.DataFrame(mappings)
    elif gold_csv.is_file():
        frame = pd.read_csv(gold_csv)
    else:
        raise FileNotFoundError(f"no SM gold for {task.task_id} in {sm_dir}")
    missing = [c for c in SM_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"SM gold for {task.task_id} lacks columns {missing}")
    if "label" not in frame.columns:
        frame["label"] = True
    frame["label"] = frame["label"].map(_parse_label)
    return frame[SM_COLUMNS + ["label"]]


# --- entity matching ---

def _parse_label(value) -> bool:
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"unparseable EM label: {value!r}")


def _has_header(path: Path) -> bool:
    """Headerless files carry a parseable label in the LAST field of row 1;
    header rows carry a column name there (e.g. 'label').

    The last field, not the third: ids may contain commas (118 dbpedia
    entity_uris do), which pushes the label right of position 2."""
    with open(path, newline="") as f:
        first = next(csv.reader(f), None)
    if first is None or len(first) < 3:
        raise ValueError(f"not an EM pair file: {path}")
    try:
        _parse_label(first[-1])
        return False
    except ValueError:
        return True


def _read_headerless_pairs(path: Path, skip_malformed: bool) -> pd.DataFrame:
    """Parse a headerless id1,id2,label file anchored on BOTH ends.

    Splitting positionally breaks on unquoted ids that contain a comma, and
    the benchmark has them: 118 dbpedia entity_uris carry one. Only the first
    and last fields are positionally safe, so id2 is whatever lies between
    them, rejoined.
    """
    rows: list[tuple[str, str, str]] = []
    with open(path, newline="") as f:
        for lineno, fields in enumerate(csv.reader(f), start=1):
            if not fields or (len(fields) == 1 and not fields[0].strip()):
                continue
            if len(fields) < 3:
                raise ValueError(
                    f"{path}:{lineno}: EM pair row with fewer than 3 fields: {fields!r}"
                )
            try:
                _parse_label(fields[-1])
            except ValueError:
                # A comma-bearing id whose row also lost its label leaves the
                # id's tail sitting in the label position. The label is not
                # recoverable from this row, so name it instead of failing
                # with an opaque value error further down.
                if skip_malformed:
                    continue
                raise ValueError(
                    f"{path}:{lineno}: last field is not a label "
                    f"(unquoted id containing a comma, with the label missing?): "
                    f"{fields!r}"
                ) from None
            rows.append((fields[0], ",".join(fields[1:-1]), fields[-1]))
    return pd.DataFrame(rows, columns=["id1", "id2", "label"], dtype=str)


def load_em_pairs(path: Path, *, skip_malformed: bool = False) -> pd.DataFrame:
    """Normalize any EM pair file to columns id1,id2,label (bool).

    Headered files select columns by NAME: the label column is the one named
    'label' (case-insensitive), ids are the first two remaining columns --
    required because variant companion files carry 5 columns
    (id1,id2,source_1,source_2,label).

    ``skip_malformed`` drops headerless rows carrying no parseable label
    instead of raising. Gold stays strict by default.
    """
    if _has_header(path):
        frame = pd.read_csv(path, dtype=str)
        label_cols = [c for c in frame.columns if str(c).strip().lower() == "label"]
        if not label_cols:
            raise ValueError(f"headered EM pair file without a label column: {path}")
        id_cols = [c for c in frame.columns if c not in label_cols][:2]
        frame = frame[id_cols + label_cols[:1]]
        frame.columns = pd.Index(["id1", "id2", "label"])
    else:
        frame = _read_headerless_pairs(path, skip_malformed)
    frame["label"] = frame["label"].map(_parse_label)
    if frame[["id1", "id2"]].isna().any().any():
        raise ValueError(f"EM pair file with missing ids: {path}")
    return frame


VARIANT_GOLD_POLICIES = ("corner_filled", "baseline_pruned", "base_legacy")
# The canonical variant gold is corner_filled (the full variant gold including
# the injected corner cases).
DEFAULT_VARIANT_GOLD = "corner_filled"


def discover_em_test_files(
    task: TaskSpec, variant_gold: str | None = None
) -> dict[str, Path]:
    """Primary EM test files, keyed by pair name.

    Base tiers: the plain ``*_test.csv`` files; companions and a combined
    ``test_gt.csv`` are excluded.

    Variant tiers: the plain ``*_test.csv`` there is a copy of the base
    gold -- up to ~46% of its positives reference records that do not exist in
    the variant data. The variant gold ships as ``*_test_corner_filled.csv``
    (full variant gold including the injected corner cases; the canonical
    policy and the default) and ``*_test_baseline_pruned.csv`` (the subset
    comparable with the base task). 'base_legacy' selects these copies
    (explicitly, for comparisons only).
    """
    em_dir = task.root / "input" / "entitymatching"
    if not em_dir.is_dir():
        return {}
    out: dict[str, Path] = {}
    if task.tier == "base" or task.is_fixture:
        suffix = "_test"
        for path in sorted(em_dir.glob("*test*.csv")):
            name = path.stem
            if any(marker in name for marker in _COMPANION_MARKERS):
                continue
            if name == "test_gt" or not name.endswith(suffix):
                continue
            out[name.removesuffix(suffix)] = path
    else:
        variant_gold = variant_gold or DEFAULT_VARIANT_GOLD
        if variant_gold not in VARIANT_GOLD_POLICIES:
            raise ValueError(
                f"{task.task_id}: variant_gold must be one of "
                f"{VARIANT_GOLD_POLICIES}, got {variant_gold!r}"
            )
        suffix = "_test" if variant_gold == "base_legacy" else f"_test_{variant_gold}"
        for path in sorted(em_dir.glob("*test*.csv")):
            name = path.stem
            if not name.endswith(suffix):
                continue
            if variant_gold == "base_legacy" and any(
                marker in name for marker in _COMPANION_MARKERS
            ):
                continue
            out[name.removesuffix(suffix)] = path
    if not out:
        raise FileNotFoundError(
            f"no EM test files for {task.task_id} (variant_gold={variant_gold})"
        )
    return out


def load_em_test_pairs(
    task: TaskSpec, variant_gold: str | None = None
) -> dict[str, pd.DataFrame]:
    return {
        pair: load_em_pairs(path)
        for pair, path in discover_em_test_files(task, variant_gold).items()
    }
