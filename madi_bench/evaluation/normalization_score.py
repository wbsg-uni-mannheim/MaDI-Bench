"""Score a system's normalized source tables against a task's normalization
test set.

The normalization test set (``input/normalization/<split>.csv``) names source
cells: for one gold entity, the source record, the attribute (target-schema
name) and the ``expected_value`` -- the gold value a correct normalizer
recovers from that record's raw value. Categories: ``identity`` (the raw
value already is the gold value), ``normalization`` (a rule or a vocabulary
the task declares recovers it: units, dates, casing, aliases, the shipped
taxonomies) and ``knowledge`` (recoverable only with outside knowledge, e.g.
a well-known name's expansion). ``rule`` names the minimal repair.

Input: the system's normalized source tables, i.e. every source record with
its values already mapped to the target schema and normalized. Either

* a directory with one CSV per source (``<source>.csv`` or the source file's
  stem, e.g. ``metacritic.csv``, ``dataset_1.csv``) whose rows carry the
  record id in an ``id`` or ``record_id`` column and the target attributes
  as columns, or
* one CSV with ``source`` and ``record_id`` columns plus the attributes.

A table is found under the source's name as the test set uses it, under the
stem of the source file the gold schema mapping names for it
(``input/schemamatching/sm_mapping_gold.json``; ``products_1`` -> ``dataset_1``)
and, for products, under every spelling in circulation (``products_1``,
``dataset_1``, ``prod1``); ids are looked up with and without the
``<source>_`` prefix.

Scoring is whole-row: a cell is correct when the normalized value equals the
expected value exactly, or, for a list attribute (companies keypeople, games
genres, music tracks, papers authors), when the two lists are equal in any
serialization the submission format allows (JSON list, ``|`` or ``;``
separated string, one bare value); keypeople compares as a set. Numbers are
NOT forgiven ('1888' is not '1888.0'). Reported per category, per attribute
and per rule, with the coverage (rows whose record was found).
"""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .fusion_gold import domain_of, tier_of

LISTS = {"companies": {"keypeople"}, "games": {"genres"}, "music": {"tracks"},
         "papers": {"authors"}, "products": set()}
TRANSFORM = ("normalization", "knowledge")
# the products sources under every spelling in circulation: the gold's
# products_N, the base data files dataset_N, the LLM pipeline's prodN
_PRODUCTS_ALIASES = {f"{spelling}{n}": {f"products_{n}", f"dataset_{n}", f"prod{n}"}
                     for n in range(1, 5) for spelling in ("products_", "dataset_", "prod")}

csv.field_size_limit(10 ** 8)


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_rows(task_dir: Path, split: str = "test") -> list[dict]:
    path = Path(task_dir) / "input" / "normalization" / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no normalization test set for this task: {path}")
    return _read(path)


def source_stems(task_dir: Path) -> dict[str, set[str]]:
    """source name -> {stem of its source file} from the task's gold schema
    mapping (``sm_mapping_gold.json``, key ``source_files``); empty when the
    task does not ship one."""
    path = Path(task_dir) / "input" / "schemamatching" / "sm_mapping_gold.json"
    if not path.is_file():
        return {}
    files = json.loads(path.read_text(encoding="utf-8")).get("source_files", {}) or {}
    return {str(name): {Path(str(file)).stem} for name, file in files.items()}


# ----------------------------------------------------------- comparator

def as_list(value: str) -> list[str]:
    s = value.strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        for load in (json.loads, ast.literal_eval):
            try:
                data = load(s)
            except (ValueError, SyntaxError):
                continue
            if isinstance(data, (list, tuple)):
                return [str(x) for x in data]
    for sep in ("|", ";"):
        if sep in s:
            return [p.strip() for p in s.split(sep)]
    return [s]


def equal(domain: str, attribute: str, got: str, expected: str) -> bool:
    """The frozen ``list_form`` comparator."""
    if got == expected:
        return True
    if attribute in LISTS[domain]:
        a, b = as_list(got), as_list(expected)
        if domain == "companies":
            a, b = sorted(a), sorted(b)
        return a == b
    return False


# ---------------------------------------------------------------- tables

class NormalizedTables:
    """The system's normalized records keyed by (source, record id).
    ``aliases`` maps a source name of the test set to other names the same
    table may carry (the file stems from ``source_stems``)."""

    def __init__(self, path: Path, aliases: dict[str, set[str]] | None = None):
        path = Path(path)
        self.aliases: dict[str, set[str]] = {k: set(v) for k, v in (aliases or {}).items()}
        self.rows: dict[tuple[str, str], dict] = {}
        self.skipped: list[str] = []
        if path.is_dir():
            for file in sorted(path.glob("*.csv")):
                rows = _read(file)
                if rows and "record_id" not in rows[0] and "id" not in rows[0]:
                    self.skipped.append(file.name)      # not a source table (e.g. a mapping file)
                    continue
                for r in rows:
                    self.rows[(file.stem, str(r.get("record_id", r.get("id"))))] = r
            if not self.rows:
                raise ValueError(f"{path}: no CSV with an 'id' or 'record_id' column")
        else:
            for r in _read(path):
                if "source" not in r or "record_id" not in r:
                    raise ValueError(f"{path}: a single table needs 'source' and 'record_id' columns")
                self.rows[(str(r["source"]), str(r["record_id"]))] = r
        self.sources = {s for s, _ in self.rows}

    def lookup(self, source: str, record_id: str) -> dict | None:
        others = self.aliases.get(source, set()) | _PRODUCTS_ALIASES.get(source, set())
        names = [source] + sorted(others - {source})
        ids = [record_id]
        for name in names:
            if record_id.startswith(name + "_"):
                ids.append(record_id[len(name) + 1:])
        for name in names:
            for rid in ids:
                row = self.rows.get((name, rid))
                if row is not None:
                    return row
        return None


# ---------------------------------------------------------------- scoring

def score_normalization(task_dir: Path, tables: Path | NormalizedTables,
                        split: str = "test") -> dict[str, Any]:
    task_dir = Path(task_dir)
    domain, tier = domain_of(task_dir), tier_of(task_dir)
    stems = source_stems(task_dir)
    if isinstance(tables, NormalizedTables):
        for name, names in stems.items():
            tables.aliases.setdefault(name, set()).update(names)
    else:
        tables = NormalizedTables(tables, aliases=stems)
    rows = test_rows(task_dir, split)
    tally: dict[str, Counter] = defaultdict(Counter)
    details = []
    found = 0
    for r in rows:
        rec = tables.lookup(r["source"], r["source_id"])
        got = (rec or {}).get(r["attribute"], "") if rec is not None else ""
        got = "" if got is None else str(got)
        ok = equal(domain, r["attribute"], got, r["expected_value"])
        found += rec is not None
        details.append({"entity_id": r["entity_id"], "source": r["source"], "source_id": r["source_id"],
                        "attribute": r["attribute"], "category": r["category"], "rule": r["rule"],
                        "found": rec is not None, "correct": ok})
        for key in ("all", f"category:{r['category']}",
                    "category:transformations" if r["category"] in TRANSFORM else None,
                    f"attribute:{r['attribute']}", f"rule:{r['rule'] or 'identity'}"):
            if key:
                tally[key]["n"] += 1
                tally[key]["ok"] += int(ok)
    metrics = {k: {"n": c["n"], "correct": c["ok"], "accuracy": round(c["ok"] / c["n"], 4) if c["n"] else None}
               for k, c in sorted(tally.items())}
    return {"task": f"{domain}_{tier}", "split": split, "comparator": "list_form", "rows": len(rows),
            "skipped_files": tables.skipped,
            "record_coverage": round(found / len(rows), 4) if rows else None,
            "accuracy": metrics["all"]["accuracy"] if rows else None,
            "accuracy_transformations": (metrics.get("category:transformations") or {}).get("accuracy"),
            "metrics": metrics, "details": details}
