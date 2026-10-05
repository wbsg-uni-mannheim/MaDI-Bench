"""The record-id map of a variant task: generator id -> published (opaque) id.

The variant generator gives its synthetic records ids that reveal matches
(knob K2: one entity index shared across sources, e.g.
``k02_interp_games_00012__dbpedia`` and ``..._00012__metacritic``; knob K4: the
copied record inside the id, ``k04__within_dup__dbpedia__dbpedia_45``). The
variant tasks of this repository carry opaque ids in the source's normal id
format instead. The translation table has the columns
``source, original_id, opaque_id``; the scorers look for it at
``<task>/output/provenance/record_id_map.csv``.

The stored P2 / P3 variant outputs under ``results/`` carry the original
generator ids. ``rescore.py`` translates them through this map before scoring
when a map is available. The task folders hold no map (it would reveal which
variant records belong together); every function here returns an empty
mapping when the map file is absent, so callers can translate unconditionally.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

MAP_RELPATH = Path("output/provenance/record_id_map.csv")
MAP_COLUMNS = ("source", "original_id", "opaque_id")


class RecordIdMapError(ValueError):
    """The map file is malformed (duplicate or ambiguous ids)."""


def map_path(task_root: Path | str) -> Path:
    return Path(task_root) / MAP_RELPATH


def has_record_id_map(task_root: Path | str) -> bool:
    return map_path(task_root).is_file()


@lru_cache(maxsize=64)
def _rows(path_str: str, mtime_ns: int, size: int) -> tuple[tuple[str, str, str], ...]:
    with open(path_str, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if tuple(header or ()) != MAP_COLUMNS:
            raise RecordIdMapError(f"{path_str}: header {header!r}, expected {list(MAP_COLUMNS)}")
        rows = tuple((r[0], r[1], r[2]) for r in reader if r)
    originals = [r[1] for r in rows]
    opaques = [r[2] for r in rows]
    if len(set(originals)) != len(originals) or len(set(opaques)) != len(opaques):
        raise RecordIdMapError(f"{path_str}: an original or opaque id occurs twice")
    if set(originals) & set(opaques):
        raise RecordIdMapError(f"{path_str}: an id is both an original and an opaque id")
    return rows


def load_record_id_rows(task_root: Path | str) -> list[tuple[str, str, str]]:
    """[(source, original_id, opaque_id), ...]; [] when the task has no map."""
    path = map_path(task_root)
    if not path.is_file():
        return []
    st = path.stat()
    return list(_rows(str(path.resolve()), st.st_mtime_ns, st.st_size))


def load_record_id_map(task_root: Path | str, inverse: bool = False,
                       source: str | None = None) -> dict[str, str]:
    """original id -> opaque id (``inverse=True``: opaque -> original),
    optionally for one source (the data file stem, e.g. ``dbpedia``)."""
    rows = load_record_id_rows(task_root)
    if source is not None:
        rows = [r for r in rows if r[0] == source]
    if inverse:
        return {opaque: original for _, original, opaque in rows}
    return {original: opaque for _, original, opaque in rows}
