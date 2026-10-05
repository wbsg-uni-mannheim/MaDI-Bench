"""Variant loader for committee validation.

``load_variant(domain, level)`` returns a :class:`VariantBundle` that
packages everything a committee needs to score a single variant:

- source DataFrames with ``attrs["dataset_name"]`` preserved
- schema-matching target schema plus the K8 mapping (when present)
- per-source-pair EM gold splits
- fusion gold and fusion validation DataFrames
- pooled positives (when the domain has a pool)

``level == "baseline"`` loads the *original* ``use cases/<domain>/base/``
directory; every other level loads
``use cases/<domain>/<level>/``. The baseline case lets
``measure_baseline.py`` reuse the same committee runners as
``validate_variant.py`` — one code path, two inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from PyDI.io import load_csv, load_json, load_xml

from .domain_config import (
    POOLS_DIR,
    USECASES_DIR,
    VALID_LEVELS,
    DomainConfig,
    data_root_for_domain,
    load_domain_config,
    task_dir,
    variant_dir,
)
from .fusion_gold_keys import normalize_source_ids_column
from .loaders import (
    em_gold_candidates,
    load_source,
    read_em_gold_pair,
)

VALID_BUNDLE_LEVELS: list[str] = ["baseline", *VALID_LEVELS]

EM_SPLITS: tuple[str, ...] = ("train", "val", "test", "all")


@dataclass
class VariantBundle:
    """Everything a committee needs to score a single variant.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).
    level : str
        One of ``"baseline"``, ``"easy"``, ``"medium"``, ``"hard"``.
    sources : dict[str, DataFrame]
        Source DataFrames keyed by source name, with
        ``attrs["dataset_name"]`` set.
    target_schema : dict
        Parsed ``target_schema.json``.
    sm_mapping : DataFrame or None
        Schema mapping: the task's SM gold (``sm_mapping_gold``) for
        ``level == "baseline"``, the Knob-8 generated mapping
        (``sm_mapping``) for a variant. ``None`` when the file is missing.
    em_gold : dict[tuple[str, str], DataFrame]
        Per-source-pair test gold, keyed by ``(src1, src2)`` ordered as
        declared in the domain config. Each frame has columns ``id1``,
        ``id2``, ``label``.
    em_gold_regenerated : dict[tuple[str, str], dict[str, dict[str, DataFrame]]]
        Per-source-pair regenerated EM gold, split by train / val /
        test and then by version (``baseline_pruned`` /
        ``corner_filled``). Loaded from
        ``input/entitymatching/<pair>_{train,val,test}_{baseline_pruned,corner_filled}.csv``
        (Knob 2 output). Outer key follows the
        same ``(src1, src2)`` convention as ``em_gold``; inner key is
        the split name; innermost key is the version. Empty dict when
        the files are missing (baseline variants; K2 not yet applied).
        Frames have columns ``id1``, ``id2``, ``label`` — source
        columns are dropped after loading. The **val** split is the
        primary variant-specific F1 surface; the **test** split is
        scored under both versions by the EM committee (4d);
        **train** is emitted for downstream public benchmark consumers.
    em_splits : dict[tuple[str, str], dict[str, DataFrame]]
        Per-source-pair split map. The inner dict has keys ``"train"``,
        ``"val"``, ``"test"``, ``"all"``; missing splits are omitted.
    fusion_gold : DataFrame
        Parsed fusion test set (``test_set.xml`` in a variant), used as
        the fusion gold standard.
    fusion_validation : DataFrame or None
        Parsed fusion validation set (``validation_set.xml`` in a
        variant) if present.
    pooled_positives : DataFrame or None
        Pooled positives from ``pools/<domain>/pooled_positives.csv``
        when present. Used as the protection set / pool diagnostic.
    variant_root : Path
        Root directory this bundle was loaded from.
    sm_task_root : Path or None
        The task directory the schema-matching stage reads as shipped
        (:mod:`usecases_synthetic.lib.sm_view`): raw source column names
        scored against the shipped SM gold, instead of this bundle's
        renamed ``sources`` / reconciled ``sm_mapping`` (which the other
        stages keep using). ``None`` for hand-built bundles, whose SM stage
        falls back to the bundle's own frames.
    """

    domain: str
    level: str
    sources: dict[str, pd.DataFrame]
    target_schema: dict[str, Any]
    sm_mapping: pd.DataFrame | None
    em_gold: dict[tuple[str, str], pd.DataFrame]
    em_splits: dict[tuple[str, str], dict[str, pd.DataFrame]]
    fusion_gold: pd.DataFrame
    fusion_validation: pd.DataFrame | None
    pooled_positives: pd.DataFrame | None
    variant_root: Path
    em_gold_regenerated: dict[tuple[str, str], dict[str, dict[str, pd.DataFrame]]] = (
        field(default_factory=dict)
    )
    knob_08_renames: dict[str, dict[str, str]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    sm_task_root: Path | None = None

    @property
    def source_pairs(self) -> list[tuple[str, str]]:
        """Return the ordered source pairs this bundle contains EM gold for."""
        return list(self.em_gold.keys())

    def resolve_column_mapping(
        self,
        static_mapping: dict[str, dict[str, str]],
    ) -> dict[str, dict[str, str]]:
        """Translate a committee's static column_mapping through K8 renames.

        The static column_mapping in ``em_blocking_committee.yaml`` /
        ``em_matching_committee.yaml`` / ``fusion_committee.yaml`` is keyed
        on *original* source column
        names (pre-K8). Two translations are applied so the resulting
        mapping renames the columns that actually exist in the variant:

        1. For every static entry ``{orig: canonical}``, the key is
           rewritten to the post-K8 column name (when K8 renamed it).
        2. For every K8 rename ``orig -> k8_col`` on a column *not* in
           the static mapping, we add ``{k8_col: orig}`` so the column
           is restored to its pre-K8 (baseline, canonical) name.

        Together these guarantee every column the variant's DataFrame
        carries is either left alone or renamed back to the name the
        downstream committee code expects.

        Columns not touched by K8 pass through unchanged. Sources
        absent from ``knob_08_renames`` pass through identity.

        Parameters
        ----------
        static_mapping : dict
            ``{source: {orig_col: canonical_col}}`` from the committee
            roster YAML.

        Returns
        -------
        dict
            ``{source: {post_k8_col: canonical_col}}`` suitable for
            renaming the variant's source DataFrames.
        """
        if not self.knob_08_renames:
            return {src: dict(m) for src, m in static_mapping.items()}

        resolved: dict[str, dict[str, str]] = {}
        all_sources = set(static_mapping) | set(self.knob_08_renames)
        for src in all_sources:
            static = static_mapping.get(src, {})
            k8 = self.knob_08_renames.get(src, {})

            result: dict[str, str] = {}
            # K8-only renames: restore post-K8 name to its pre-K8 name.
            for orig, k8_col in k8.items():
                if orig not in static:
                    result[k8_col] = orig
            # Static entries: rewrite key through K8 (identity if unrenamed).
            for orig, canonical in static.items():
                result[k8.get(orig, orig)] = canonical

            resolved[src] = result
        return resolved


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def variant_root(domain: str, level: str) -> Path:
    """Return the root directory for ``(domain, level)``.

    Parameters
    ----------
    domain : str
        Domain name.
    level : str
        ``"baseline"`` maps to ``use cases/<domain>/base/``; any other level
        maps to ``use cases/<domain>/<level>/`` (see
        :func:`~usecases_synthetic.lib.domain_config.task_dir`).

    Returns
    -------
    Path
        Variant root directory.
    """
    if level == "baseline":
        # Original-input bundle honors the ``data_root`` override for
        # the input domain (e.g. products → usecases_synthetic/usecases).
        root = data_root_for_domain(domain) or USECASES_DIR
        return task_dir(domain, root=root)
    # Augmented variants always live under the top-level USECASES_DIR
    # (``use cases/<domain>/<level>/``) for cross-domain consistency; the
    # per-domain ``data_root`` does NOT apply here.
    return variant_dir(domain, level, root=USECASES_DIR)


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def _load_baseline_sources(config: DomainConfig) -> dict[str, pd.DataFrame]:
    """Load original-format sources for the baseline variant."""
    sources: dict[str, pd.DataFrame] = {}
    for spec in config.sources:
        df = load_source(
            domain=config.domain,
            source_name=spec.name,
            source_file=spec.file,
            source_format=spec.format,
            reader_kwargs=spec.reader_kwargs,
            inject_id=spec.inject_id,
            id_column=spec.id_column,
        )
        df.attrs["dataset_name"] = spec.name
        sources[spec.name] = df
    return sources


def _load_augmented_sources(
    config: DomainConfig,
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load CSV-serialised sources from a packaged variant directory.

    The packager writes every source as ``<source>.csv`` regardless of
    original format (XML / JSON / TSV all collapse to CSV). See
    :func:`usecases_synthetic.scripts.package_variant.write_sources_as_csv`.

    Applies the same post-load normalization as
    :func:`usecases_synthetic.lib.loaders.load_source` (music ``tracks``
    list-literal parse, games ``genres`` comma-split, discogs
    ``duration`` 0-sentinel coalesce) so list-aware fusion strategies
    + Jaccard eval consume identical in-memory shapes on baseline and
    variant runs. ``DataFrame.to_csv`` collapses Python lists to their
    string repr, so without this step the variant's tracks/genres
    columns would come back as strings and break list-shape eval.
    """
    from .loaders import normalize_loaded_source

    sources: dict[str, pd.DataFrame] = {}
    for spec in config.sources:
        csv_path = data_dir / f"{spec.name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Variant source missing: {csv_path}")
        df = load_csv(csv_path, name=spec.name)
        df = normalize_loaded_source(df, domain=config.domain, source_name=spec.name)
        df.attrs["dataset_name"] = spec.name
        sources[spec.name] = df
    return sources


# ---------------------------------------------------------------------------
# Schema matching
# ---------------------------------------------------------------------------


def _load_target_schema(sm_dir: Path) -> dict[str, Any]:
    """Parse ``target_schema.json`` from a schema-matching directory."""
    path = sm_dir / "target_schema.json"
    if not path.exists():
        raise FileNotFoundError(f"Target schema missing: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_SM_GOLD_COLUMNS = [
    "source_dataset",
    "source_column",
    "target_dataset",
    "target_column",
    "score",
]


def _sm_mapping_from_json(path: Path) -> pd.DataFrame:
    """Parse a ``pydi_schema_mapping_gold`` JSON file into the flat mapping
    frame the SM committee expects.

    The committed ``sm_mapping_gold.json`` (``kind: pydi_schema_mapping_gold``)
    supersedes the legacy ``sm_mapping_gold.csv``. Only positive (``label``
    truthy) mappings are returned, projected to the legacy ``source_dataset,
    source_column, target_dataset, target_column, score`` columns so every
    downstream consumer is unchanged.

    NOTE: the JSON carries *raw* on-disk source column names; callers that
    score against the loaded (renamed) frames must reconcile via
    :func:`_reconcile_sm_gold_source_columns`. The SM stage itself reads the
    raw sources (:mod:`usecases_synthetic.lib.sm_view`) and uses this gold
    unreconciled.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    rows = [
        {
            "source_dataset": m["source_dataset"],
            "source_column": m["source_column"],
            "target_dataset": m["target_dataset"],
            "target_column": m["target_column"],
            "score": m.get("score", 1.0),
        }
        for m in payload.get("mappings", [])
        if m.get("label", True)
    ]
    return pd.DataFrame(rows, columns=_SM_GOLD_COLUMNS)


def _load_sm_mapping(sm_dir: Path, *, baseline: bool = False) -> pd.DataFrame | None:
    """Load schema-matching gold mapping if present.

    The committed ``sm_mapping_gold.json`` (``kind: pydi_schema_mapping_gold``)
    is preferred over the legacy ``sm_mapping_gold.csv`` when present. The CSV
    is read as a fallback for unmigrated domains.

    Parameters
    ----------
    sm_dir : Path
        Schema-matching directory.
    baseline : bool
        When ``True``, load the hand-authored baseline gold mapping
        (``sm_mapping_gold.{json,csv}``). When ``False``, load the knob-8
        generated variant mapping (``sm_mapping.{json,csv}``).

    Returns
    -------
    DataFrame or None
        Gold mapping, or ``None`` if neither file exists.
    """
    stem = "sm_mapping_gold" if baseline else "sm_mapping"
    json_path = sm_dir / f"{stem}.json"
    if json_path.exists():
        return _sm_mapping_from_json(json_path)
    csv_path = sm_dir / f"{stem}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def _reconcile_sm_gold_source_columns(
    sm_mapping: pd.DataFrame | None,
    config: "DomainConfig",
    domain: str,
) -> pd.DataFrame | None:
    """Map gold ``source_column`` names onto the loaded (renamed) frames.

    The committed SM gold (JSON or CSV) is authored against *raw*
    on-disk column names, but the loader renames columns before any committee
    sees them: each source's configured ``id_column`` becomes ``id``, and
    papers maps every raw column to its canonical name
    (``loaders._PAPERS_SOURCE_COLUMN_MAP``). Applying the same renames to the
    gold's ``source_column`` makes the bundle consumers of ``sm_mapping``
    (normalization resolution, the pipeline's e2e panel) land on columns that
    actually exist post-load. Schema-matching scoring no longer uses this
    reconciled gold: it reads the raw sources and the raw gold
    through :mod:`usecases_synthetic.lib.sm_view`, so the renames cannot hand
    the matchers free correspondences. Rows whose source column is already a
    loaded name pass through unchanged. The music and products golds name the
    sources' raw (native) columns; only their id columns differ from the loaded
    names (music ``Attribute_1`` / ``rec_uid`` / ``item_code``), and those are
    reconciled through the configured ``id_column``.
    """
    if sm_mapping is None or sm_mapping.empty:
        return sm_mapping

    from .domain_config import _resolve_knob_config_alias
    from .loaders import _PAPERS_SOURCE_COLUMN_MAP

    canonical_domain = _resolve_knob_config_alias(domain) or domain
    rename_by_source: dict[str, dict[str, str]] = {}
    for src in config.sources:
        rmap: dict[str, str] = {}
        if src.id_column and src.id_column != "id":
            rmap[src.id_column] = "id"
        if canonical_domain == "papers":
            rmap.update(_PAPERS_SOURCE_COLUMN_MAP.get(src.name, {}))
        rename_by_source[src.name] = rmap

    out = sm_mapping.copy()
    out["source_column"] = [
        rename_by_source.get(str(ds), {}).get(str(sc), sc)
        for ds, sc in zip(out["source_dataset"], out["source_column"])
    ]
    return out


# ---------------------------------------------------------------------------
# Entity matching
# ---------------------------------------------------------------------------


def _em_file(em_dir: Path, pair: tuple[str, str], split: str) -> Path:
    """Return the EM CSV path for ``<src1>_2_<src2>_<split>.csv``."""
    src1, src2 = pair
    return em_dir / f"{src1}_2_{src2}_{split}.csv"


def _load_em_gold(
    em_dir: Path,
    source_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], pd.DataFrame]:
    """Load EM gold for each declared source pair.

    Prefers the ``_all`` file (full gold standard) when available;
    falls back to ``_test`` when it doesn't exist.

    Pair direction is matched modulo orientation
    (``<src1>_2_<src2>_<split>.csv`` OR
    ``<src2>_2_<src1>_<split>.csv``). When loaded from the reverse
    direction, ``id1`` / ``id2`` are swapped so that the returned frame
    always carries ``id1`` belonging to ``src1`` and ``id2`` belonging
    to ``src2``. Downstream consumers (``committee_em.py``, the
    matcher inputs) look up ``id1`` in ``df_left = sources[src1]`` and
    ``id2`` in ``df_right = sources[src2]``; without the swap, every
    lookup fails on reverse-direction gold (regression observed
    on games ``metacritic_dbpedia`` after the initial direction-tolerance
    fix landed without the swap — magellan crashed, Ditto returned 0
    predictions).

    Parameters
    ----------
    em_dir : Path
        Entity-matching directory.
    source_pairs : list of tuple
        Source pairs to load. Pairs without a gold CSV are skipped.

    Returns
    -------
    dict
        ``{(src1, src2): DataFrame}`` for every pair with a gold CSV.
        Each frame has columns ``id1`` (src1's ids), ``id2`` (src2's
        ids), ``label``.
    """
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for pair in source_pairs:
        for split in ("all", "test"):
            match = next(
                (
                    (path, swap)
                    for path, swap in em_gold_candidates(em_dir, pair, split)
                    if path.exists()
                ),
                None,
            )
            if match is not None:
                # read_em_gold_pair swaps id1<->id2 for reverse-direction
                # files so id1 always belongs to the pair's src1.
                out[pair] = read_em_gold_pair(*match)
                break
    return out


_REGEN_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
_REGEN_VERSION_NAMES: tuple[str, ...] = ("baseline_pruned", "corner_filled")


def _load_em_gold_regenerated(
    em_dir: Path,
    source_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, dict[str, pd.DataFrame]]]:
    """Load Knob-2 regenerated EM gold per source pair per split per version.

    Looks for files at
    ``input/entitymatching/<src1>_2_<src2>_<split>_<version>.csv`` for
    each authored pair × each split in ``train / val / test`` × each
    version in ``baseline_pruned / corner_filled``.
    Each file carries ``id1, id2, source_1, source_2, label``;
    source columns are dropped after loading.

    Pair direction is matched modulo orientation (``src1_2_src2`` OR
    ``src2_2_src1``) so the loader tolerates either ordering on disk.

    Legacy ``*_regenerated.csv`` files are no longer recognised — the
    next variant regen overwrites with the new naming, and the
    legacy artifacts are not preserved.

    Parameters
    ----------
    em_dir : Path
        Entity-matching directory.
    source_pairs : list of tuple
        Source pairs declared by the domain config (canonical ordering).

    Returns
    -------
    dict
        ``{(src1, src2): {split: {version: DataFrame}}}`` with columns
        ``id1``, ``id2``, ``label``. Empty outer dict when no regen
        files exist (baseline variants; K2 not yet applied). Inner
        dicts omit splits / versions whose file is missing.
    """
    out: dict[tuple[str, str], dict[str, dict[str, pd.DataFrame]]] = {}
    for pair in source_pairs:
        src1, src2 = pair
        per_split: dict[str, dict[str, pd.DataFrame]] = {}
        for split in _REGEN_SPLIT_NAMES:
            per_version: dict[str, pd.DataFrame] = {}
            for version in _REGEN_VERSION_NAMES:
                candidates = [
                    em_dir / f"{src1}_2_{src2}_{split}_{version}.csv",
                    em_dir / f"{src2}_2_{src1}_{split}_{version}.csv",
                ]
                path = next((p for p in candidates if p.exists()), None)
                if path is None:
                    continue
                df = pd.read_csv(path)
                if df.empty:
                    continue
                required = {"id1", "id2", "label"}
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(
                        f"{path} missing required columns {sorted(missing)}; "
                        "regenerate with the current apply_knob_02_niche.py."
                    )
                per_version[version] = (
                    df[["id1", "id2", "label"]].reset_index(drop=True).copy()
                )
            if per_version:
                per_split[split] = per_version
        if per_split:
            out[pair] = per_split
    return out


def _load_em_splits(
    em_dir: Path,
    source_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, pd.DataFrame]]:
    """Load all EM splits for each source pair.

    Parameters
    ----------
    em_dir : Path
        Entity-matching directory.
    source_pairs : list of tuple
        Source pairs to load.

    Returns
    -------
    dict
        ``{pair: {split: DataFrame}}``. Missing split files are omitted.
    """
    out: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for pair in source_pairs:
        per_split: dict[str, pd.DataFrame] = {}
        for split in EM_SPLITS:
            match = next(
                (
                    (path, swap)
                    for path, swap in em_gold_candidates(em_dir, pair, split)
                    if path.exists()
                ),
                None,
            )
            if match is not None:
                per_split[split] = read_em_gold_pair(*match)
        if per_split:
            out[pair] = per_split
    return out


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def _load_fusion_xml(path: Path, name: str) -> pd.DataFrame:
    """Load a fusion XML file via PyDI with the standard flags."""
    df = load_xml(path, name=name, nested_handling="aggregate")
    df.attrs["dataset_name"] = name
    return df


def _load_fusion_file(path: Path, name: str, domain: str | None = None) -> pd.DataFrame:
    """Load a fusion gold file, dispatching on file content (with extension
    as fallback).

    ``.xml`` (every pre-2026 domain) loads through PyDI's XML reader with
    the aggregate flag, exactly as before. The 2026 papers domain ships
    its fusion gold as JSON-lines (``fusion_test.jsonl`` /
    ``fusion_val.jsonl``: one flat fused record per line with a
    ``source_ids`` list of its member record ids, no ``id`` and no DOI),
    read with ``lines=True``; a plain ``.json`` array is also accepted.
    Non-XML gold carries no per-attribute ``provenance``, so
    source-attribution fusion metrics are unavailable for such domains --
    value-correctness by the configured ``gold_id_column`` join still
    works.

    A ``source_ids`` column (papers at every tier, the products task
    variants' ``<source_ids>`` element, the generator's products copy) is
    normalised to the comma-joined string the fusion evaluator splits,
    the anchor first (:func:`fusion_gold_keys.join_source_ids`); a
    JSON list would otherwise reach the evaluator as its ``str()`` repr
    and align no gold row.

    Content sniffing: the papers variants hold JSON lines under the
    canonical ``test_set.xml`` / ``validation_set.xml`` filenames, so
    the first non-whitespace byte picks the reader, not the extension:
    ``{`` -> JSONL, ``[`` -> JSON array, ``<`` -> XML; anything else
    falls back to the extension.
    """
    first_byte = ""
    try:
        with path.open("rb") as fh:
            chunk = fh.read(4096)
        for b in chunk:
            ch = chr(b)
            if not ch.isspace():
                first_byte = ch
                break
    except OSError:
        first_byte = ""

    if first_byte == "{":
        df = load_json(path, name=name, lines=True)
    elif first_byte == "[":
        df = load_json(path, name=name)
    elif first_byte == "<":
        df = _load_fusion_xml(path, name)
    else:
        # Empty or unrecognised: fall back to the extension dispatcher so
        # the original error surface (e.g. XMLSyntaxError) is preserved.
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            df = load_json(path, name=name, lines=True)
        elif suffix == ".json":
            df = load_json(path, name=name)
        else:
            df = _load_fusion_xml(path, name)
    # source_ids-keyed gold: one evaluator-ready string per record (no-op
    # for id-keyed gold, which has no source_ids column).
    df = normalize_source_ids_column(df, domain)
    df.attrs["dataset_name"] = name
    return df


def _load_fusion(
    fusion_dir: Path,
    test_filename: str = "test_set.xml",
    validation_filename: str = "validation_set.xml",
    domain: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load the fusion test (gold) + validation (optional) files.

    ``test_filename`` / ``validation_filename`` default to the canonical
    variant-directory names. ``load_variant`` overrides them with the
    domain config's ``fusion_files`` block when ``level == "baseline"``
    so the source-side ``*_set_final.xml`` for games + music (or the
    papers ``fusion_{test,val}.jsonl``) is picked up; packaged variants
    always carry the canonical names per ``package_variant.copy_fusion``.
    Format is chosen per file content (see :func:`_load_fusion_file`);
    ``domain`` picks the anchor order of ``source_ids`` members
    (inferred from the id prefixes when omitted).
    """
    test_path = fusion_dir / test_filename
    if not test_path.exists():
        raise FileNotFoundError(f"Fusion gold missing: {test_path}")
    gold = _load_fusion_file(test_path, "fusion_test_set", domain)

    val_path = fusion_dir / validation_filename
    val: pd.DataFrame | None = None
    if val_path.exists():
        val = _load_fusion_file(val_path, "fusion_validation_set", domain)
    return gold, val


# ---------------------------------------------------------------------------
# Pooled positives
# ---------------------------------------------------------------------------


def _load_pooled_positives(domain: str) -> pd.DataFrame | None:
    """Load ``pools/<domain>/pooled_positives.csv`` if present."""
    path = POOLS_DIR / domain / "pooled_positives.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Knob-08 renames
# ---------------------------------------------------------------------------


def _load_knob_08_renames(root: Path) -> dict[str, dict[str, str]]:
    """Load per-source column renames from the K8 provenance CSV.

    The K8 apply script writes ``output/provenance/knob_08_naming.csv``
    with one row per non-identity column rename, using columns
    ``source``, ``attribute`` (pre-K8 column name) and ``new_value``
    (post-K8 column name).

    Parameters
    ----------
    root : Path
        Variant root directory.

    Returns
    -------
    dict
        ``{source: {pre_k8_col: post_k8_col}}``. Empty dict when the
        provenance CSV is absent (baseline variants or identity rung).
    """
    path = root / "output" / "provenance" / "knob_08_naming.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}

    renames: dict[str, dict[str, str]] = {}
    for row in df.itertuples(index=False):
        src = str(row.source)
        orig = str(row.attribute)
        new = str(row.new_value)
        renames.setdefault(src, {})[orig] = new
    return renames


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def _sm_task_root(
    domain: str,
    level: str,
    root: Path,
    root_override: Path | None,
) -> Path:
    """The task directory the SM stage reads as shipped.

    The base task resolves to the public task (``use cases/<domain>/base``)
    when it exists, even when the bundle itself honors a ``data_root``
    override: products' generator copy (``usecases_synthetic/usecases/
    products``) keeps serving the other committee stages, while schema
    matching reads the released ``dataset_<n>.json`` files and their SM gold.
    Legacy domains without a public base task (``-small``) and every
    variant / ``root_override`` bundle use the bundle's own root.
    """
    if root_override is not None or level != "baseline":
        return root
    public = task_dir(domain, "base", root=USECASES_DIR)
    if (public / "input" / "schemamatching").is_dir():
        return public
    return root


def load_variant(
    domain: str,
    level: str = "baseline",
    *,
    root_override: Path | None = None,
) -> VariantBundle:
    """Load a variant bundle for ``(domain, level)``.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).
    level : str, optional
        One of ``"baseline"``, ``"easy"``, ``"medium"``, ``"hard"``.
        Default ``"baseline"`` (original use case directory).
    root_override : Path, optional
        Override the variant root directory entirely. Used by tests to
        point at a fixture directory. When set, ``level`` still controls
        which source-loading strategy is used (original formats for
        baseline, CSV for augmented).

    Returns
    -------
    VariantBundle
        Everything the committees need.
    """
    if level not in VALID_BUNDLE_LEVELS:
        raise ValueError(f"Invalid level: {level!r}. Valid: {VALID_BUNDLE_LEVELS}")

    config = load_domain_config(domain)
    root = root_override if root_override is not None else variant_root(domain, level)

    input_dir = root / "input"
    data_dir = input_dir / "data"
    sm_dir = input_dir / "schemamatching"
    em_dir = input_dir / "entitymatching"
    fusion_dir = input_dir / "fusion"

    if level == "baseline" and root_override is None:
        sources = _load_baseline_sources(config)
    else:
        sources = _load_augmented_sources(config, data_dir)

    target_schema = _load_target_schema(sm_dir)
    sm_mapping = _load_sm_mapping(sm_dir, baseline=(level == "baseline"))
    if level == "baseline":
        # The baseline gold (JSON or CSV) is authored against raw on-disk
        # column names; reconcile them onto the loaded (renamed) frames so
        # normalization resolution hits columns that exist post-load. (SM
        # scoring reads the raw task via sm_task_root, see sm_view.)
        sm_mapping = _reconcile_sm_gold_source_columns(sm_mapping, config, domain)

    em_gold = _load_em_gold(em_dir, config.source_pairs)
    em_gold_regenerated = _load_em_gold_regenerated(em_dir, config.source_pairs)
    em_splits = _load_em_splits(em_dir, config.source_pairs)

    if level == "baseline" and root_override is None:
        fusion_gold, fusion_validation = _load_fusion(
            fusion_dir,
            test_filename=config.fusion_files["test"],
            validation_filename=config.fusion_files["validation"],
            domain=domain,
        )
    else:
        fusion_gold, fusion_validation = _load_fusion(fusion_dir, domain=domain)

    pooled = _load_pooled_positives(domain)

    knob_08_renames = _load_knob_08_renames(root)

    return VariantBundle(
        sm_task_root=_sm_task_root(domain, level, root, root_override),
        domain=domain,
        level=level,
        sources=sources,
        target_schema=target_schema,
        sm_mapping=sm_mapping,
        em_gold=em_gold,
        em_gold_regenerated=em_gold_regenerated,
        em_splits=em_splits,
        fusion_gold=fusion_gold,
        fusion_validation=fusion_validation,
        pooled_positives=pooled,
        variant_root=root,
        knob_08_renames=knob_08_renames,
    )
