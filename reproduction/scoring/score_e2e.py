"""End-to-end metric panel (paper Table 8): reference-free, silver-reference
and ground-truth metrics of a fused output, computed with PyDI's
``compute_e2e_panel`` at the pinned PyDI commit.

The computation is the one the paper's panels ran.

Three blocks:

* reference-free (RF): properties of the fused output alone (record gain over
  the largest input, density gain, output density, fusion ratio, schema
  validity);
* ground truth (GT): the output against the task's fusion test gold. The gold
  becomes a PyDI ``SilverStandard``: for the XML tasks via PyDI's
  ``load_workflow_silver`` (with the list-column repair of
  ``panel_gold._repair_nested_gold_columns``), for papers (every tier) and
  products base from the gold's ``source_ids`` lists (``_source_ids_silver``);
* silver reference (SR): the output against the human-engineered pipeline's
  (P1's) fused output, ``use cases/<domain>/base/output/data_fusion/fused.csv``
  (or its compressed copy ``fused.csv.gz``), see ``load_p1_silver`` and
  ``score_e2e_silver``.

The GT fusion-accuracy row of the paper is NOT taken from the panel: it is
the all-gold accuracy of ``madi_bench.evaluation.score_fusion`` (as in
``results/scores_v2.json``).

Panel configuration (column types, numeric tolerances, composite weights,
record-id prefix -> source map) comes from ``panel_configs/<domain>.yaml``.
Written panel folders (``out_dir``) contain per-cluster tables derived from
the gold (``*_gold.csv``); do not commit them.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from . import aliases
from . import gold as gold_mod
from .panel_gold import (
    ANCHOR_ID,
    DOMAIN_ANCHOR_PREFIX,
    _parse_id_list,
    _repair_nested_gold_columns,
    anchors_by_cluster,
    load_fusion_gold,
    papers_gold_path,
)
from .tasks import TaskSpec

PANEL_CONFIG_DIR = Path(__file__).resolve().parent / "panel_configs"
# PyDI's panel returns ~0 for the conflict metrics (a raw-vs-canonical
# column-name mismatch); they are not part of the paper's set.
KNOWN_BUG_FIELDS = ("conflict_rate", "conflict_only_accuracy")


@dataclass(frozen=True)
class PanelDomainConfig:
    domain: str
    column_types: dict
    tolerance_default: float
    tolerance_overrides: dict
    composite_weights: dict | None
    source_prefix_map: dict | None


def load_panel_config(domain: str) -> PanelDomainConfig:
    path = PANEL_CONFIG_DIR / f"{domain}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no panel config for domain {domain!r} at {path}")
    raw = yaml.safe_load(path.read_text())
    tolerance = raw.get("panel_tolerance") or {}
    return PanelDomainConfig(
        domain=domain,
        column_types=raw.get("column_types") or {},
        tolerance_default=float(tolerance.get("default", 0.05)),
        tolerance_overrides=tolerance.get("overrides") or {},
        composite_weights=raw.get("composite_weights") or None,
        source_prefix_map=raw.get("source_prefix_map") or None,
    )


def load_sources(task: TaskSpec) -> list[pd.DataFrame]:
    """Source tables via PyDI loaders, renamed to the target attribute names
    through the gold schema mapping (the conflict detector reads source
    values by target column name)."""
    from PyDI.io.loaders import load_csv, load_json

    sm_gold = gold_mod.load_sm_gold(task)
    frames = []
    for path in sorted((task.root / "input" / "data").iterdir()):
        if not path.is_file() or path.name.endswith("_metadata.json"):
            continue
        name = path.stem
        if path.suffix == ".csv":
            frame = load_csv(str(path), name=name)
        elif path.suffix == ".jsonl":
            frame = load_json(str(path), name=name, lines=True)
        elif path.suffix == ".json":
            frame = load_json(str(path), name=name)
        else:
            continue
        canonical_name = aliases.canonical_source_name(task, name)
        rename = {
            str(row["source_column"]): str(row["target_column"])
            for _, row in sm_gold.iterrows()
            if aliases.canonical_source_name(task, str(row["source_dataset"]))
            == canonical_name
            and row["source_column"] != row["target_column"]
        }
        if rename:
            frame = frame.rename(columns=rename)
        frame.attrs["dataset_name"] = name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no source tables under {task.root}/input/data")
    return frames


def load_panel_sources(task: TaskSpec) -> list[pd.DataFrame]:
    """load_sources with the id column in the panel's currency (products
    base: products_<N>_-prefixed ids, like the gold; no-op elsewhere)."""
    out = []
    for frame in load_sources(task):
        name = frame.attrs["dataset_name"]
        frame = aliases.normalize_source_ids(task, frame, name)
        frame.attrs["dataset_name"] = name
        out.append(frame)
    return out


def load_panel_gold(task: TaskSpec, prefix_map: dict | None):
    """The ground-truth reference (a PyDI SilverStandard) for the panel."""
    from PyDI.evaluation.silver_standard import load_workflow_silver

    if task.domain == "papers" or (
        task.domain == "products" and task.tier == "base"
    ):
        return _source_ids_silver(task, prefix_map)
    if task.domain == "products":
        # products VARIANT XML records carry <source_ids> instead of <id>
        return load_workflow_silver(
            task.root, domain=task.domain, prefix_map=prefix_map,
            id_tag="source_ids",
        )
    silver = load_workflow_silver(
        task.root, domain=task.domain, prefix_map=prefix_map
    )
    return _repair_silver_nested_columns(silver, task)


def _repair_silver_nested_columns(silver, task: TaskSpec):
    """Apply the nested-list repair (``panel_gold._repair_nested_gold_columns``)
    to the panel gold: list attributes nested in the XML (games genres,
    companies keypeople) would otherwise arrive empty."""
    prefix = DOMAIN_ANCHOR_PREFIX.get(task.domain)
    fused, membership = silver.fused, silver.membership
    if prefix is None or fused is None or membership is None:
        return silver
    if "cluster_id" not in fused.columns:
        return silver
    probe = fused.copy()
    probe["_repair_anchor"] = probe["cluster_id"].map(anchors_by_cluster(membership, task.domain))
    repaired = _repair_nested_gold_columns(probe, task, "_repair_anchor")
    if repaired is probe:
        return silver          # nothing was repairable
    return dataclasses.replace(
        silver, fused=repaired.drop(columns=["_repair_anchor"])
    )


def _source_ids_silver(task: TaskSpec, prefix_map: dict | None):
    """SilverStandard built from the fusion gold's per-entity source_ids --
    products base (fusion_test_set.csv) and papers, every tier
    (fusion_test.jsonl / the JSON lines in the variants' test_set.xml).

    cluster_id = the record's anchor id; membership derived from source_ids
    via the prefix map; cell_provenance is None. For papers the cell values
    are the file's native JSON types (read_json dtype=False): authors stay a
    list, the year an int and the volume a string.
    """
    from PyDI.evaluation.silver_standard import SilverStandard

    gold = load_fusion_gold(task)
    fused = gold.rename(columns={ANCHOR_ID: "cluster_id"})
    if task.domain == "papers":
        # load_fusion_gold keeps read_json's index through prep and dropna
        raw = pd.read_json(
            papers_gold_path(task), lines=True, dtype=False,
        ).loc[gold.index]
        fused = raw.assign(cluster_id=gold[ANCHOR_ID])
    rows = []
    prefix_map = prefix_map or {}

    for _, row in gold.iterrows():
        for record_id in _parse_id_list(row["source_ids"]):
            source = next(
                (name for prefix, name in prefix_map.items()
                 if record_id.startswith(prefix)),
                None,
            )
            rows.append({
                "record_id": record_id,
                "source": source,
                "cluster_id": row[ANCHOR_ID],
            })
    membership = pd.DataFrame(rows, columns=["record_id", "source", "cluster_id"])
    unresolved = membership["source"].isna()
    if unresolved.any():
        # Records whose ids carry no source prefix are resolved by lookup in
        # the task's own source tables.
        wanted = set(membership.loc[unresolved, "record_id"].astype(str))
        owner: dict[str, str] = {}
        for frame in load_sources(task):
            if "id" not in frame.columns:
                continue
            name = aliases.canonical_source_name(
                task, frame.attrs["dataset_name"])
            for record_id in wanted & set(frame["id"].astype(str)):
                owner.setdefault(record_id, name)
        membership.loc[unresolved, "source"] = (
            membership.loc[unresolved, "record_id"].astype(str).map(owner)
        )
    return SilverStandard(fused=fused, membership=membership, cell_provenance=None)


def build_membership_from_fused(fused: pd.DataFrame) -> pd.DataFrame:
    """Long-form (record_id, source, cluster_id) from the fused frame's
    ``_fusion_sources`` / ``_fusion_source_datasets`` columns, cluster_id =
    the fused row's ``_id`` (port of the best-of-breed pipeline's builder)."""

    def coerce_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [value]
        return [value] if value is not None else []

    rows = []
    for _, row in fused.iterrows():
        sources = coerce_list(row.get("_fusion_sources"))
        datasets = coerce_list(row.get("_fusion_source_datasets"))
        if len(datasets) < len(sources):
            datasets = datasets + [None] * (len(sources) - len(datasets))
        for record_id, dataset in zip(sources, datasets):
            rows.append({
                "record_id": record_id,
                "source": dataset,
                "cluster_id": row["_id"],
            })
    return pd.DataFrame(rows, columns=["record_id", "source", "cluster_id"])


def _fusion_ratio(fused: pd.DataFrame,
                  membership: pd.DataFrame | None) -> float | None:
    """Share of output records assembled from >= 2 source records (the
    reference-free fusion ratio). Not emitted by the PyDI panel."""
    if membership is None or "_id" not in fused.columns:
        return None
    sizes = membership.groupby("cluster_id")["record_id"].nunique()
    fused_ids = set(fused["_id"].astype(str))
    sizes = sizes[sizes.index.astype(str).isin(fused_ids)]
    if not len(fused):
        return None
    return round(float((sizes >= 2).sum()) / len(fused), 4)


# Ground-truth cluster metrics that carry no information when every gold
# cluster holds one record (see degenerate_gold_clusters_warning).
DEGENERATE_CLUSTER_FIELDS = ("bcubed_recall", "bcubed_f1", "mean_jaccard")


def degenerate_gold_clusters_warning(
    gold_membership: pd.DataFrame | None,
) -> str | None:
    """Warning text when no gold cluster has >= 2 records, else None.

    PyDI aligns and scores clusters on the records BOTH memberships share.
    With singleton gold clusters, whenever the two memberships share records,
    BCubed recall is 1.0 by construction, BCubed F1 = 2P/(1+P) restates
    precision, and the mean Jaccard alignment equals precision; those three
    are nulled. bcubed_precision still drops when the output merges two gold
    entities, and entity_recovery drops for merges and for gold records the
    output's membership leaves out; those two stay. The games gold
    (test_set.xml) lists only the metacritic anchor id per entity, so all
    100 clusters are singletons in every tier. An empty or absent membership
    is not called degenerate.
    """
    if gold_membership is None or gold_membership.empty:
        return None
    sizes = gold_membership.groupby("cluster_id")["record_id"].nunique()
    if (sizes >= 2).any():
        return None
    return (
        f"ground_truth {', '.join(DEGENERATE_CLUSTER_FIELDS)} set to None: "
        f"all {len(sizes)} gold clusters hold a single record, so BCubed "
        f"recall is 1.0 by construction and F1 and mean Jaccard restate "
        f"precision; bcubed_precision counts merged gold entities, "
        f"entity_recovery merged and missing ones"
    )


def extract_panel_metrics(
    panel: dict,
    fusion_ratio: float | None,
    *,
    gold_membership: pd.DataFrame | None = None,
    warnings_sink: list[str] | None = None,
) -> dict:
    """The paper's reporting set: individual reference-free (RF) and
    ground-truth (GR) metrics; PyDI's composite score is not part of it.
    Absent panel sections yield None.

    ``gold_membership``: when every gold cluster is a singleton the
    cluster-structure fields are nulled (degenerate_gold_clusters_warning)
    and the reason is appended to ``warnings_sink``."""

    def dig(*path):
        node = panel
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                return None
        return round(node, 4) if isinstance(node, float) else node

    density_deltas = dig("coverage", "fact", "GR", "density_delta_per_attribute")
    value_density_delta = None
    if isinstance(density_deltas, dict) and density_deltas:
        deltas = [float(v["delta"]) for v in density_deltas.values()
                  if isinstance(v, dict) and v.get("delta") is not None]
        if deltas:
            value_density_delta = round(sum(deltas) / len(deltas), 4)

    ground_truth = {
        "entity_recovery": dig("coverage", "entity", "GR", "recovery_rate"),
        "value_drift": dig("coverage", "fact", "GR", "overall_drift"),
        "value_density_delta": value_density_delta,
        "schema_validity_delta": dig("consistency", "GR", "delta"),
        "bcubed_precision": dig("correctness", "cluster", "GR",
                                "bcubed", "precision"),
        "bcubed_recall": dig("correctness", "cluster", "GR",
                             "bcubed", "recall"),
        "bcubed_f1": dig("correctness", "cluster", "GR", "bcubed", "f1"),
        "fusion_accuracy": dig("correctness", "fact", "GR",
                               "macro_accuracy"),
        "fully_correct_rate": dig("correctness", "fact", "GR",
                                  "fully_correct_cluster_rate"),
        "mean_jaccard": dig("correctness", "cluster", "GR", "alignment",
                            "mean_jaccard"),
        "same_source_collision_rate": dig("coverage", "source_based", "GR",
                                          "same_source_collision_rate",
                                          "pipe"),
    }
    warning = degenerate_gold_clusters_warning(gold_membership)
    # recall is 1.0 by construction only if the memberships share records;
    # at 0.0 nothing aligned and the zeros are the finding
    if warning is not None and ground_truth.get("bcubed_recall") == 1.0:
        for field in DEGENERATE_CLUSTER_FIELDS:
            ground_truth[field] = None
        if warnings_sink is not None:
            warnings_sink.append(warning)

    return {
        "reference_free": {
            "entity_gain": dig("coverage", "entity", "RF",
                               "row_gain_vs_largest_input"),
            "density_gain": dig("coverage", "fact", "RF", "density_gain"),
            "output_density": dig("coverage", "fact", "RF", "density_output"),
            "fusion_ratio": fusion_ratio,
            "schema_validity": dig("consistency", "RF", "consistency_score"),
        },
        "ground_truth": ground_truth,
    }


def _coerce_declared_strings(frame: pd.DataFrame, schema_path: Path) -> pd.DataFrame:
    """Give attributes the target schema declares as ``type: string`` their
    text back before the panel.

    ``pd.read_csv`` infers numbers: a column filled with "1999" or "19"
    arrives as int64/float64, and PyDI's schema checker would then fail every
    cell on ``type`` -- the reader's inference, not what the system wrote.
    Only numeric-inferred columns are touched; integer-valued floats render
    without ".0", missing cells stay missing. Array and numeric schema
    attributes are left alone.
    """
    try:
        with open(schema_path) as handle:
            properties = json.load(handle).get("properties", {})
    except (OSError, ValueError):
        return frame

    def declared_string(node) -> bool:
        types = node.get("type")
        types = types if isinstance(types, list) else [types]
        return "string" in types and not ({"array", "integer", "number"} & set(types))

    columns = [
        column for column, node in properties.items()
        if isinstance(node, dict) and declared_string(node)
        and column in frame.columns and frame[column].dtype.kind in "biuf"
    ]
    if not columns:
        return frame

    def text(value):
        if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
            return value
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    frame = frame.copy()
    for column in columns:
        frame[column] = frame[column].astype(object).map(text)
    return frame


def _coerce_declared_arrays(frame: pd.DataFrame, schema_path: Path) -> pd.DataFrame:
    """Honour the submission format's array serialization before the panel.

    An attribute the target schema declares as ``type: array`` may be
    written as a JSON list (``["A", "B"]``) or as a ``|``-separated string
    (``A|B``). PyDI's schema-consistency checker only parses strings that
    start with '[' and end with ']', so ``A|B`` (and a bare single value)
    would fail on ``type``. Pipe-joined and bare scalars are lifted to lists
    for the declared array columns only (games genres, companies keypeople,
    music tracks, papers authors).
    """
    try:
        with open(schema_path) as handle:
            properties = json.load(handle).get("properties", {})
    except (OSError, ValueError):
        return frame
    array_columns = [
        column for column, node in properties.items()
        if node.get("type") == "array" and column in frame.columns
    ]
    if not array_columns:
        return frame

    def lift(value):
        if isinstance(value, (list, tuple, set)) or value is None:
            return value
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text or (text.startswith("[") and text.endswith("]")):
            return value
        return [part.strip() for part in text.split("|") if part.strip()]

    frame = frame.copy()
    for column in array_columns:
        frame[column] = frame[column].map(lift)
    return frame


def score_e2e(
    task: TaskSpec,
    fused: pd.DataFrame,
    correspondences: pd.DataFrame,
    *,
    membership: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Reference-free and ground-truth panel rows of one fused output."""
    from PyDI.evaluation.panel import compute_e2e_panel

    cfg = load_panel_config(task.domain)
    gold = load_panel_gold(task, cfg.source_prefix_map)
    fused = aliases.normalize_fused_sources(task, fused)
    membership = aliases.normalize_membership(task, membership)
    if membership is None:
        membership = build_membership_from_fused(fused)
    column_types = {
        k: v for k, v in cfg.column_types.items() if k in set(fused.columns)
    }
    schema_path = task.root / "input" / "schemamatching" / "target_schema.json"

    fused = _coerce_declared_arrays(fused, schema_path)
    fused = _coerce_declared_strings(fused, schema_path)

    panel = compute_e2e_panel(
        pipe_fused=fused,
        correspondences_pipe=correspondences,
        sources_pipe=load_panel_sources(task),
        gold=gold,
        column_types=column_types,
        target_schema=schema_path,
        taxonomy_base_path=task.root,
        pipe_id_column="_id",
        gold_id_column="cluster_id",
        pipe_membership=membership,
        numerical_tolerance=cfg.tolerance_default,
        numerical_tolerance_overrides=cfg.tolerance_overrides,
        composite_weights=cfg.composite_weights,
        source_prefix_map=cfg.source_prefix_map,
        usecase=task.domain,
        gold_source_label="fusion gold",
    )
    if out_dir is not None:
        panel.write(out_dir)
    warnings = list(panel.warnings or [])
    metrics = extract_panel_metrics(
        panel.panel or {}, _fusion_ratio(fused, membership),
        gold_membership=gold.membership, warnings_sink=warnings,
    )
    return {
        "stage": "e2e_panel",
        "task_id": task.task_id,
        # the paper's reporting set (PyDI's composite score is left out; the
        # paper does not use it)
        **metrics,
        "warnings": warnings,
        "known_bug_fields": list(KNOWN_BUG_FIELDS),
    }


# --------------------------------------------------------------------------
# Silver reference: another pipeline's fused output (the paper: P1's) as the
# reference, computed as for the paper's end-to-end table.

def source_lookup(task: TaskSpec, sources: list[pd.DataFrame] | None = None) -> dict[str, str]:
    """record id -> canonical source name over the task's source tables."""
    lookup: dict[str, str] = {}
    for frame in sources if sources is not None else load_panel_sources(task):
        src = aliases.canonical_source_name(task, frame.attrs["dataset_name"])
        for rid in frame["id"].astype(str):
            if lookup.get(rid, src) != src:
                raise ValueError(f"{task.task_id}: record id {rid} occurs in two sources")
            lookup[rid] = src
    return lookup


def repaired_membership(task: TaskSpec, fused: pd.DataFrame, lookup: dict[str, str],
                        supplied: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Membership (record_id, source, cluster_id) in the gold currency, with
    missing or empty source names restored from the record-id -> source
    lookup (the stored P2 papers output leaves some source names empty).
    ``supplied``: a membership table; default: built from ``_fusion_sources``."""
    m = supplied if supplied is not None else build_membership_from_fused(fused)
    m = aliases.normalize_membership(task, m).astype(str)
    fixes = 0
    unknown = 0
    srcs = []
    for rid, s in zip(m["record_id"], m["source"]):
        exp = lookup.get(rid)
        if exp is None:
            unknown += 1
            srcs.append(s)
            continue
        if s != exp:
            if not (s in ("None", "none", "nan", "unknown", "")
                    or aliases.canonical_source_name(task, s) == exp):
                raise ValueError(f"{task.task_id}: record {rid} names source {s!r}, the data says {exp!r}")
            fixes += 1
        srcs.append(exp)
    m["source"] = srcs
    return m, {"rows": int(len(m)), "source_repairs": fixes, "unknown_record_ids": unknown}


def read_pipeline_fused(path: Path) -> pd.DataFrame:
    """A stored fused table as the paper's panels read it: ``_id`` as text,
    PyDI's bookkeeping columns ``_fusion_metadata`` / ``_fusion_confidence``
    dropped (gzip accepted)."""
    return pd.read_csv(path, dtype={"_id": str}, low_memory=False,
                       usecols=lambda c: c not in ("_fusion_metadata", "_fusion_confidence"))


def load_p1_silver(task: TaskSpec, fused_path: Path, lookup: dict[str, str]):
    """P1's fused output as a PyDI SilverStandard: cluster_id = P1's ``_id``,
    membership from ``_fusion_sources``, values coerced like a pipeline
    output (declared arrays lifted, declared strings restored)."""
    from PyDI.evaluation.silver_standard import SilverStandard

    schema = task.root / "input" / "schemamatching" / "target_schema.json"
    p1 = read_pipeline_fused(fused_path)
    p1 = aliases.normalize_fused_sources(task, p1)
    pm, pm_info = repaired_membership(task, p1, lookup)
    p1 = _coerce_declared_strings(_coerce_declared_arrays(p1, schema), schema)
    p1 = p1.drop(columns=["cluster_id"], errors="ignore").rename(columns={"_id": "cluster_id"})
    if not p1["cluster_id"].is_unique:
        raise ValueError(f"{fused_path}: _id is not unique")
    info = {"clusters": int(len(p1)), "membership": pm_info}
    return SilverStandard(p1, pm, None), info


def install_sparse_alignment():
    """Exact sparse cluster alignment for PyDI's ``cluster_alignment``.

    PyDI compares every silver cluster with every output cluster; on the
    large domains that is too slow. The patched copy iterates, for each
    silver cluster, only the output clusters that share at least one record
    with it, in the original output-cluster order; pairs it skips have zero
    overlap and contribute nothing. Every other line of PyDI's function is
    unchanged (the patch is applied to its source text and fails loudly if
    that text changed). The paper's silver rows were computed with it; on
    companies the unpatched alignment was run as well and gave identical
    metrics.

    Returns ``use(which)`` -- ``use("sparse")`` / ``use("original")`` swaps
    the function in PyDI's clustering and panel modules -- and an info dict
    with the sha256 of both function texts.
    """
    import inspect

    import PyDI.evaluation.clustering as clustering
    import PyDI.evaluation.panel as panel_mod

    original = clustering.cluster_alignment
    source = inspect.getsource(original)
    needle = "        for pipe_id, pipe_members in pipe_clusters.items():\n"
    head = "    table_rows: List[Dict[str, object]] = []\n"
    if source.count(needle) != 1 or source.count(head) != 1:
        raise RuntimeError("PyDI cluster_alignment changed; the sparse patch no longer applies "
                           "(use the pinned PyDI commit or alignment='original')")
    fast = source.replace(head, "    pipe_order = {pid: i for i, pid in enumerate(pipe_clusters)}\n" + head)
    fast = fast.replace(needle,
                        "        candidates = {pipe[r] for r in silver_members}\n"
                        "        for pipe_id in sorted(candidates, key=pipe_order.__getitem__):\n"
                        "            pipe_members = pipe_clusters[pipe_id]\n")
    ns = dict(clustering.__dict__)
    exec(compile(fast, "<exact-sparse-cluster-alignment>", "exec"), ns)  # noqa: S102
    optimized = ns["cluster_alignment"]

    def use(which: str) -> None:
        fn = optimized if which == "sparse" else original
        clustering.cluster_alignment = fn
        panel_mod.cluster_alignment = fn

    info = {"original_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "optimized_sha256": hashlib.sha256(fast.encode()).hexdigest()}
    return use, info


def sr_metrics(panel: dict) -> dict:
    """The silver-reference (SR) rows of the paper's end-to-end table."""
    def clean(x):
        try:
            import numpy as np
            if isinstance(x, np.generic):
                x = x.item()
        except ImportError:
            pass
        if isinstance(x, float) and x != x:
            return None
        return x

    c, k, s = panel["coverage"], panel["correctness"], panel["consistency"]
    deltas = c["fact"]["SR"]["density_delta_per_attribute"]
    vals = [float(v["delta"]) for v in deltas.values() if isinstance(v, dict) and v.get("delta") is not None]
    return {key: clean(value) for key, value in {
        "entity_recovery": c["entity"]["SR"]["recovery_rate"],
        "value_drift": c["fact"]["SR"]["overall_drift"],
        "value_density_delta": sum(vals) / len(vals) if vals else None,
        "schema_validity_delta": s["SR"]["delta"],
        "bcubed_precision": k["cluster"]["SR"]["bcubed"]["precision"],
        "bcubed_recall": k["cluster"]["SR"]["bcubed"]["recall"],
        "bcubed_f1": k["cluster"]["SR"]["bcubed"]["f1"],
        "fusion_accuracy_micro": k["fact"]["SR"]["micro_accuracy"],
        "fusion_accuracy_macro": k["fact"]["SR"]["macro_accuracy"],
        "fully_correct_rate": k["fact"]["SR"]["fully_correct_cluster_rate"],
    }.items()}


def score_e2e_silver(
    task: TaskSpec,
    fused: pd.DataFrame,
    membership: pd.DataFrame,
    silver,
    *,
    sources: list[pd.DataFrame] | None = None,
    out_dir: Path | None = None,
    silver_label: str = "P1 fused output",
) -> tuple[dict, list]:
    """Silver-reference panel rows of one fused output against ``silver``
    (``load_p1_silver``). ``membership``: the output's membership in the gold
    currency (``repaired_membership``). The caller chooses the cluster
    alignment beforehand (``install_sparse_alignment``)."""
    from PyDI.evaluation.panel import compute_e2e_panel

    cfg = load_panel_config(task.domain)
    schema = task.root / "input" / "schemamatching" / "target_schema.json"
    fused = aliases.normalize_fused_sources(task, fused)
    fused = _coerce_declared_strings(_coerce_declared_arrays(fused, schema), schema)
    panel = compute_e2e_panel(
        pipe_fused=fused, sources_pipe=sources if sources is not None else load_panel_sources(task),
        column_types={k: v for k, v in cfg.column_types.items() if k in fused.columns},
        silver=silver, pipe_id_column="_id", silver_id_column="cluster_id",
        pipe_membership=membership, target_schema=schema, taxonomy_base_path=task.root,
        numerical_tolerance=cfg.tolerance_default, numerical_tolerance_overrides=cfg.tolerance_overrides,
        composite_weights=cfg.composite_weights, source_prefix_map=cfg.source_prefix_map,
        usecase=task.domain, silver_source_label=silver_label)
    if out_dir is not None:
        panel.write(out_dir)
    return sr_metrics(panel.panel), list(panel.warnings or [])
