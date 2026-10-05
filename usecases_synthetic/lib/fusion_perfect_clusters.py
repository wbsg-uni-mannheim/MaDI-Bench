"""Build perfect-cluster correspondences for fusion-committee evaluation.

Each committee evaluates against the
**perfect** output of the prior pipeline step, isolating the signal of
"how good is *this* committee" from "how good is the pipeline". For the
fusion committee, "perfect prior step" means: assume entity matching
discovered the cross-source positive set the likely-positive pool already declares
for the fusion validation + test entities.

Cluster ground truth comes from
``usecases_synthetic/pools/<domain>/pooled_positives.csv`` (built in
``scripts/build_pool.py`` as the evidence union of EM gold + human-baseline rule-matcher +
Ditto, with cross-source transitive closure, for companies, games and music;
``scripts/build_pool_products.py`` and ``scripts/build_pool_papers.py`` build
the products and papers pools). The
pool already names every cross-source positive
the variant pipeline can reach; record IDs survive every K-knob
mutation so the pool stays the authoritative cluster source across
baseline + augmented variants.

For each fusion-gold entity ID:

1. Look up the entity in the pool's partner graph (symmetric over
   ``id1`` / ``id2`` columns).
2. Build the transitive closure across all source-pair partners so a
   metacritic↔dbpedia + dbpedia↔sales partnership becomes a single
   3-source cluster.
3. Emit hub-and-spoke correspondences (entity_id → each partner)
   that the fusion engine's connected-components grouping turns into
   a single record group.

Entities that appear in the fusion gold but not in the pool are
emitted as singleton self-edges so ``include_singletons=True`` in the
fusion engine keeps them in the fused output for evaluation (the eval
then degrades gracefully to "compare the lone source's value to the
gold").
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .fusion_gold_keys import gold_anchor, source_id_members
from .variant_loader import VariantBundle

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = REPO_ROOT / "usecases_synthetic" / "pools"


# ---------------------------------------------------------------------------
# Pool loading + partner graph
# ---------------------------------------------------------------------------


def _pool_path(domain: str) -> Path:
    return POOL_DIR / domain / "pooled_positives.csv"


def _partner_graph(domain: str) -> dict[str, set[str]]:
    """Return a symmetric ``{id: {partner, ...}}`` map from the pool."""
    path = _pool_path(domain)
    if not path.exists():
        raise FileNotFoundError(
            f"Pool for domain {domain!r} not found at {path}. "
            "Run scripts/build_pool.py to (re)generate it."
        )
    pool = pd.read_csv(path)
    partners: dict[str, set[str]] = {}
    for a, b in zip(pool["id1"].astype(str), pool["id2"].astype(str), strict=True):
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return partners


def _transitive_closure(seed: str, partners: dict[str, set[str]]) -> set[str]:
    """BFS the partner graph starting at ``seed`` to return its full cluster."""
    visited: set[str] = set()
    queue: list[str] = [seed]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        for nbr in partners.get(node, set()):
            if nbr not in visited:
                queue.append(nbr)
    return visited


# ---------------------------------------------------------------------------
# Legacy DOI helpers -- NOT used by the generator any more
# ---------------------------------------------------------------------------
#
# No domain carries a DOI today: the released papers sources and fusion gold
# have none, and the gold is keyed by ``source_ids`` (see
# :func:`build_perfect_clusters`). The two helpers stay importable only for
# the DOI-gated code in ``pipelines/lib`` (``stage_runners`` behind
# ``gold_id_column == "doi"``, ``canonical_loader``'s papers workflow
# silver), which imports them lazily.

_DOI_PREFIX_RE = re.compile(r"^(https?://)?(dx\.)?doi\.org/", re.IGNORECASE)


def _normalize_doi(value: object) -> str | None:
    """Lowercase + prefix-strip a DOI; ``None`` for empty/NaN (legacy)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("none", "nan"):
        return None
    text = _DOI_PREFIX_RE.sub("", text)
    return text.lower() or None


def _doi_to_record_ids(bundle: VariantBundle) -> dict[str, list[str]]:
    """``{normalized_doi: [source_record_id, ...]}`` across the sources that
    carry a ``doi`` column -- none today (legacy)."""
    out: dict[str, list[str]] = {}
    for df in bundle.sources.values():
        if "doi" not in df.columns or "id" not in df.columns:
            continue
        for rid, doi in zip(df["id"].astype(str), df["doi"], strict=False):
            nd = _normalize_doi(doi)
            if nd is not None:
                out.setdefault(nd, []).append(rid)
    return out


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def build_perfect_clusters(domain: str, bundle: VariantBundle) -> dict[str, set[str]]:
    """Return the perfect-cluster map for a domain.

    Maps each fusion-gold entity to the set of source-record IDs that
    belong to its cluster. Clusters are derived from the transitive
    closure of the pool's partner graph.

    * id-keyed gold (companies, games, music, the generator's products
      copy): the key is the gold ``id``, itself a source-record id, and the
      cluster is its pool closure (always including the id).
    * source_ids-keyed gold (papers at every tier, the products task
      variants; no ``id`` column): the key is the anchor of the
      record's members (:func:`fusion_gold_keys.gold_anchor`, the id the
      public scorer keys the record on) and the cluster is the members
      plus the pool closure of every member. The members are the gold's
      own statement of the cluster; the closure adds what the pool links
      to them, exactly as for an id-keyed record. Members a variant
      dropped stay in the cluster (like dropped pool partners of an
      id-keyed record) -- the fusion engine groups only records present.
    """
    if bundle.fusion_gold is None or bundle.fusion_gold.empty:
        return {}
    partners = _partner_graph(domain)
    fg = bundle.fusion_gold
    clusters: dict[str, set[str]] = {}
    if "id" in fg.columns:
        # Standard domains: each fusion-gold entity ID is itself a source-record
        # ID and an anchor node in the pool partner graph.
        for entity_id in fg["id"].astype(str):
            eid = entity_id.strip()
            if not eid:
                continue
            cluster = _transitive_closure(eid, partners)
            if not cluster:
                cluster = {eid}
            clusters[eid] = cluster
    elif "source_ids" in fg.columns:
        n_unkeyed = 0
        for value in fg["source_ids"]:
            members = source_id_members(value)
            anchor = gold_anchor(members, domain) if members else None
            if anchor is None:
                n_unkeyed += 1
                continue
            cluster: set[str] = set(members)
            for rid in members:
                cluster |= _transitive_closure(rid, partners)
            clusters.setdefault(anchor, set()).update(cluster)
        if n_unkeyed:
            logger.warning(
                "Perfect clusters for %s: %d fusion-gold rows carry no "
                "source_ids member of a %s source and are skipped",
                domain,
                n_unkeyed,
                domain,
            )
    else:
        raise KeyError(
            f"fusion_gold for domain {domain!r} has neither an 'id' nor a "
            f"'source_ids' column to key clusters on; columns={list(fg.columns)}"
        )
    return clusters


def build_perfect_clusters_correspondences(
    domain: str, bundle: VariantBundle
) -> pd.DataFrame:
    """Return correspondences DataFrame that yields the perfect clusters.

    Strategy: for each cluster of N members, emit ``N-1`` hub-and-spoke
    edges between member[0] and every other member. Connected components
    of the resulting graph = the perfect clusters. Singleton clusters
    (N=1) emit one self-edge so ``include_singletons=True`` keeps them
    in the fused output.

    Parameters
    ----------
    domain
        Domain name (``companies`` / ``games`` / ``music`` / ``products`` /
        ``papers``).
    bundle
        Variant bundle (baseline or augmented).

    Returns
    -------
    DataFrame
        Columns ``id1``, ``id2``, ``score`` (= 1.0 everywhere).
    """
    clusters = build_perfect_clusters(domain, bundle)
    rows: list[tuple[str, str, float]] = []
    n_singletons = 0
    cluster_sizes: list[int] = []
    for entity_id, members in clusters.items():
        sorted_members = sorted(members)
        cluster_sizes.append(len(sorted_members))
        if len(sorted_members) == 1:
            sole = sorted_members[0]
            rows.append((sole, sole, 1.0))
            n_singletons += 1
            continue
        hub = sorted_members[0]
        for other in sorted_members[1:]:
            rows.append((hub, other, 1.0))
    if not rows:
        return pd.DataFrame(columns=["id1", "id2", "score"])
    df = pd.DataFrame(rows, columns=["id1", "id2", "score"])
    df = df.drop_duplicates(subset=["id1", "id2"], ignore_index=True)
    if cluster_sizes:
        avg_size = sum(cluster_sizes) / len(cluster_sizes)
    else:
        avg_size = 0.0
    logger.info(
        "Perfect-cluster correspondences for %s: %d clusters, %d edges, "
        "%d singletons, avg cluster size %.2f",
        domain,
        len(clusters),
        len(df),
        n_singletons,
        avg_size,
    )
    return df
