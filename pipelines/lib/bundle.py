"""Pipeline state container for the best-of-breed orchestrator.

Wraps an existing :class:`usecases_synthetic.lib.variant_loader.VariantBundle`
and threads each stage's winner output through the pipeline. The
underlying ``VariantBundle`` carries the immutable inputs (source
DataFrames, SM gold, EM gold splits, fusion gold); the
``PipelineState`` adds the mutable per-stage outputs (winner predictions
+ chained intermediates).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from usecases_synthetic.lib.variant_loader import VariantBundle, load_variant

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Mutable state threaded through the best-of-breed pipeline.

    Parameters
    ----------
    bundle : VariantBundle
        The underlying bundle (base task or variant): the run's inputs.
    sm_winner : str
        Name of the SM committee member chosen for this run.
    sm_mapping_df : pandas.DataFrame
        The winning SM mapping (``source_dataset``, ``source_column``,
        ``target_dataset``, ``target_column``, ``score``).
    norm_winner : str
        Name of the Norm committee member chosen (informational only:
        Norm does not transform downstream frames here).
    blocker_winner_per_pair : dict
        ``{pair_key: blocker_name}`` — picked per source-pair by the
        EM blocking committee composition logic.
    matcher_winner : str
        Name of the EM matcher chosen for this run.
    matcher_predictions : dict
        ``{pair_key: DataFrame[id1, id2, score]}`` — the winning
        matcher's post-clustering predictions per pair.
    refinement_winner : str
        Name of the chosen refinement method (``baseline`` /
        ``greedy`` / ``mbm``).
    correspondences : pandas.DataFrame
        Concatenated post-refinement correspondences across all pairs
        (``id1``, ``id2``, ``score``, ``notes``). Input to fusion.
    fusion_winner : str
        Name of the chosen fusion committee member.
    fused : pandas.DataFrame
        Final fused output from the winning fusion strategy.
    """

    bundle: VariantBundle

    sm_winner: str = ""
    sm_mapping_df: pd.DataFrame | None = None

    norm_winner: str = ""

    blocker_winner_per_pair: dict[str, str] = field(default_factory=dict)
    matcher_winner: str = ""
    matcher_predictions: dict[str, pd.DataFrame] = field(default_factory=dict)

    refinement_winner: str = ""
    correspondences: pd.DataFrame | None = None

    fusion_winner: str = ""
    fused: pd.DataFrame | None = None


def load_pipeline_bundle(
    domain: str,
    *,
    level: str = "baseline",
    bundle_source: str = "synthetic_baseline",
) -> VariantBundle:
    """Load a ``VariantBundle`` for the best-of-breed pipeline.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"products"``, ``"music"``).
    level : str, default ``"baseline"``
        Bundle level: ``"baseline"`` (the base task) or a variant level
        (``"easy"``, ``"medium"``, ``"hard"``).
    bundle_source : str, default ``"synthetic_baseline"``
        Which physical data tree to load from.

        - ``"synthetic_baseline"``: route through
          :func:`usecases_synthetic.lib.variant_loader.load_variant`.
          For domains with a synthetic ``data_root`` override
          (currently only products) this lands on
          ``usecases_synthetic/usecases/<domain>/``; for others it
          lands on ``use cases/<domain>/base/``. The papers base task
          loads through
          :func:`pipelines.lib.canonical_loader.load_canonical_papers_bundle`
          under either ``bundle_source``.
        - ``"canonical"``: read the base task directly from
          ``use cases/<domain>/base/`` regardless of any synthetic
          ``data_root`` override. Implemented for products via
          :func:`pipelines.lib.canonical_loader.load_canonical_products_bundle`;
          for music / games / companies this is equivalent to
          ``synthetic_baseline`` because no override is in effect.

        Papers ignores ``bundle_source``: its base task always loads through
        :func:`pipelines.lib.canonical_loader.load_canonical_papers_bundle`.

    Returns
    -------
    VariantBundle
        Loaded bundle.
    """
    # Synthetic-side variant_loader.VALID_BUNDLE_LEVELS =
    # ['baseline','easy','medium','hard']. The canonical_loader path
    # only handles baseline; pass any non-baseline level through to
    # load_variant (which serves the use cases/<domain>/<level>/ tree).
    if level not in {"baseline", "easy", "medium", "hard"}:
        raise ValueError(
            f"Unknown level {level!r}; expected one of "
            "{baseline, easy, medium, hard}."
        )
    if bundle_source not in {"synthetic_baseline", "canonical"}:
        raise ValueError(
            f"Unknown bundle_source {bundle_source!r}; "
            "expected 'synthetic_baseline' or 'canonical'."
        )

    if domain == "products":
        return _load_products_bundle(level=level, bundle_source=bundle_source)

    if level == "baseline" and domain == "papers":
        # The papers base task (``use cases/papers/base/``) loads through
        # the canonical loader, whatever ``bundle_source`` says.
        from .canonical_loader import load_canonical_papers_bundle

        logger.info("Loading papers canonical bundle (no synthetic-side variant)")
        return load_canonical_papers_bundle()

    if level != "baseline" and domain == "papers":
        # The papers variants (``use cases/papers/{easy,medium,hard}/``)
        # load through ``load_variant``. Their fusion gold uses the
        # canonical ``test_set.xml``/``validation_set.xml`` filenames but
        # holds JSON lines; ``_load_fusion_file`` content-sniffs the first
        # non-whitespace byte to dispatch correctly regardless of extension.
        logger.info(
            "Loading papers %s variant via load_variant "
            "(canonical_loader supports baseline only)",
            level,
        )

    logger.info(
        "Loading %s %s bundle via load_variant (bundle_source=%s)",
        domain,
        level,
        bundle_source,
    )
    return load_variant(domain, level=level)


def _load_products_bundle(*, level: str, bundle_source: str) -> VariantBundle:
    """Products: base task via the canonical loader, variants via
    ``load_variant`` (``use cases/products/<level>/``), both finished by
    :func:`~pipelines.lib.canonical_loader.finalize_products_bundle`: the
    ``cluster_id`` column is dropped from every source (never a matcher
    feature, never a fused value) and both fusion-gold frames are keyed on the
    anchor of their ``source_ids`` (the variant XML has no ``<id>``).
    ``bundle_source="synthetic_baseline"`` + baseline keeps the legacy
    generator-copy path (``usecases_synthetic/usecases/products/``), finished
    the same way.
    """
    from .canonical_loader import (
        finalize_products_bundle,
        load_canonical_products_bundle,
    )

    if level == "baseline" and bundle_source == "canonical":
        logger.info("Loading products canonical base bundle (use cases/products/base)")
        bundle = load_canonical_products_bundle()
    else:
        logger.info(
            "Loading products %s bundle via load_variant (bundle_source=%s)",
            level,
            bundle_source,
        )
        bundle = load_variant("products", level=level)
    return finalize_products_bundle(bundle)


__all__ = ["PipelineState", "load_pipeline_bundle"]
