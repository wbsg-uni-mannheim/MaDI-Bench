"""
Labeled set generation for entity matching.

This module provides functions to generate and cache labeled training and
validation sets for entity resolution. Labels are generated using LLM-based
matching.

Key concepts:
- Validation sets are used to evaluate blocking and matching performance
- Training sets are used to train ML-based matchers
- Both are cached as CSV files to avoid regeneration
- Uses iterative labeling with early stopping to minimize LLM costs
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..entitymatching.blocking import blocker_from_spec
from ..entitymatching import LLMBasedMatcher

logger = logging.getLogger(__name__)


# =============================================================================
# Cache Management
# =============================================================================


def _config_hash(config: Dict[str, Any]) -> str:
    """Generate a short hash for a configuration dict."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


def load_labeled_set_from_cache(
    output_dir: Path,
    left_name: str,
    right_name: str,
    *,
    set_type: str = "validation",
    config_hash: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Try to load a labeled set from cache.

    Parameters
    ----------
    output_dir : Path
        Directory containing cached files
    left_name, right_name : str
        Dataset names used in filename
    set_type : str
        Either "validation" or "training"
    config_hash : str, optional
        Config hash suffix for training sets with blocker_spec

    Returns
    -------
    pd.DataFrame or None
        Cached labeled set, or None if not found
    """
    output_dir = Path(output_dir)
    suffix = f"_{config_hash}" if config_hash else ""

    # Try both orderings (left_right and right_left)
    for left, right in [(left_name, right_name), (right_name, left_name)]:
        cache_path = output_dir / f"{set_type}_{left}_{right}{suffix}.csv"
        if cache_path.exists():
            logger.info(f"Loading cached {set_type} set from {cache_path}")
            return pd.read_csv(cache_path, dtype={"label": str})

    return None


def save_labeled_set_to_cache(
    df: pd.DataFrame,
    output_dir: Path,
    left_name: str,
    right_name: str,
    *,
    set_type: str = "validation",
    config_hash: Optional[str] = None,
    save_latest_alias: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a labeled set to cache.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled set with columns [id1, id2, label]
    output_dir : Path
        Directory to save to
    left_name, right_name : str
        Dataset names used in filename
    set_type : str
        Either "validation" or "training"
    config_hash : str, optional
        Config hash suffix for training sets
    save_latest_alias : bool
        If True, also save as {set_type}_{left}_{right}_latest.csv
    metadata : dict, optional
        Additional metadata to save as JSON alongside the CSV

    Returns
    -------
    Path
        Path to the saved cache file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{config_hash}" if config_hash else ""
    cache_path = output_dir / f"{set_type}_{left_name}_{right_name}{suffix}.csv"
    df.to_csv(cache_path, index=False)
    logger.info(f"Saved {set_type} set to {cache_path}")

    if save_latest_alias:
        latest_path = output_dir / f"{set_type}_{left_name}_{right_name}_latest.csv"
        df.to_csv(latest_path, index=False)

        if metadata:
            meta_path = output_dir / f"{set_type}_{left_name}_{right_name}_latest.json"
            meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))

    return cache_path


# =============================================================================
# Pair Utilities
# =============================================================================


def _pairs_set(pairs: pd.DataFrame) -> set[tuple[str, str]]:
    """Convert pairs DataFrame to set of (id1, id2) tuples."""
    if pairs.empty:
        return set()
    required = {"id1", "id2"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Pairs DataFrame missing required columns: {sorted(missing)}")
    return set(zip(pairs["id1"].astype(str), pairs["id2"].astype(str)))


def _safe_pairs_key(df: pd.DataFrame) -> pd.Series:
    """Create a unique string key for each pair."""
    return df["id1"].astype("string") + "||" + df["id2"].astype("string")


def drop_overlapping_pairs(df: pd.DataFrame, *, exclude_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Remove pairs from df that exist in exclude_pairs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with id1, id2 columns
    exclude_pairs : pd.DataFrame
        Pairs to exclude

    Returns
    -------
    pd.DataFrame
        df with overlapping pairs removed
    """
    if df.empty or exclude_pairs.empty:
        return df
    excluded = _pairs_set(exclude_pairs)
    keep_mask = ~df.apply(lambda r: (str(r["id1"]), str(r["id2"])) in excluded, axis=1)
    return df.loc[keep_mask].reset_index(drop=True)


def _training_file_prefix(cache_suffix: Optional[str] = None) -> str:
    """Return the active-training filename prefix for this cache family."""
    return f"training_{cache_suffix}" if cache_suffix else "training"


def _normalize_labeled_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Return [id1, id2, label] rows with labels normalized to TRUE/FALSE."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["id1", "id2", "label"])

    required = {"id1", "id2", "label"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["id1", "id2", "label"])

    normalized = df[["id1", "id2", "label"]].copy()
    normalized = normalized.dropna(subset=["id1", "id2", "label"])

    def _label_value(value: Any) -> Optional[str]:
        if isinstance(value, (bool, np.bool_)):
            return "TRUE" if bool(value) else "FALSE"
        text = str(value).strip().upper()
        if text in {"TRUE", "1", "YES", "Y", "MATCH"}:
            return "TRUE"
        if text in {"FALSE", "0", "NO", "N", "NON_MATCH", "NONMATCH"}:
            return "FALSE"
        return None

    normalized["label"] = normalized["label"].map(_label_value)
    normalized = normalized.dropna(subset=["label"]).reset_index(drop=True)
    return normalized


def _dedupe_labeled_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate labeled pairs while preserving the latest label for a pair."""
    df = _normalize_labeled_pairs(df)
    if df.empty:
        return df
    keyed = df.copy()
    keyed["_pair_key"] = _safe_pairs_key(keyed)
    keyed = keyed.drop_duplicates("_pair_key", keep="last")
    return keyed.drop(columns=["_pair_key"]).reset_index(drop=True)


def _load_active_progress_files(
    output_dir: Path,
    left_name: str,
    right_name: str,
    *,
    cache_suffix: Optional[str] = None,
) -> tuple[pd.DataFrame, List[Path]]:
    """Load unfinished active-learning label progress for a source pair."""
    output_dir = Path(output_dir)
    prefix = _training_file_prefix(cache_suffix)
    modes = ["active_bootstrap", "active_learning", "active_negative", "active_positive"]
    frames: List[pd.DataFrame] = []
    paths: List[Path] = []

    for mode in modes:
        for left, right in [(left_name, right_name), (right_name, left_name)]:
            path = output_dir / f"{prefix}_{mode}_{left}_{right}.csv"
            if not path.exists():
                continue
            try:
                frame = _normalize_labeled_pairs(pd.read_csv(path, dtype={"label": str}))
            except Exception as exc:
                logger.warning(f"Could not load active-learning progress {path}: {exc}")
                continue
            if frame.empty:
                continue
            frames.append(frame)
            paths.append(path)

    if not frames:
        return pd.DataFrame(columns=["id1", "id2", "label"]), []

    return _dedupe_labeled_pairs(pd.concat(frames, ignore_index=True)), paths


def _merge_active_progress(
    training_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    output_dir: Path,
    left_name: str,
    right_name: str,
    *,
    cache_suffix: Optional[str] = None,
) -> tuple[pd.DataFrame, int, List[Path]]:
    """Merge unfinished active-learning labels into the current training set."""
    base = _dedupe_labeled_pairs(training_set)
    progress, paths = _load_active_progress_files(
        output_dir,
        left_name,
        right_name,
        cache_suffix=cache_suffix,
    )
    if progress.empty:
        return base if not base.empty else training_set, 0, []

    combined = _dedupe_labeled_pairs(pd.concat([base, progress], ignore_index=True))
    if validation_set is not None and not validation_set.empty:
        combined = drop_overlapping_pairs(combined, exclude_pairs=validation_set[["id1", "id2"]])

    resumed = max(0, len(combined) - len(base))
    return combined, resumed, paths


# =============================================================================
# Candidate Collection
# =============================================================================


def collect_candidates_from_blockers(
    blockers: Union[List[Any], Any, pd.DataFrame],
    *,
    max_candidates: int = 5000,
    max_per_blocker: Optional[int] = None,
) -> pd.DataFrame:
    """
    Collect candidate pairs from one or more blockers.

    Parameters
    ----------
    blockers : list of blockers, single blocker, or DataFrame
        - List of blocker instances (each iterable yielding DataFrames)
        - Single blocker instance
        - Pre-computed DataFrame with id1, id2 columns
    max_candidates : int
        Maximum total candidates to collect
    max_per_blocker : int, optional
        Maximum candidates per blocker. If None, divides max_candidates evenly.

    Returns
    -------
    pd.DataFrame
        Deduplicated candidate pairs with columns [id1, id2]
    """
    if isinstance(blockers, pd.DataFrame):
        return blockers[["id1", "id2"]].drop_duplicates().head(max_candidates)

    # Normalize to list
    if not isinstance(blockers, list):
        blockers = [blockers]

    if not blockers:
        return pd.DataFrame(columns=["id1", "id2"])

    per_blocker = max_per_blocker or (max_candidates // len(blockers))
    frames: List[pd.DataFrame] = []

    for i, blocker in enumerate(blockers):
        try:
            candidates = _collect_from_single_blocker(blocker, limit=per_blocker)
            if not candidates.empty:
                frames.append(candidates[["id1", "id2"]])
                logger.debug(f"Blocker {i+1}/{len(blockers)}: {len(candidates)} candidates")
        except Exception as e:
            logger.warning(f"Blocker {i+1}/{len(blockers)} failed: {e}")

    if not frames:
        return pd.DataFrame(columns=["id1", "id2"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["id1", "id2"]).head(max_candidates)
    return combined.reset_index(drop=True)


def _collect_from_single_blocker(blocker, *, limit: int) -> pd.DataFrame:
    """Collect candidates from a single blocker (which yields batches)."""
    if limit <= 0:
        return pd.DataFrame(columns=["id1", "id2"])

    frames: List[pd.DataFrame] = []
    total = 0

    for batch in blocker:
        if batch is None or batch.empty:
            continue
        remaining = limit - total
        if remaining <= 0:
            break
        take = batch.head(remaining)
        frames.append(take)
        total += len(take)
        if total >= limit:
            break

    if not frames:
        return pd.DataFrame(columns=["id1", "id2"])
    return pd.concat(frames, ignore_index=True)


def _cap_pairs_per_cluster(
    candidates: pd.DataFrame, *, max_per_cluster: int, random_state: int = 42
) -> pd.DataFrame:
    """
    Limit pairs per id1 to avoid over-representing a single cluster.

    Parameters
    ----------
    candidates : pd.DataFrame
        Candidate pairs with id1, id2 columns
    max_per_cluster : int
        Max pairs per unique id1
    random_state : int
        Random seed for sampling

    Returns
    -------
    pd.DataFrame
        Capped and shuffled candidates
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=["id1", "id2"])
    if max_per_cluster <= 0:
        return candidates[["id1", "id2"]].head(0)

    candidates = candidates[["id1", "id2"]].dropna(subset=["id1", "id2"]).drop_duplicates()
    if "id1" not in candidates.columns:
        return candidates.reset_index(drop=True)

    capped = (
        candidates.groupby("id1", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), max_per_cluster), random_state=random_state))
        .reset_index(drop=True)
    )
    # Shuffle so labeling doesn't walk one id1 at a time
    capped = capped.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return capped


def _order_candidates_by_similarity_strata(
    candidates: pd.DataFrame,
    *,
    n_strata: int = 5,
    score_column: str = "similarity",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Order candidates by interleaving similarity strata.

    Divides candidates into n_strata by similarity score, then
    interleaves them so labeling hits all strata evenly. This ensures
    we label pairs across the full similarity spectrum rather than
    processing in random order.

    Parameters
    ----------
    candidates : pd.DataFrame
        Candidate pairs with id1, id2, and optionally a similarity column
    n_strata : int
        Number of similarity strata to create
    score_column : str
        Column containing similarity scores
    random_state : int
        Random seed for shuffling within strata

    Returns
    -------
    pd.DataFrame
        Candidates reordered to interleave similarity strata
    """
    if candidates is None or candidates.empty:
        return candidates

    if score_column not in candidates.columns or candidates[score_column].isna().all():
        # No similarity scores available - just shuffle randomly
        return candidates.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Assign to strata based on similarity quantiles
    candidates = candidates.copy()
    try:
        candidates["_stratum"] = pd.qcut(
            candidates[score_column].fillna(0),
            q=n_strata,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        # Not enough unique values for n_strata bins
        return candidates.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Shuffle within each stratum, then interleave
    strata = [
        g.sample(frac=1, random_state=random_state)
        for _, g in candidates.groupby("_stratum")
    ]

    if not strata:
        return candidates.drop(columns=["_stratum"]).reset_index(drop=True)

    # Round-robin interleave from all strata
    result = []
    max_len = max(len(s) for s in strata)
    for i in range(max_len):
        for stratum in strata:
            if i < len(stratum):
                result.append(stratum.iloc[i])

    if not result:
        return candidates.drop(columns=["_stratum"]).reset_index(drop=True)

    return pd.DataFrame(result).drop(columns=["_stratum"]).reset_index(drop=True)


def _select_balanced_set_with_hard_negatives(
    labeled: pd.DataFrame,
    target_size: int,
    target_positives: int,
    *,
    score_column: str = "similarity",
    hard_negative_threshold: float = 0.6,
    min_hard_negative_ratio: float = 0.3,
) -> pd.DataFrame:
    """
    Select a balanced set with a minimum ratio of hard negatives.

    Hard negatives are FALSE labels with high similarity scores.
    These are the most informative examples for training because
    they represent cases that look like matches but aren't.

    Parameters
    ----------
    labeled : pd.DataFrame
        Labeled pairs with columns [id1, id2, label] and optionally similarity
    target_size : int
        Target total number of pairs
    target_positives : int
        Target number of positive (matching) pairs
    score_column : str
        Column containing similarity scores
    hard_negative_threshold : float
        Similarity threshold above which a negative is considered "hard"
    min_hard_negative_ratio : float
        Minimum ratio of hard negatives among all negatives (0.0 to 1.0)

    Returns
    -------
    pd.DataFrame
        Balanced set with prioritized hard negatives
    """
    if labeled.empty:
        return labeled

    positives = labeled[labeled["label"].astype(str).str.upper() == "TRUE"]
    negatives = labeled[labeled["label"].astype(str).str.upper() == "FALSE"]

    # Select positives (unchanged from original logic)
    n_pos = min(len(positives), target_positives)
    selected_pos = positives.head(n_pos)

    # Calculate how many negatives we need
    n_neg = min(len(negatives), max(target_size - n_pos, 0))

    if n_neg == 0:
        result = selected_pos.copy()
        result = result.sample(frac=1, random_state=42).reset_index(drop=True)
        return result

    # Check if we have similarity scores for hard negative selection
    has_scores = (
        score_column in negatives.columns
        and negatives[score_column].notna().any()
    )

    if has_scores:
        # Separate hard and easy negatives
        hard_negs = negatives[negatives[score_column] >= hard_negative_threshold]
        easy_negs = negatives[negatives[score_column] < hard_negative_threshold]

        # Calculate target number of hard negatives
        target_hard = int(n_neg * min_hard_negative_ratio)
        n_hard = min(len(hard_negs), target_hard)

        # Fill remaining with easy negatives
        n_easy = n_neg - n_hard

        # If we don't have enough hard negatives, take what we have and fill with easy
        if n_hard < target_hard and len(easy_negs) > n_easy:
            # We can take more easy negatives to compensate
            n_easy = min(len(easy_negs), n_neg - n_hard)

        selected_neg = pd.concat(
            [hard_negs.head(n_hard), easy_negs.head(n_easy)],
            ignore_index=True,
        )

        # Log the hard negative ratio achieved
        actual_hard_ratio = n_hard / len(selected_neg) if len(selected_neg) > 0 else 0
        logger.info(
            f"Hard negative mining: {n_hard}/{len(selected_neg)} negatives are hard "
            f"({actual_hard_ratio:.1%}, target was {min_hard_negative_ratio:.1%})"
        )
    else:
        # No similarity scores - fall back to simple selection
        selected_neg = negatives.head(n_neg)

    result = pd.concat([selected_pos, selected_neg], ignore_index=True)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    return result


# =============================================================================
# Core Labeling
# =============================================================================


def generate_labeled_set(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    blockers: Union[List[Any], Any, pd.DataFrame],
    matcher: LLMBasedMatcher,
    chat_model,
    *,
    id_column: str = "id",
    target_size: int = 100,
    target_positives: int = 30,
    max_candidates: int = 5000,
    max_pairs_per_cluster: int = 0,
    exclude_pairs: Optional[pd.DataFrame] = None,
    progress_path: Optional[Path] = None,
    label_batch_size: int = 25,
    hard_negative_mining: bool = False,
    hard_negative_threshold: float = 0.6,
    min_hard_negative_ratio: float = 0.3,
    similarity_strata: int = 5,
    label_max_workers: int = 10,
) -> pd.DataFrame:
    """
    Generate a labeled set using provided blockers and matcher.

    This is the core labeling function. It:
    1. Collects candidates from blockers
    2. Optionally caps pairs per cluster
    3. Removes excluded pairs
    4. Labels iteratively until targets are met
    5. Optionally saves progress to disk for resumption

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Datasets to match
    blockers : list of blockers, single blocker, or DataFrame
        Candidate generators. Can be:
        - List of blocker instances
        - Single blocker instance
        - Pre-computed DataFrame with id1, id2 columns (may include similarity column)
    matcher : LLMBasedMatcher
        Matcher instance for labeling
    chat_model : BaseChatModel
        LangChain chat model for labeling
    id_column : str
        ID column name
    target_size : int
        Target number of pairs in labeled set
    target_positives : int
        Target number of positive (matching) pairs
    max_candidates : int
        Maximum candidates to collect from blockers
    max_pairs_per_cluster : int
        Max pairs per id1 to avoid over-representation. 0 = no limit.
    exclude_pairs : pd.DataFrame, optional
        Pairs to exclude (e.g., validation set when generating training)
    progress_path : Path, optional
        Path to save labeling progress for resumption
    label_batch_size : int
        Number of pairs to label per batch
    hard_negative_mining : bool
        If True, prioritize hard negatives (high-similarity non-matches) in final selection.
        Requires candidates to have a 'similarity' column.
    hard_negative_threshold : float
        Similarity threshold above which a negative is considered "hard" (default 0.6)
    min_hard_negative_ratio : float
        Target minimum ratio of hard negatives among all negatives (default 0.3)
    similarity_strata : int
        Number of similarity strata for stratified candidate ordering (default 5)
    label_max_workers : int
        Number of parallel LLM labeling workers. Use 1 for serial labeling.

    Returns
    -------
    pd.DataFrame
        Labeled set with columns [id1, id2, label] (and optionally similarity if available)
    """
    # Step 1: Collect candidates
    candidates = collect_candidates_from_blockers(blockers, max_candidates=max_candidates)

    if candidates.empty:
        raise ValueError("No candidates generated by blockers")

    logger.info(f"Collected {len(candidates)} candidate pairs from blockers")

    # Check if similarity scores are available
    has_similarity = "similarity" in candidates.columns and candidates["similarity"].notna().any()
    if hard_negative_mining and has_similarity:
        logger.info("Hard negative mining enabled with similarity scores")
    elif hard_negative_mining:
        logger.warning("Hard negative mining requested but no similarity scores available")

    # Step 2: Cap pairs per cluster if requested (preserve similarity column)
    if max_pairs_per_cluster > 0:
        before = len(candidates)
        # Save similarity before capping (which only keeps id1, id2)
        similarity_backup = None
        if has_similarity:
            similarity_backup = candidates[["id1", "id2", "similarity"]].copy()

        candidates = _cap_pairs_per_cluster(candidates, max_per_cluster=max_pairs_per_cluster)

        # Restore similarity scores
        if similarity_backup is not None and not candidates.empty:
            candidates = candidates.merge(
                similarity_backup, on=["id1", "id2"], how="left"
            )

        if len(candidates) != before:
            logger.info(f"Capped to {max_pairs_per_cluster} pairs per cluster: {before} -> {len(candidates)}")

    # Step 3: Remove excluded pairs
    if exclude_pairs is not None and not exclude_pairs.empty:
        before = len(candidates)
        candidates = drop_overlapping_pairs(candidates, exclude_pairs=exclude_pairs)
        removed = before - len(candidates)
        if removed:
            logger.info(f"Removed {removed} excluded pairs")

    if candidates.empty:
        raise ValueError("No candidates remaining after filtering")

    # Step 4: Order candidates by similarity strata if hard negative mining enabled
    if hard_negative_mining and has_similarity:
        candidates = _order_candidates_by_similarity_strata(
            candidates, n_strata=similarity_strata
        )
        logger.info(f"Ordered candidates by {similarity_strata} similarity strata")

    # Step 5: Label iteratively
    labeled = _label_iteratively(
        df_left=df_left,
        df_right=df_right,
        candidates=candidates,
        matcher=matcher,
        chat_model=chat_model,
        id_column=id_column,
        target_size=target_size,
        target_positives=target_positives,
        progress_path=progress_path,
        batch_size=label_batch_size,
        preserve_similarity=hard_negative_mining and has_similarity,
        max_workers=label_max_workers,
    )

    # Step 6: Select balanced set (with hard negative priority if enabled)
    if hard_negative_mining and "similarity" in labeled.columns:
        result = _select_balanced_set_with_hard_negatives(
            labeled,
            target_size,
            target_positives,
            hard_negative_threshold=hard_negative_threshold,
            min_hard_negative_ratio=min_hard_negative_ratio,
        )
    else:
        result = _select_balanced_set(labeled, target_size, target_positives)

    return result


def _label_iteratively(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    candidates: pd.DataFrame,
    matcher: LLMBasedMatcher,
    chat_model,
    *,
    id_column: str,
    target_size: int,
    target_positives: int,
    progress_path: Optional[Path],
    batch_size: int,
    preserve_similarity: bool = False,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Label candidates iteratively with early stopping and optional progress saving.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Source datasets
    candidates : pd.DataFrame
        Candidate pairs to label (may include similarity column)
    matcher : LLMBasedMatcher
        Matcher for labeling
    chat_model : BaseChatModel
        Chat model for LLM calls
    id_column : str
        ID column name
    target_size : int
        Target total pairs
    target_positives : int
        Target positive pairs
    progress_path : Path, optional
        Path to save/resume progress
    batch_size : int
        Pairs per batch
    preserve_similarity : bool
        If True, preserve similarity scores in output for hard negative mining
    max_workers : int
        Number of parallel LLM batch-labeling workers. Use 1 to preserve the
        previous serial behavior.

    Returns
    -------
    pd.DataFrame
        Labeled pairs with columns [id1, id2, label] (and similarity if preserve_similarity=True)
    """
    # Determine which columns to keep
    keep_cols = ["id1", "id2"]
    has_similarity = "similarity" in candidates.columns
    if preserve_similarity and has_similarity:
        keep_cols.append("similarity")

    # Build a lookup for similarity scores if needed
    similarity_lookup: Dict[tuple, float] = {}
    if preserve_similarity and has_similarity:
        for _, row in candidates.iterrows():
            key = (str(row["id1"]), str(row["id2"]))
            if pd.notna(row.get("similarity")):
                similarity_lookup[key] = float(row["similarity"])

    candidates_filtered = candidates[["id1", "id2"]].drop_duplicates().reset_index(drop=True)
    if candidates_filtered.empty:
        cols = ["id1", "id2", "label"]
        if preserve_similarity:
            cols.append("similarity")
        return pd.DataFrame(columns=cols)

    # Resume from progress if exists
    output_cols = ["id1", "id2", "label"]
    if preserve_similarity:
        output_cols.append("similarity")

    if progress_path and progress_path.exists():
        labeled = pd.read_csv(progress_path, dtype={"label": str})
        available_cols = [c for c in output_cols if c in labeled.columns]
        labeled = labeled[available_cols].dropna(subset=["id1", "id2"]).drop_duplicates()
        logger.info(f"Resuming from {len(labeled)} previously labeled pairs")
    else:
        labeled = pd.DataFrame(columns=output_cols)

    # Find remaining candidates
    already_labeled = set(_safe_pairs_key(labeled)) if not labeled.empty else set()
    remaining = candidates_filtered.loc[
        ~_safe_pairs_key(candidates_filtered).isin(already_labeled)
    ].reset_index(drop=True)

    if remaining.empty:
        # Return all labeled pairs (selection happens in caller)
        return labeled

    # Calculate budget
    target_negatives = target_size - target_positives
    max_labels = target_size * 5  # Cap total labeling to control LLM costs

    positives_list: List[Dict] = []
    negatives_list: List[Dict] = []

    # Initialize from existing labels
    if not labeled.empty:
        for _, row in labeled.iterrows():
            item = {"id1": row["id1"], "id2": row["id2"], "label": row["label"]}
            if preserve_similarity and "similarity" in labeled.columns:
                item["similarity"] = row.get("similarity")
            if str(row["label"]).upper() == "TRUE":
                positives_list.append(item)
            else:
                negatives_list.append(item)

    total_labeled = len(labeled)
    batch_num = 0
    candidate_idx = 0

    max_workers = max(1, int(max_workers or 1))
    if max_workers > 1:
        logger.info(f"Parallel LLM labeling enabled: {max_workers} workers")
    progress_bar = None
    try:
        from tqdm import tqdm
        progress_bar = tqdm(
            total=min(max_labels, len(remaining)),
            desc="    Labeling pairs",
            unit="pair",
            leave=True,
        )
    except ImportError:
        progress_bar = None

    def _targets_reached() -> bool:
        # Check stopping criteria
        have_enough_pos = len(positives_list) >= target_positives
        have_enough_neg = len(negatives_list) >= target_negatives

        if have_enough_pos and have_enough_neg:
            logger.info(f"Early stopping: targets reached ({len(positives_list)} pos, {len(negatives_list)} neg)")
            return True
        return False

    def _run_label_batch(batch: pd.DataFrame, batch_matcher: LLMBasedMatcher) -> pd.DataFrame:
        return batch_matcher.match(
            df_left=df_left,
            df_right=df_right,
            candidates=batch,
            id_column=id_column,
            chat_model=chat_model,
            generate_explanations=False,
            parse_strictness="skip",
        )

    def _handle_prediction(pred: Optional[pd.DataFrame]) -> None:
        nonlocal total_labeled, batch_num

        if pred is None or pred.empty:
            return

        for _, row in pred.iterrows():
            label = "TRUE" if bool(row.get("match", False)) else "FALSE"
            item = {"id1": row["id1"], "id2": row["id2"], "label": label}

            # Add similarity score if preserving
            if preserve_similarity:
                key = (str(row["id1"]), str(row["id2"]))
                item["similarity"] = similarity_lookup.get(key)

            if label == "TRUE":
                positives_list.append(item)
            else:
                negatives_list.append(item)

        total_labeled += len(pred)
        batch_num += 1

        # Save progress from the main thread only.
        if progress_path:
            all_labeled = pd.DataFrame(positives_list + negatives_list)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            all_labeled.to_csv(progress_path, index=False)

        logger.info(
            f"Batch {batch_num}: {len(positives_list)}/{target_positives} pos, "
            f"{len(negatives_list)}/{target_negatives} neg"
        )
        if progress_bar is not None:
            progress_bar.set_postfix(
                pos=f"{len(positives_list)}/{target_positives}",
                neg=f"{len(negatives_list)}/{target_negatives}",
                batches=batch_num,
            )
            progress_bar.update(len(pred))

    try:
        if max_workers == 1:
            while total_labeled < max_labels and candidate_idx < len(remaining):
                if _targets_reached():
                    break

                # Get next batch
                batch_end = min(candidate_idx + batch_size, len(remaining))
                batch = remaining.iloc[candidate_idx:batch_end].copy()
                candidate_idx = batch_end

                if batch.empty:
                    break

                # Label batch
                try:
                    _handle_prediction(_run_label_batch(batch, matcher))

                except Exception as e:
                    logger.warning(f"LLM labeling error: {e}")
                    break
        else:
            futures: Dict[Any, int] = {}
            completed: Dict[int, Any] = {}
            next_sequence = 0
            next_to_commit = 0
            stop_requested = False

            with ThreadPoolExecutor(max_workers=max_workers) as executor:

                def _submit_until_full() -> None:
                    nonlocal candidate_idx, next_sequence

                    while (
                        len(futures) < max_workers
                        and candidate_idx < len(remaining)
                        and total_labeled < max_labels
                        and not stop_requested
                        and not _targets_reached()
                    ):
                        batch_end = min(candidate_idx + batch_size, len(remaining))
                        batch = remaining.iloc[candidate_idx:batch_end].copy()
                        candidate_idx = batch_end

                        if batch.empty:
                            return

                        future = executor.submit(_run_label_batch, batch, LLMBasedMatcher())
                        futures[future] = next_sequence
                        next_sequence += 1

                _submit_until_full()
                while futures or completed:
                    if not futures:
                        break

                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        sequence = futures.pop(future)
                        try:
                            completed[sequence] = future.result()
                        except Exception as e:
                            completed[sequence] = e

                    while next_to_commit in completed:
                        result = completed.pop(next_to_commit)
                        next_to_commit += 1

                        if isinstance(result, Exception):
                            logger.warning(f"LLM labeling error: {result}")
                        else:
                            _handle_prediction(result)

                        if total_labeled >= max_labels or _targets_reached():
                            stop_requested = True
                            for future in futures:
                                future.cancel()
                            futures.clear()
                            break

                    if stop_requested:
                        break

                    _submit_until_full()
    finally:
        if progress_bar is not None:
            progress_bar.close()

    # Combine all labeled pairs
    all_labeled = pd.DataFrame(positives_list + negatives_list)

    if all_labeled.empty:
        cols = ["id1", "id2", "label"]
        if preserve_similarity:
            cols.append("similarity")
        return pd.DataFrame(columns=cols)

    # Save final progress
    if progress_path:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        all_labeled.to_csv(progress_path, index=False)

    # Return all labeled pairs - selection happens in caller (generate_labeled_set)
    return all_labeled


def _select_balanced_set(
    labeled: pd.DataFrame, target_size: int, target_positives: int
) -> pd.DataFrame:
    """Select a balanced set of positives and negatives."""
    if labeled.empty:
        return labeled

    positives = labeled[labeled["label"].astype(str).str.upper() == "TRUE"]
    negatives = labeled[labeled["label"].astype(str).str.upper() == "FALSE"]

    n_pos = min(len(positives), target_positives)
    selected_pos = positives.head(n_pos)

    n_neg = min(len(negatives), max(target_size - n_pos, 0))
    selected_neg = negatives.head(n_neg)

    result = pd.concat([selected_pos, selected_neg], ignore_index=True)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    return result


# =============================================================================
# High-Level Functions (Backward Compatible)
# =============================================================================


def load_or_generate_validation_set(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
    chat_model,
    output_dir: Path,
    *,
    blockers: Optional[List[Any]] = None,
    id_column: str = "id",
    target_size: int = 100,
    target_positives: int = 30,
    force_regenerate: bool = False,
    label_max_workers: int = 10,
) -> pd.DataFrame:
    """
    Load validation set from cache or generate if not exists.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Datasets to match
    left_name, right_name : str
        Names for the datasets (used in filename)
    chat_model : BaseChatModel
        LangChain chat model for labeling
    output_dir : Path
        Directory to store/load validation sets
    blockers : list of blockers, optional
        Blockers to use for candidate generation. If None, uses default blockers.
    id_column : str
        ID column name
    target_size : int
        Target number of pairs
    target_positives : int
        Target positive pairs
    force_regenerate : bool
        If True, regenerate even if cache exists
    label_max_workers : int
        Number of parallel LLM labeling workers. Use 1 for serial labeling.

    Returns
    -------
    pd.DataFrame
        Validation set with columns [id1, id2, label]
    """
    output_dir = Path(output_dir)

    # Try cache first
    if not force_regenerate:
        cached = load_labeled_set_from_cache(
            output_dir, left_name, right_name, set_type="validation"
        )
        if cached is not None:
            n_pos = (cached["label"].astype(str).str.upper() == "TRUE").sum()
            n_neg = (cached["label"].astype(str).str.upper() == "FALSE").sum()
            logger.info(f"Loaded {len(cached)} pairs ({n_pos} positive, {n_neg} negative)")
            return cached

    # Generate new validation set
    logger.info(f"Generating new validation set for {left_name} <-> {right_name}")

    # Use provided blockers or create defaults
    if blockers is None:
        blockers = _create_default_blockers(df_left, df_right, id_column, chat_model)

    matcher = LLMBasedMatcher()

    val_set = generate_labeled_set(
        df_left=df_left,
        df_right=df_right,
        blockers=blockers,
        matcher=matcher,
        chat_model=chat_model,
        id_column=id_column,
        target_size=target_size,
        target_positives=target_positives,
        max_candidates=target_size * 10,
        label_max_workers=label_max_workers,
    )

    # Save to cache
    save_labeled_set_to_cache(
        val_set, output_dir, left_name, right_name, set_type="validation"
    )

    return val_set


def load_or_generate_training_set(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
    chat_model,
    output_dir: Path,
    *,
    blockers: Optional[List[Any]] = None,
    blocker_spec: Optional[Dict[str, Any]] = None,
    id_column: str = "id",
    target_size: int = 500,
    target_positives: int = 150,
    force_regenerate: bool = False,
    exclude_pairs: Optional[pd.DataFrame] = None,
    max_candidate_pairs: Optional[int] = None,
    label_batch_size: int = 25,
    max_pairs_per_cluster: int = 3,
    hard_negative_mining: bool = True,
    hard_negative_threshold: float = 0.6,
    min_hard_negative_ratio: float = 0.3,
    label_max_workers: int = 10,
) -> pd.DataFrame:
    """
    Load training set from cache or generate if not exists.

    Training sets support incremental labeling with on-disk progress saving,
    allowing resumption if interrupted.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Datasets to match
    left_name, right_name : str
        Names for the datasets (used in filename)
    chat_model : BaseChatModel
        LangChain chat model for labeling
    output_dir : Path
        Directory to store/load training sets
    blockers : list of blockers, optional
        Blockers to use. If None and blocker_spec provided, creates from spec.
    blocker_spec : dict, optional
        Blocker specification (used for cache hashing and blocker creation)
    id_column : str
        ID column name
    target_size : int
        Target number of pairs
    target_positives : int
        Target positive pairs
    force_regenerate : bool
        If True, regenerate even if cache exists
    exclude_pairs : pd.DataFrame, optional
        Pairs to exclude (e.g., validation set)
    max_candidate_pairs : int, optional
        Max candidates to generate
    label_batch_size : int
        Batch size for labeling
    max_pairs_per_cluster : int
        Max pairs per id1 to avoid over-representation
    hard_negative_mining : bool
        If True (default), prioritize hard negatives in training set selection.
        Hard negatives are high-similarity non-matches that are most informative.
    hard_negative_threshold : float
        Similarity threshold above which a negative is considered "hard" (default 0.6)
    min_hard_negative_ratio : float
        Target minimum ratio of hard negatives among all negatives (default 0.3)
    label_max_workers : int
        Number of parallel LLM labeling workers. Use 1 for serial labeling.

    Returns
    -------
    pd.DataFrame
        Training set with columns [id1, id2, label]
    """
    output_dir = Path(output_dir)

    # Compute config hash for caching
    config_hash = _config_hash(blocker_spec) if blocker_spec else None

    # Set up paths
    suffix = f"_{config_hash}" if config_hash else ""
    cache_path = output_dir / f"training_{left_name}_{right_name}{suffix}.csv"
    candidates_path = output_dir / f"training_candidates_{left_name}_{right_name}{suffix}.csv"
    labeled_path = output_dir / f"training_labeled_{left_name}_{right_name}{suffix}.csv"
    latest_cache_path = output_dir / f"training_{left_name}_{right_name}_latest.csv"
    latest_meta_path = output_dir / f"training_{left_name}_{right_name}_latest.json"

    # Try cache first
    if not force_regenerate:
        cached = load_labeled_set_from_cache(
            output_dir, left_name, right_name, set_type="training", config_hash=config_hash
        )
        if cached is not None:
            # Check if we need to resume (partial set)
            n_pos = (cached["label"].astype(str).str.upper() == "TRUE").sum()
            want_pos = min(target_positives, target_size)

            if len(cached) >= target_size and n_pos >= want_pos:
                # Have enough, just apply exclusions
                if exclude_pairs is not None and not exclude_pairs.empty:
                    cached = drop_overlapping_pairs(cached, exclude_pairs=exclude_pairs)
                logger.info(f"Loaded {len(cached)} pairs ({n_pos} positive)")
                return cached

            logger.info(f"Cached set is partial ({len(cached)}/{target_size}, pos={n_pos}/{want_pos}), resuming...")

    # Generate/resume training set
    logger.info(f"Generating training set for {left_name} <-> {right_name}")

    # Calculate limits
    max_pool = max_candidate_pairs or max(target_size * 50, 5000)

    # Get or create blockers
    if blockers is None:
        if blocker_spec:
            blocker = blocker_from_spec(blocker_spec, df_left, df_right, id_column)
            blockers = [blocker]
            logger.info(f"Using blocker from spec: {blocker_spec.get('blocker_name')}")
        else:
            blockers = _create_default_blockers(df_left, df_right, id_column, chat_model)

    # Try to load cached candidates
    if candidates_path.exists() and not force_regenerate:
        candidates = pd.read_csv(candidates_path)
        logger.info(f"Loaded {len(candidates)} cached candidates")
    else:
        candidates = collect_candidates_from_blockers(blockers, max_candidates=max_pool)
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(candidates_path, index=False)
        logger.info(f"Generated {len(candidates)} candidates")

    if candidates.empty:
        raise ValueError("No candidates generated")

    # Check if similarity scores are available for hard negative mining
    has_similarity = "similarity" in candidates.columns and candidates["similarity"].notna().any()
    if hard_negative_mining and has_similarity:
        logger.info("Hard negative mining enabled with similarity scores")
    elif hard_negative_mining:
        logger.info("Hard negative mining requested but no similarity scores in candidates")

    # Cap pairs per cluster (preserve similarity column)
    if max_pairs_per_cluster > 0:
        before = len(candidates)
        # Save similarity before capping (which only keeps id1, id2)
        similarity_backup = None
        if has_similarity:
            similarity_backup = candidates[["id1", "id2", "similarity"]].copy()

        candidates = _cap_pairs_per_cluster(candidates, max_per_cluster=max_pairs_per_cluster)

        # Restore similarity scores
        if similarity_backup is not None and not candidates.empty:
            candidates = candidates.merge(
                similarity_backup, on=["id1", "id2"], how="left"
            )

        if len(candidates) != before:
            logger.info(f"Capped to {max_pairs_per_cluster} per cluster: {before} -> {len(candidates)}")

    # Remove excluded pairs
    if exclude_pairs is not None and not exclude_pairs.empty:
        before = len(candidates)
        candidates = drop_overlapping_pairs(candidates, exclude_pairs=exclude_pairs)
        if before != len(candidates):
            logger.info(f"Removed {before - len(candidates)} excluded pairs")

    if candidates.empty:
        raise ValueError("No candidates after filtering")

    # Order candidates by similarity strata if hard negative mining enabled
    if hard_negative_mining and has_similarity:
        candidates = _order_candidates_by_similarity_strata(candidates, n_strata=5)
        logger.info("Ordered candidates by 5 similarity strata for diverse labeling")

    # Label with progress saving
    matcher = LLMBasedMatcher()

    labeled = _label_iteratively(
        df_left=df_left,
        df_right=df_right,
        candidates=candidates,
        matcher=matcher,
        chat_model=chat_model,
        id_column=id_column,
        target_size=target_size,
        target_positives=target_positives,
        progress_path=labeled_path,
        batch_size=label_batch_size,
        preserve_similarity=hard_negative_mining and has_similarity,
        max_workers=label_max_workers,
    )

    if labeled.empty:
        raise ValueError("No pairs labeled")

    # Apply hard negative selection if enabled
    if hard_negative_mining and "similarity" in labeled.columns:
        labeled = _select_balanced_set_with_hard_negatives(
            labeled,
            target_size,
            target_positives,
            hard_negative_threshold=hard_negative_threshold,
            min_hard_negative_ratio=min_hard_negative_ratio,
        )
    else:
        labeled = _select_balanced_set(labeled, target_size, target_positives)

    # Save to cache (drop similarity column for storage - it's only used for selection)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_cols = ["id1", "id2", "label"]
    labeled[save_cols].to_csv(cache_path, index=False)
    labeled[save_cols].to_csv(latest_cache_path, index=False)

    # Save metadata
    try:
        latest_meta_path.write_text(json.dumps({
            "left": left_name,
            "right": right_name,
            "hash_suffix": config_hash or "",
            "cache_path": str(cache_path),
            "candidates_path": str(candidates_path),
            "labeled_path": str(labeled_path),
            "blocker_spec": blocker_spec,
            "hard_negative_mining": hard_negative_mining,
        }, indent=2, sort_keys=True, default=str))
    except Exception:
        pass

    n_pos = (labeled["label"].astype(str).str.upper() == "TRUE").sum()
    n_neg = (labeled["label"].astype(str).str.upper() == "FALSE").sum()
    logger.info(f"Training set: {len(labeled)} pairs ({n_pos} positive, {n_neg} negative)")

    return labeled[save_cols]


def _create_default_blockers(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    id_column: str,
    chat_model,
    *,
    include_embedding: bool = True,
    embedding_threshold: float = 0.9,
    embedding_top_k: int = 3,
) -> List[Any]:
    """Create default blockers using LLM-selected columns.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Datasets to block
    id_column : str
        ID column name
    chat_model
        LangChain chat model for selecting blocking columns
    include_embedding : bool
        If True, include EmbeddingBlocker for semantic similarity matching.
        This is recommended for entity resolution tasks.
    embedding_threshold : float
        Minimum similarity threshold for embedding blocker (0.0-1.0)
    embedding_top_k : int
        Number of top similar candidates to return per record
    """
    from ..entitymatching.blocking import TokenBlocker, SortedNeighbourhoodBlocker
    from .entity_resolution import select_blocking_columns, parse_blocking_strategy

    # Get blocking strategies from LLM (may include combined like "name+developer")
    blocking_strategies = select_blocking_columns(df_left, df_right, chat_model, id_column)

    if not blocking_strategies:
        raise ValueError("No suitable blocking columns found")

    logger.info(f"LLM selected blocking strategies: {blocking_strategies}")

    # Extract unique columns from all strategies for embedding blocker
    all_cols = []
    for strategy in blocking_strategies:
        all_cols.extend(parse_blocking_strategy(strategy))
    unique_cols = list(dict.fromkeys(all_cols))  # Preserve order, remove duplicates

    blockers = []

    # Add embedding blocker using columns from strategies
    if include_embedding:
        try:
            from ..entitymatching.blocking import EmbeddingBlocker
            embed_cols = unique_cols[:3]  # Use up to 3 unique columns
            blockers.append(EmbeddingBlocker(
                df_left,
                df_right,
                text_cols=embed_cols,
                id_column=id_column,
                threshold=embedding_threshold,
                top_k=embedding_top_k,
                index_backend="sklearn",
            ))
            logger.info(f"Added EmbeddingBlocker on {embed_cols} with threshold={embedding_threshold}, top_k={embedding_top_k}")
        except Exception as e:
            logger.warning(f"EmbeddingBlocker failed: {e}")

    # Add token blocker on the primary column from the first strategy
    if blocking_strategies:
        primary_cols = parse_blocking_strategy(blocking_strategies[0])
        primary_col = primary_cols[0]  # Use first column from first strategy
        try:
            blockers.append(TokenBlocker(
                df_left, df_right, column=primary_col, id_column=id_column, min_token_len=4, num_tokens=2, ngram_type="word"
            ))
            logger.info(f"Added TokenBlocker on '{primary_col}'")
        except Exception as e:
            logger.warning(f"TokenBlocker on {primary_col} failed: {e}")

    # Add sorted neighbourhood on primary column
    if unique_cols:
        try:
            blockers.append(SortedNeighbourhoodBlocker(
                df_left, df_right, key=unique_cols[0], id_column=id_column, window=13
            ))
        except Exception as e:
            logger.warning(f"SortedNeighbourhoodBlocker failed: {e}")

    if not blockers:
        raise ValueError("Failed to create any blockers")

    return blockers


def augment_training_set_with_disagreements(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
    training_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    disagreement_pairs: pd.DataFrame,
    chat_model,
    output_dir: Path,
    *,
    id_column: str = "id",
    max_new_labels: int = 500,
    label_batch_size: int = 25,
    label_max_workers: int = 10,
    target_positives_for_labeling: Optional[int] = None,
    progress_name: str = "active_learning",
    cache_suffix: Optional[str] = None,
) -> tuple[pd.DataFrame, bool]:
    """
    Augment training set using active learning with matcher disagreement pairs.

    This function labels disagreement pairs (where matchers disagree) and adds
    them to the training set. These pairs are the most informative for training.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Source datasets
    left_name, right_name : str
        Dataset names (used in filenames)
    training_set : pd.DataFrame
        Current training set with columns [id1, id2, label]
    validation_set : pd.DataFrame
        Validation set to exclude from labeling
    disagreement_pairs : pd.DataFrame
        Pairs where matchers disagree, with columns [id1, id2].
        These are prioritized for labeling as they're most informative.
    chat_model : BaseChatModel
        LangChain chat model for labeling
    output_dir : Path
        Directory to save augmented training set
    id_column : str
        ID column name
    max_new_labels : int
        Maximum number of new pairs to label
    label_batch_size : int
        Batch size for labeling
    label_max_workers : int
        Number of parallel LLM labeling workers. Use 1 for serial labeling.
    target_positives_for_labeling : int, optional
        Target positives passed to the iterative labeler. Defaults to labeling
        all candidates without early positive stopping. Use 0 when mining likely
        negatives.
    progress_name : str
        Name used in the progress cache filename.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        (augmented_training_set, was_augmented)
        - augmented_training_set: The training set (possibly augmented)
        - was_augmented: True if additional examples were labeled
    """
    output_dir = Path(output_dir)

    if disagreement_pairs is None or disagreement_pairs.empty:
        logger.warning("No disagreement pairs provided, cannot augment training set")
        return training_set, False

    # Combine existing training set with validation set for exclusion
    exclude_pairs = pd.concat(
        [training_set[["id1", "id2"]], validation_set[["id1", "id2"]]],
        ignore_index=True
    ).drop_duplicates()

    # Filter disagreement pairs to exclude already-labeled ones
    candidates = disagreement_pairs[["id1", "id2"]].drop_duplicates()
    candidates = drop_overlapping_pairs(candidates, exclude_pairs=exclude_pairs)

    if candidates.empty:
        logger.warning("No disagreement pairs remaining after filtering")
        return training_set, False

    # Cap candidates to max_new_labels while preserving acquisition ranking.
    if len(candidates) > max_new_labels:
        candidates = candidates.head(max_new_labels).reset_index(drop=True)

    logger.info(
        f"Active learning: labeling {len(candidates)} candidate pairs "
        f"(excluding {len(exclude_pairs)} existing pairs)"
    )

    try:
        # Label candidates
        matcher = LLMBasedMatcher()

        # Set up progress path for resumption
        training_prefix = _training_file_prefix(cache_suffix)
        progress_path = output_dir / f"{training_prefix}_{progress_name}_{left_name}_{right_name}.csv"
        label_target_positives = (
            len(candidates)
            if target_positives_for_labeling is None
            else max(0, min(int(target_positives_for_labeling), len(candidates)))
        )

        additional_labeled = _label_iteratively(
            df_left=df_left,
            df_right=df_right,
            candidates=candidates,
            matcher=matcher,
            chat_model=chat_model,
            id_column=id_column,
            target_size=len(candidates),  # Label all candidates up to max
            target_positives=label_target_positives,
            progress_path=progress_path,
            batch_size=label_batch_size,
            max_workers=label_max_workers,
        )

        if additional_labeled.empty:
            logger.warning("No additional pairs labeled during active learning")
            return training_set, False

        # Merge with existing training set
        # additional_labeled pairs were already filtered to exclude training_set pairs,
        # so no duplicates should occur. Just concatenate without deduplication to
        # preserve any intentional duplicates in the original training set.
        combined = pd.concat([training_set, additional_labeled], ignore_index=True)

        # Save the combined training set
        combined_path = output_dir / f"{training_prefix}_{left_name}_{right_name}_latest.csv"
        combined.to_csv(combined_path, index=False)

        new_pos = (combined["label"].astype(str).str.upper() == "TRUE").sum()
        new_neg = (combined["label"].astype(str).str.upper() == "FALSE").sum()

        logger.info(
            f"Active learning augmented training set for {left_name} <-> {right_name}: "
            f"{len(combined)} pairs ({new_pos} positive, {new_neg} negative)"
        )

        # Clean up progress file on success
        if progress_path.exists():
            try:
                progress_path.unlink()
            except Exception:
                pass

        return combined, True

    except Exception as e:
        logger.error(f"Error in active learning augmentation: {e}")
        return training_set, False


def run_active_learning(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
    training_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    chat_model,
    output_dir: Path,
    *,
    candidates: Optional[pd.DataFrame] = None,
    blocker_spec: Optional[Dict[str, Any]] = None,
    id_column: str = "id",
    target_positives: int = 100,
    target_negatives: Optional[int] = None,
    max_total_labels: int = 500,
    labels_per_iteration: int = 100,
    max_candidates: int = 10000,
    max_iterations: int = 10,
    low_label_patience: int = 4,
    low_label_threshold: int = 10,
    label_batch_size: int = 25,
    label_max_workers: int = 10,
    ensemble_score_workers: int = 5,
    targeted_pool_multiplier: int = 2,
    cache_suffix: Optional[str] = None,
) -> tuple[pd.DataFrame, bool, Dict[str, Any]]:
    """
    Run active learning loop to augment training set using matcher disagreements.

    This function iteratively:
    1. Trains matchers on current training set
    2. Runs top 2 matchers on candidate pairs
    3. Finds pairs where matchers disagree
    4. Labels disagreement pairs and adds to training set
    5. Repeats until targets reached or max_total_labels exhausted

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Source datasets
    left_name, right_name : str
        Dataset names (used in filenames)
    training_set : pd.DataFrame
        Current training set with columns [id1, id2, label]
    validation_set : pd.DataFrame
        Validation set for matcher evaluation (also excluded from labeling)
    chat_model : BaseChatModel
        LangChain chat model for labeling
    output_dir : Path
        Directory to save augmented training set
    candidates : pd.DataFrame, optional
        Pre-generated candidate pairs (e.g., from FAISS). Must have columns [id1, id2].
        If provided, blocker_spec is ignored.
    blocker_spec : dict, optional
        Blocker specification for generating candidates. Only used if candidates is None.
    id_column : str
        ID column name
    target_positives : int
        Stop when training set has this many positive examples
    target_negatives : int, optional
        Stop when training set has this many negative examples.
        If None, only target_positives is used as stopping condition.
    max_total_labels : int
        Maximum total new pairs to label across all iterations
    labels_per_iteration : int
        Maximum pairs to label per iteration
    max_candidates : int
        Maximum candidates to use (applies to both pre-generated and blocker-generated)
    max_iterations : int
        Maximum number of active learning iterations
    low_label_patience : int
        Stop after this many consecutive iterations add no more than
        low_label_threshold labels. Set to 0 or less to disable.
    low_label_threshold : int
        Per-iteration new-label count treated as low-yield.
    label_batch_size : int
        Batch size for LLM labeling
    label_max_workers : int
        Number of parallel LLM labeling workers. Use 1 for serial labeling.
    ensemble_score_workers : int
        Number of top active-learning matchers to score in parallel.
    targeted_pool_multiplier : int
        Multiplier for the active candidate scoring pool when only positives or
        only negatives are still missing.
    cache_suffix : str, optional
        Cache family suffix, e.g. "excluded_doi", used to avoid mixing
        checkpoints generated with different runtime column policies.

    Returns
    -------
    tuple[pd.DataFrame, bool, Dict[str, Any]]
        (training_set, was_augmented, summary)
        - training_set: The (possibly augmented) training set
        - was_augmented: True if augmentation occurred
        - summary: Dict with statistics about the active learning run
    """
    from .matching_optimization import optimize_matching

    output_dir = Path(output_dir)
    training_set, resumed_progress_count, resumed_progress_paths = _merge_active_progress(
        training_set,
        validation_set,
        output_dir,
        left_name,
        right_name,
        cache_suffix=cache_suffix,
    )
    original_positives = (training_set["label"].astype(str).str.upper() == "TRUE").sum()
    original_negatives = (training_set["label"].astype(str).str.upper() == "FALSE").sum()
    original_size = len(training_set)

    summary: Dict[str, Any] = {
        "left": left_name,
        "right": right_name,
        "original_size": original_size,
        "original_positives": original_positives,
        "original_negatives": original_negatives,
        "iterations": [],
        "low_label_patience": low_label_patience,
        "low_label_threshold": low_label_threshold,
    }
    if resumed_progress_paths:
        summary["resumed_progress_files"] = [str(path) for path in resumed_progress_paths]
        summary["resumed_progress_labels"] = resumed_progress_count
        logger.info(
            f"Resumed active-learning progress for {left_name} <-> {right_name}: "
            f"+{resumed_progress_count} labels from {len(resumed_progress_paths)} file(s)"
        )
        print(
            f"  Resumed active-learning progress: +{resumed_progress_count} labels "
            f"from {len(resumed_progress_paths)} file(s)"
        )

    target_str = f"target: {target_positives} pos"
    if target_negatives is not None:
        target_str += f", {target_negatives} neg"

    logger.info(
        f"Starting active learning for {left_name} <-> {right_name}: "
        f"current training set has {original_size} pairs ({original_positives} pos, {original_negatives} neg), "
        f"{target_str}, max labels: {max_total_labels}"
    )

    # Check if we already have enough positives AND negatives
    positives_satisfied = original_positives >= target_positives
    negatives_satisfied = target_negatives is None or original_negatives >= target_negatives

    if positives_satisfied and negatives_satisfied:
        logger.info(
            f"Already have {original_positives} positives >= {target_positives} and "
            f"{original_negatives} negatives >= {target_negatives or 'N/A'}, skipping"
        )
        summary["stop_reason"] = "targets_met"
        summary["training_set_done"] = True
        return training_set, resumed_progress_count > 0, summary

    # Get candidates: either pre-generated (e.g., from FAISS) or from blocker
    if candidates is not None:
        # Use pre-generated candidates (e.g., from FAISS)
        logger.info(f"Using {len(candidates)} pre-generated candidates")
        full_candidate_pool = candidates.copy()
        original_candidate_count = len(candidates)
        # Limit to max_candidates if needed
        if len(candidates) > max_candidates:
            candidates = candidates.head(max_candidates)
            logger.info(f"Limited to {max_candidates} candidates")
        print(
            f"    Active candidate scoring pool: {len(candidates):,}/{original_candidate_count:,} candidates",
            flush=True,
        )
    elif blocker_spec is not None:
        # Generate candidates from blocker
        blocker = blocker_from_spec(blocker_spec, df_left, df_right, id_column)
        targeted_limit = max_candidates * max(1, int(targeted_pool_multiplier or 1))
        full_candidate_pool = collect_candidates_from_blockers([blocker], max_candidates=targeted_limit)
        original_candidate_count = len(full_candidate_pool)
        candidates = full_candidate_pool.head(max_candidates).reset_index(drop=True)
    else:
        logger.warning("No candidates or blocker_spec provided")
        summary["stop_reason"] = "no_candidates"
        return training_set, False, summary

    if candidates.empty:
        logger.warning("No candidates available for active learning")
        summary["stop_reason"] = "no_candidates"
        return training_set, False, summary

    summary["candidates_generated"] = len(candidates)
    summary["candidate_pool_available"] = len(full_candidate_pool)
    logger.info(f"Generated {len(candidates)} candidates for disagreement analysis")

    current_training = training_set.copy()
    total_labeled = 0
    was_augmented = resumed_progress_count > 0
    base_targeted_multiplier = max(1, int(targeted_pool_multiplier or 1))
    adaptive_targeted_multipliers = {
        "negative": base_targeted_multiplier,
        "positive": base_targeted_multiplier,
    }
    low_label_streak = 0

    bootstrap_min_positives = min(target_positives, 10)
    bootstrap_min_negatives = min(target_negatives or 0, 10)
    current_positives = int((current_training["label"].astype(str).str.upper() == "TRUE").sum())
    current_negatives = int((current_training["label"].astype(str).str.upper() == "FALSE").sum())
    needs_bootstrap = (
        current_positives < bootstrap_min_positives
        or (
            target_negatives is not None
            and current_negatives < bootstrap_min_negatives
        )
    )
    if needs_bootstrap:
        labels_remaining = max_total_labels - total_labeled
        bootstrap_labels = min(
            labels_per_iteration,
            labels_remaining,
            max(
                50,
                (bootstrap_min_positives - min(current_positives, bootstrap_min_positives))
                + (
                    bootstrap_min_negatives
                    - min(current_negatives, bootstrap_min_negatives)
                ),
            ),
        )
        bootstrap_labels = max(0, bootstrap_labels)
        if bootstrap_labels > 0:
            if target_negatives is None:
                bootstrap_target_positives = bootstrap_labels
            elif current_positives >= bootstrap_min_positives:
                bootstrap_target_positives = 0
            elif current_negatives >= bootstrap_min_negatives:
                bootstrap_target_positives = bootstrap_labels
            else:
                bootstrap_target_positives = min(
                    bootstrap_labels,
                    max(bootstrap_min_positives, bootstrap_labels // 2),
                )
            print(
                f"    Bootstrap active training: current seed has "
                f"{current_positives} positives, {current_negatives} negatives; "
                f"labeling up to {bootstrap_labels} non-validation FAISS candidates",
                flush=True,
            )
            bootstrapped, bootstrap_augmented = augment_training_set_with_disagreements(
                df_left=df_left,
                df_right=df_right,
                left_name=left_name,
                right_name=right_name,
                training_set=current_training,
                validation_set=validation_set,
                disagreement_pairs=full_candidate_pool,
                chat_model=chat_model,
                output_dir=output_dir,
                id_column=id_column,
                max_new_labels=bootstrap_labels,
                label_batch_size=label_batch_size,
                label_max_workers=label_max_workers,
                target_positives_for_labeling=bootstrap_target_positives,
                progress_name="active_bootstrap",
                cache_suffix=cache_suffix,
            )
            if bootstrap_augmented:
                new_labels = len(bootstrapped) - len(current_training)
                total_labeled += max(0, new_labels)
                current_training = bootstrapped
                was_augmented = True
                current_positives = int((current_training["label"].astype(str).str.upper() == "TRUE").sum())
                current_negatives = int((current_training["label"].astype(str).str.upper() == "FALSE").sum())
                summary["bootstrap_labels"] = int(max(0, new_labels))
                summary["bootstrap_positives"] = current_positives
                summary["bootstrap_negatives"] = current_negatives
                print(
                    f"    Bootstrap result: {len(current_training)} pairs "
                    f"({current_positives} pos, {current_negatives} neg)",
                    flush=True,
                )

    def _new_label_rows(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
        if after is None or after.empty or len(after) <= len(before):
            return pd.DataFrame(columns=["id1", "id2", "label"])
        return after.iloc[len(before):].copy()

    def _maybe_expand_targeted_pool(
        direction: str,
        new_rows: pd.DataFrame,
        iteration_summary: Dict[str, Any],
    ) -> None:
        if new_rows.empty or "label" not in new_rows.columns:
            return

        desired_label = "FALSE" if direction == "negative" else "TRUE"
        desired_count = int((new_rows["label"].astype(str).str.upper() == desired_label).sum())
        total_count = len(new_rows)
        desired_ratio = desired_count / total_count if total_count else 0.0
        iteration_summary[f"{direction}_target_yield"] = desired_ratio
        iteration_summary[f"{direction}_target_hits"] = desired_count

        if desired_ratio >= 0.5:
            return

        old_multiplier = adaptive_targeted_multipliers[direction]
        old_pool_size = min(len(full_candidate_pool), max_candidates * old_multiplier)
        if old_pool_size >= len(full_candidate_pool):
            iteration_summary[f"{direction}_target_pool_expanded"] = False
            return

        new_multiplier = old_multiplier * 2
        new_pool_size = min(len(full_candidate_pool), max_candidates * new_multiplier)
        adaptive_targeted_multipliers[direction] = new_multiplier
        iteration_summary[f"{direction}_target_pool_expanded"] = True
        iteration_summary[f"{direction}_target_pool_multiplier_next"] = new_multiplier
        iteration_summary[f"{direction}_target_pool_next"] = new_pool_size

        print(
            f"    Targeted {direction} yield: {desired_count}/{total_count} "
            f"({desired_ratio:.1%}) < 50%; next {direction} search pool will use "
            f"{new_pool_size:,}/{len(full_candidate_pool):,} candidates "
            f"(multiplier={new_multiplier}).",
            flush=True,
        )

    for iteration in range(max_iterations):
        current_positives = (current_training["label"].astype(str).str.upper() == "TRUE").sum()
        current_negatives = (current_training["label"].astype(str).str.upper() == "FALSE").sum()

        # Check stopping conditions - need BOTH positives AND negatives targets met
        positives_met = current_positives >= target_positives
        negatives_met = target_negatives is None or current_negatives >= target_negatives

        if positives_met and negatives_met:
            logger.info(
                f"Reached targets: {current_positives} pos >= {target_positives}, "
                f"{current_negatives} neg >= {target_negatives or 'N/A'}"
            )
            print(
                f"    Reached targets for {left_name} <-> {right_name}: "
                f"{current_positives}/{target_positives} positives, "
                f"{current_negatives}/{target_negatives or 'N/A'} negatives"
            )
            summary["stop_reason"] = "targets_met"
            summary["training_set_done"] = True
            break
        if total_labeled >= max_total_labels:
            logger.info(f"Reached max total labels: {total_labeled} >= {max_total_labels}")
            summary["stop_reason"] = "max_total_labels"
            break

        logger.info(f"\n--- Active Learning Iteration {iteration + 1} ---")
        logger.info(
            f"Current training set: {len(current_training)} pairs "
            f"({current_positives} pos, {current_negatives} neg)"
        )
        print(
            f"\n    --- Active Learning Iteration {iteration + 1}: {left_name} <-> {right_name} ---"
        )
        print(
            f"    Current: {current_positives}/{target_positives} positives, "
            f"{current_negatives}/{target_negatives or 'N/A'} negatives"
        )

        iteration_start_size = len(current_training)
        iteration_summary = {"iteration": iteration + 1}

        # Step 1: Train matchers on current training set
        logger.info("Training matchers on current training set...")
        try:
            match_opt = optimize_matching(
                df_left=df_left,
                df_right=df_right,
                validation_set=validation_set,
                id_column=id_column,
                training_set=current_training,
                include_rule_based=True,
                include_ml_based=True,
                include_llm_based=False,
            )
            all_results = match_opt.get("all_results", [])
            all_artifacts = match_opt.get("all_artifacts", [])  # Artifacts already sorted by F1
            if isinstance(all_results, pd.DataFrame):
                results_list = all_results.to_dict("records")
            else:
                results_list = list(all_results) if all_results else []
        except Exception as e:
            logger.warning(f"Error training matchers: {e}")
            summary["stop_reason"] = "matcher_training_failed"
            break

        if len(results_list) < 2:
            logger.warning("Not enough matcher results for disagreement analysis")
            summary["stop_reason"] = "insufficient_matchers"
            break

        # Select top 5 matchers by F1 score for ensemble disagreement detection
        n_ensemble = min(5, len(results_list))
        paired = list(zip(results_list, all_artifacts or [{}] * len(results_list)))
        # Sort by best_f1 descending
        paired_sorted = sorted(paired, key=lambda x: x[0].get("best_f1", 0), reverse=True)
        top_results = [r for r, a in paired_sorted[:n_ensemble]]
        top_artifacts = [a for r, a in paired_sorted[:n_ensemble]]
        logger.info(f"Using top {n_ensemble} matchers by F1 for ensemble disagreement")

        top_matchers = [r.get("matcher") for r in top_results]
        iteration_summary["top_matchers"] = top_matchers
        logger.info(f"Selected matchers: {top_matchers}")

        positives_needed = max(target_positives - current_positives, 0)
        negatives_needed = max((target_negatives or 0) - current_negatives, 0)
        labels_remaining = max_total_labels - total_labeled
        labels_this_iteration = min(labels_per_iteration, labels_remaining)
        negative_mining_mode = positives_needed == 0 and negatives_needed > 0
        positive_mining_mode = positives_needed > 0 and negatives_met
        targeted_mining_mode = negative_mining_mode or positive_mining_mode

        scoring_candidates = candidates
        if targeted_mining_mode:
            direction = "negative" if negative_mining_mode else "positive"
            targeted_multiplier = adaptive_targeted_multipliers[direction]
            targeted_limit = max_candidates * targeted_multiplier
            scoring_candidates = full_candidate_pool.head(targeted_limit).reset_index(drop=True)
            iteration_summary["targeted_direction"] = direction
            iteration_summary["targeted_pool_multiplier"] = targeted_multiplier
            if len(scoring_candidates) > len(candidates):
                print(
                    f"    Targeted {direction} search pool: "
                    f"{len(scoring_candidates):,}/{len(full_candidate_pool):,} candidates "
                    f"(base={len(candidates):,})",
                    flush=True,
                )

        iteration_summary["scored_candidates"] = len(scoring_candidates)

        # Step 2: Run matchers on candidates
        def _score_active_matcher(item: tuple[int, Dict[str, Any], Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            _, result, artifacts = item
            matcher_name = result.get("matcher", "")
            thr = result.get("threshold", 0.5)
            f1 = result.get("f1", 0)

            try:
                corr = _run_matcher_on_candidates(
                    df_left=df_left,
                    df_right=df_right,
                    candidates=scoring_candidates,
                    matcher_config=result,
                    matcher_artifacts=artifacts,  # Pass artifacts for ML matchers
                    id_column=id_column,
                )
                if corr is not None and not corr.empty:
                    logger.info(f"  {matcher_name}: scored {len(corr)} pairs")
                    return {
                        "matcher": matcher_name,
                        "f1": f1,
                        "threshold": thr,
                        "correspondences": corr,
                    }
            except Exception as e:
                logger.warning(f"Error running {matcher_name}: {e}")
            return None

        correspondences_list = []
        score_jobs = list(enumerate(zip(top_results, top_artifacts)))
        score_jobs = [(idx, result, artifacts) for idx, (result, artifacts) in score_jobs]
        ml_score_jobs = [
            job
            for job in score_jobs
            if job[1].get("matcher") == "MLBasedMatcher"
            and job[2].get("classifier") is not None
            and job[2].get("feature_extractor") is not None
        ]
        ml_score_job_indexes = {ml_job[0] for ml_job in ml_score_jobs}
        other_score_jobs = [
            job for job in score_jobs if job[0] not in ml_score_job_indexes
        ]
        scored_by_index: Dict[int, Dict[str, Any]] = {}
        job_by_index = {idx: (result, artifacts) for idx, result, artifacts in score_jobs}

        if ml_score_jobs:
            print(
                f"    Scoring active ML ensemble: {len(ml_score_jobs)} classifiers "
                f"with shared feature extraction",
                flush=True,
            )
            ml_correspondences = _score_ml_ensemble_on_candidates(
                df_left=df_left,
                df_right=df_right,
                candidates=scoring_candidates,
                ml_jobs=ml_score_jobs,
                id_column=id_column,
            )
            for idx, corr in ml_correspondences.items():
                if corr is None or corr.empty:
                    continue
                result, _ = job_by_index[idx]
                matcher_name = result.get("matcher", "")
                logger.info(f"  {matcher_name}: scored {len(corr)} pairs")
                scored_by_index[idx] = {
                    "matcher": matcher_name,
                    "f1": result.get("f1", 0),
                    "threshold": result.get("threshold", 0.5),
                    "correspondences": corr,
                }

        score_workers = max(1, min(int(ensemble_score_workers or 1), len(other_score_jobs)))

        if other_score_jobs and score_workers > 1:
            print(
                f"    Scoring remaining active ensemble: {len(other_score_jobs)} "
                f"matchers with {score_workers} workers",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=score_workers) as executor:
                futures = {
                    executor.submit(_score_active_matcher, job): job[0]
                    for job in other_score_jobs
                }
                for future in futures:
                    result = future.result()
                    if result is not None:
                        scored_by_index[futures[future]] = result
        else:
            for job in other_score_jobs:
                result = _score_active_matcher(job)
                if result is not None:
                    scored_by_index[job[0]] = result

        correspondences_list = [
            scored_by_index[idx]
            for idx in sorted(scored_by_index)
        ]

        if len(correspondences_list) < 2:
            logger.warning("Could not run 2 matchers on candidates")
            summary["stop_reason"] = "insufficient_correspondences"
            break

        if negative_mining_mode:
            # Positives are already satisfied; stay in the active candidate pool
            # but select pairs the ensemble currently scores as likely negatives.
            disagreement_pairs = find_likely_negative_active_pairs(
                correspondences_list,
                top_n=n_ensemble,
            )
            iteration_summary["likely_negative_candidates"] = len(disagreement_pairs)
            label_target_positives = 0
            progress_name = "active_negative"
            labels_this_iteration = min(labels_this_iteration, negatives_needed)
            print(
                f"    Selecting likely-negative active candidates: "
                f"{labels_this_iteration} labels from {len(disagreement_pairs)} scored pairs"
            )
        elif positive_mining_mode:
            # Negatives are already satisfied; stay in the active candidate pool
            # but select pairs the ensemble currently scores as likely positives.
            disagreement_pairs = find_likely_positive_active_pairs(
                correspondences_list,
                top_n=n_ensemble,
            )
            iteration_summary["likely_positive_candidates"] = len(disagreement_pairs)
            label_target_positives = min(labels_this_iteration, positives_needed)
            progress_name = "active_positive"
            labels_this_iteration = min(labels_this_iteration, positives_needed)
            print(
                f"    Selecting likely-positive active candidates: "
                f"{labels_this_iteration} labels from {len(disagreement_pairs)} scored pairs"
            )
        else:
            # While positives are still needed, use the standard disagreement
            # acquisition function.
            disagreement_pairs = find_matcher_disagreements(correspondences_list, top_n=2)
            iteration_summary["disagreements"] = len(disagreement_pairs)
            label_target_positives = None
            progress_name = "active_learning"

        if disagreement_pairs.empty or labels_this_iteration <= 0:
            logger.info("No active-learning candidates selected for labeling")
            iteration_summary["new_labels_total"] = 0
            iteration_summary["low_label_streak"] = low_label_streak
            summary["iterations"].append(iteration_summary)
            summary["stop_reason"] = "no_label_candidates"
            break

        augmented, iter_augmented = augment_training_set_with_disagreements(
            df_left=df_left,
            df_right=df_right,
            left_name=left_name,
            right_name=right_name,
            training_set=current_training,
            validation_set=validation_set,
            disagreement_pairs=disagreement_pairs,
            chat_model=chat_model,
            output_dir=output_dir,
            id_column=id_column,
            max_new_labels=labels_this_iteration,
            label_batch_size=label_batch_size,
            label_max_workers=label_max_workers,
            target_positives_for_labeling=label_target_positives,
            progress_name=progress_name,
            cache_suffix=cache_suffix,
        )

        if iter_augmented:
            new_rows = _new_label_rows(current_training, augmented)
            new_labels = len(augmented) - len(current_training)
            total_labeled += new_labels
            current_training = augmented
            was_augmented = True

            new_positives = (current_training["label"].astype(str).str.upper() == "TRUE").sum()
            new_negatives = (current_training["label"].astype(str).str.upper() == "FALSE").sum()
            iteration_summary["new_labels"] = new_labels
            iteration_summary["total_size"] = len(current_training)
            iteration_summary["total_positives"] = new_positives
            iteration_summary["total_negatives"] = new_negatives

            logger.info(
                f"Iteration {iteration + 1}: labeled {new_labels} pairs, "
                f"total: {len(current_training)} ({new_positives} pos, {new_negatives} neg)"
            )

            if negative_mining_mode:
                _maybe_expand_targeted_pool("negative", new_rows, iteration_summary)
            elif positive_mining_mode:
                _maybe_expand_targeted_pool("positive", new_rows, iteration_summary)

            # If positives were needed first but negatives are still missing, use
            # likely-negative candidates from the same active ensemble scores.
            labels_remaining = max_total_labels - total_labeled
            negatives_needed = (target_negatives - new_negatives) if target_negatives else 0
            iteration_label_budget_remaining = max(
                0,
                labels_this_iteration - max(0, len(current_training) - iteration_start_size),
            )

            if (
                not negative_mining_mode
                and negatives_needed > 0
                and labels_remaining > 0
                and iteration_label_budget_remaining > 0
            ):
                negative_sample_size = min(
                    negatives_needed,
                    labels_remaining,
                    iteration_label_budget_remaining,
                )
                logger.info(
                    f"Still need {negatives_needed} negatives. "
                    f"Selecting {negative_sample_size} likely-negative active candidates "
                    f"within the remaining iteration label budget..."
                )

                negative_candidates = find_likely_negative_active_pairs(
                    correspondences_list,
                    top_n=n_ensemble,
                )

                if not negative_candidates.empty:
                    try:
                        augmented_negatives, neg_augmented = augment_training_set_with_disagreements(
                            df_left=df_left,
                            df_right=df_right,
                            left_name=left_name,
                            right_name=right_name,
                            training_set=current_training,
                            validation_set=validation_set,
                            disagreement_pairs=negative_candidates,
                            chat_model=chat_model,
                            output_dir=output_dir,
                            id_column=id_column,
                            max_new_labels=negative_sample_size,
                            label_batch_size=label_batch_size,
                            label_max_workers=label_max_workers,
                            target_positives_for_labeling=0,
                            progress_name="active_negative",
                            cache_suffix=cache_suffix,
                        )

                        if neg_augmented:
                            new_negative_rows = _new_label_rows(current_training, augmented_negatives)
                            new_negative_labels = len(augmented_negatives) - len(current_training)
                            current_training = augmented_negatives
                            total_labeled += new_negative_labels

                            # Update counts
                            new_positives = (current_training["label"].astype(str).str.upper() == "TRUE").sum()
                            new_negatives = (current_training["label"].astype(str).str.upper() == "FALSE").sum()
                            iteration_summary["total_size"] = len(current_training)
                            iteration_summary["total_positives"] = new_positives
                            iteration_summary["total_negatives"] = new_negatives
                            iteration_summary["likely_negative_sample"] = new_negative_labels

                            logger.info(
                                f"Likely-negative active sample: +{new_negative_labels} labels. "
                                f"Total: {len(current_training)} ({new_positives} pos, {new_negatives} neg)"
                            )
                            _maybe_expand_targeted_pool("negative", new_negative_rows, iteration_summary)
                    except Exception as e:
                        logger.warning(f"Error during likely-negative active sampling: {e}")

            # Symmetric case: if negatives are satisfied but positives are still
            # missing, label likely-positive active candidates from the same
            # scored candidate pool.
            labels_remaining = max_total_labels - total_labeled
            positives_needed = max(target_positives - new_positives, 0)
            negatives_met_after = target_negatives is None or new_negatives >= target_negatives
            iteration_label_budget_remaining = max(
                0,
                labels_this_iteration - max(0, len(current_training) - iteration_start_size),
            )

            if (
                not positive_mining_mode
                and positives_needed > 0
                and negatives_met_after
                and labels_remaining > 0
                and iteration_label_budget_remaining > 0
            ):
                positive_sample_size = min(
                    positives_needed,
                    labels_remaining,
                    iteration_label_budget_remaining,
                )
                logger.info(
                    f"Still need {positives_needed} positives and negatives are satisfied. "
                    f"Selecting {positive_sample_size} likely-positive active candidates "
                    f"within the remaining iteration label budget..."
                )

                positive_candidates = find_likely_positive_active_pairs(
                    correspondences_list,
                    top_n=n_ensemble,
                )

                if not positive_candidates.empty:
                    try:
                        augmented_positives, pos_augmented = augment_training_set_with_disagreements(
                            df_left=df_left,
                            df_right=df_right,
                            left_name=left_name,
                            right_name=right_name,
                            training_set=current_training,
                            validation_set=validation_set,
                            disagreement_pairs=positive_candidates,
                            chat_model=chat_model,
                            output_dir=output_dir,
                            id_column=id_column,
                            max_new_labels=positive_sample_size,
                            label_batch_size=label_batch_size,
                            label_max_workers=label_max_workers,
                            target_positives_for_labeling=positive_sample_size,
                            progress_name="active_positive",
                            cache_suffix=cache_suffix,
                        )

                        if pos_augmented:
                            new_positive_rows = _new_label_rows(current_training, augmented_positives)
                            new_positive_labels = len(augmented_positives) - len(current_training)
                            current_training = augmented_positives
                            total_labeled += new_positive_labels

                            new_positives = (current_training["label"].astype(str).str.upper() == "TRUE").sum()
                            new_negatives = (current_training["label"].astype(str).str.upper() == "FALSE").sum()
                            iteration_summary["total_size"] = len(current_training)
                            iteration_summary["total_positives"] = new_positives
                            iteration_summary["total_negatives"] = new_negatives
                            iteration_summary["likely_positive_sample"] = new_positive_labels

                            logger.info(
                                f"Likely-positive active sample: +{new_positive_labels} labels. "
                                f"Total: {len(current_training)} ({new_positives} pos, {new_negatives} neg)"
                            )
                            _maybe_expand_targeted_pool("positive", new_positive_rows, iteration_summary)
                    except Exception as e:
                        logger.warning(f"Error during likely-positive active sampling: {e}")
        else:
            logger.info(f"Iteration {iteration + 1}: no new pairs labeled")
            iteration_summary["new_labels"] = 0

        iteration_new_labels_total = max(0, len(current_training) - iteration_start_size)
        iteration_summary["new_labels_total"] = iteration_new_labels_total
        if low_label_patience > 0 and iteration_new_labels_total <= low_label_threshold:
            low_label_streak += 1
        else:
            low_label_streak = 0
        iteration_summary["low_label_streak"] = low_label_streak

        summary["iterations"].append(iteration_summary)

        if low_label_patience > 0 and low_label_streak >= low_label_patience:
            summary["stop_reason"] = "low_label_yield"
            summary["training_set_done"] = True
            summary["low_label_stop_iteration"] = iteration + 1
            logger.info(
                f"Stopping active learning for {left_name} <-> {right_name}: "
                f"{low_label_streak} consecutive iterations added <= {low_label_threshold} labels"
            )
            print(
                f"    Active learning stopping early for {left_name} <-> {right_name}: "
                f"{low_label_streak} consecutive iterations added <= {low_label_threshold} labels.",
                flush=True,
            )
            break

        # If no new labels, stop
        if not iter_augmented:
            summary["stop_reason"] = "no_new_labels"
            break

    # Final summary
    final_positives = (current_training["label"].astype(str).str.upper() == "TRUE").sum()
    final_negatives = (current_training["label"].astype(str).str.upper() == "FALSE").sum()
    summary["final_size"] = len(current_training)
    summary["final_positives"] = final_positives
    summary["final_negatives"] = final_negatives
    summary["total_new_labels"] = total_labeled
    summary["final_negative_pool_multiplier"] = adaptive_targeted_multipliers["negative"]
    summary["final_positive_pool_multiplier"] = adaptive_targeted_multipliers["positive"]
    summary.setdefault("stop_reason", "max_iterations")

    if was_augmented:
        output_dir.mkdir(parents=True, exist_ok=True)
        training_prefix = _training_file_prefix(cache_suffix)
        current_training.to_csv(
            output_dir / f"{training_prefix}_{left_name}_{right_name}_latest.csv",
            index=False,
        )

    logger.info(
        f"Active learning complete for {left_name} <-> {right_name}: "
        f"{original_size} -> {len(current_training)} pairs, "
        f"{original_positives} -> {final_positives} pos, "
        f"{original_negatives} -> {final_negatives} neg, "
        f"{total_labeled} new labels across {len(summary['iterations'])} iterations"
    )

    return current_training, was_augmented, summary


def _run_matcher_on_candidates(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    candidates: pd.DataFrame,
    matcher_config: Dict[str, Any],
    id_column: str,
    matcher_artifacts: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Run a matcher on candidate pairs and return correspondences.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Source datasets
    candidates : pd.DataFrame
        Candidate pairs with id1, id2 columns
    matcher_config : dict
        Matcher configuration from optimization results
    id_column : str
        ID column name
    matcher_artifacts : dict, optional
        Trained matcher artifacts (classifier, feature_extractor, etc.)

    Returns
    -------
    pd.DataFrame or None
        Correspondences with id1, id2, score columns
    """
    from ..entitymatching import RuleBasedMatcher, MLBasedMatcher
    from ..entitymatching.comparators import StringComparator

    matcher_name = matcher_config.get("matcher", "")
    matcher_artifacts = matcher_artifacts or {}

    if matcher_name == "RuleBasedMatcher":
        comparators = matcher_artifacts.get("comparators")
        if comparators:
            matcher = RuleBasedMatcher()
            return matcher.match(
                df_left=df_left,
                df_right=df_right,
                candidates=candidates,
                id_column=id_column,
                comparators=comparators,
                threshold=0.0,
            )

        # Build comparators from config as a fallback for older artifacts.
        sim_fn = matcher_config.get("string_similarity_function", "jaro_winkler")
        tokenization = matcher_config.get("string_tokenization") or None

        # Get matching columns (shared between datasets, excluding id)
        matching_cols = [
            c for c in df_left.columns
            if c in df_right.columns and c != id_column
        ][:5]

        comparators = [
            StringComparator(col, sim_fn, tokenization=tokenization, preprocess=str.lower)
            for col in matching_cols
        ]

        matcher = RuleBasedMatcher()
        corr = matcher.match(
            df_left=df_left,
            df_right=df_right,
            candidates=candidates,
            id_column=id_column,
            comparators=comparators,
            threshold=0.0,
        )
        return corr

    elif matcher_name == "MLBasedMatcher":
        # Use trained classifier and feature extractor from artifacts
        classifier = matcher_artifacts.get("classifier")
        feature_extractor = matcher_artifacts.get("feature_extractor")

        if classifier is None or feature_extractor is None:
            logger.warning("MLBasedMatcher requires classifier and feature_extractor in artifacts")
            return None

        use_prob = matcher_config.get("use_probabilities", True)

        # MLBasedMatcher requires feature_extractor in constructor
        matcher = MLBasedMatcher(feature_extractor=feature_extractor)
        corr = matcher.match(
            df_left=df_left,
            df_right=df_right,
            candidates=candidates,
            id_column=id_column,
            trained_classifier=classifier,
            threshold=0.0,  # Score all candidates, let disagreement logic use thresholds
            use_probabilities=use_prob,
        )
        return corr

    # For other matchers (e.g., LLMBasedMatcher), log and skip
    logger.warning(f"Matcher type {matcher_name} not supported for active learning")
    return None


def _score_ml_ensemble_on_candidates(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    candidates: pd.DataFrame,
    ml_jobs: List[tuple[int, Dict[str, Any], Dict[str, Any]]],
    id_column: str,
    chunk_size: int = 1000,
) -> Dict[int, pd.DataFrame]:
    """Score ML ensemble members while sharing candidate feature extraction.

    Active learning often selects several trained ML classifiers that were fit on
    the same FeatureExtractor. Rebuilding the pairwise feature matrix for each
    classifier is much slower than the classifier predictions themselves, so this
    helper extracts each chunk once per distinct extractor and fans the feature
    matrix out to every classifier in that group.
    """
    if not ml_jobs or candidates.empty:
        return {}

    left_lookup = df_left.set_index(id_column).to_dict("index")
    left_lookup.update({str(k): v for k, v in list(left_lookup.items())})
    right_lookup = df_right.set_index(id_column).to_dict("index")
    right_lookup.update({str(k): v for k, v in list(right_lookup.items())})

    extractor_groups: Dict[int, Dict[str, Any]] = {}
    for idx, result, artifacts in ml_jobs:
        classifier = artifacts.get("classifier")
        feature_extractor = artifacts.get("feature_extractor")
        if classifier is None or feature_extractor is None:
            logger.warning(
                "Skipping MLBasedMatcher active scoring without classifier or feature_extractor"
            )
            continue
        use_probabilities = result.get("use_probabilities", True)
        if use_probabilities and not hasattr(classifier, "predict_proba"):
            logger.warning(
                "Skipping MLBasedMatcher active scoring without predict_proba support"
            )
            continue

        group = extractor_groups.setdefault(
            id(feature_extractor),
            {"feature_extractor": feature_extractor, "jobs": []},
        )
        group["jobs"].append(
            {
                "idx": idx,
                "result": result,
                "classifier": classifier,
                "use_probabilities": use_probabilities,
            }
        )

    scored: Dict[int, pd.DataFrame] = {}
    for group in extractor_groups.values():
        feature_extractor = group["feature_extractor"]
        group_jobs = group["jobs"]
        result_frames: Dict[int, List[pd.DataFrame]] = {
            job["idx"]: [] for job in group_jobs
        }
        feature_seconds = 0.0
        predict_seconds = 0.0
        rows_featured = 0

        progress_bar = None
        try:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=len(candidates),
                desc="    Active ML features",
                unit="pairs",
                leave=True,
            )
            progress_bar.set_postfix(classifiers=len(group_jobs))
        except ImportError:
            progress_bar = None

        try:
            for start in range(0, len(candidates), chunk_size):
                batch = candidates.iloc[start:start + chunk_size]
                if batch.empty:
                    continue

                try:
                    feature_start = time.perf_counter()
                    try:
                        feature_df = feature_extractor.create_features(
                            df_left,
                            df_right,
                            batch,
                            id_column,
                            labels=None,
                            left_lookup=left_lookup,
                            right_lookup=right_lookup,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Error extracting active ML features for batch {start}: {exc}"
                        )
                        continue
                    feature_seconds += time.perf_counter() - feature_start

                    if feature_df.empty:
                        continue

                    feature_columns = [
                        col for col in feature_df.columns if col not in ["id1", "id2"]
                    ]
                    if not feature_columns:
                        logger.warning("No feature columns found during active ML scoring")
                        continue

                    rows_featured += len(feature_df)

                    for job in group_jobs:
                        classifier = job["classifier"]
                        expected_columns = list(
                            getattr(classifier, "feature_names_in_", feature_columns)
                        )
                        X = feature_df.reindex(columns=expected_columns, fill_value=0.0)

                        predict_start = time.perf_counter()
                        if job["use_probabilities"] and hasattr(classifier, "predict_proba"):
                            try:
                                probas = classifier.predict_proba(X)
                                if probas.shape[1] == 2:
                                    scores = probas[:, 1]
                                else:
                                    scores = np.max(probas, axis=1)
                            except Exception as exc:
                                logger.warning(
                                    f"Error getting active ML probabilities, falling back to predictions: {exc}"
                                )
                                scores = classifier.predict(X).astype(float)
                        else:
                            scores = classifier.predict(X).astype(float)
                        predict_seconds += time.perf_counter() - predict_start

                        scores = np.asarray(scores, dtype=float)
                        keep_mask = scores >= 0.0
                        if not keep_mask.any():
                            continue

                        corr = feature_df.loc[keep_mask, ["id1", "id2"]].copy()
                        corr["score"] = scores[keep_mask]
                        corr["notes"] = f"ml_classifier={type(classifier).__name__}"
                        result_frames[job["idx"]].append(corr)
                finally:
                    if progress_bar is not None:
                        rows_per_second = (
                            rows_featured / feature_seconds
                            if feature_seconds > 0
                            else 0.0
                        )
                        progress_bar.update(len(batch))
                        progress_bar.set_postfix(
                            classifiers=len(group_jobs),
                            rows_s=f"{rows_per_second:.1f}",
                        )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        for job in group_jobs:
            frames = result_frames[job["idx"]]
            if frames:
                scored[job["idx"]] = pd.concat(frames, ignore_index=True)
            else:
                scored[job["idx"]] = pd.DataFrame(columns=["id1", "id2", "score", "notes"])

        rows_per_second = rows_featured / feature_seconds if feature_seconds > 0 else 0.0
        logger.info(
            "Active ML feature-cached scoring: %s classifiers, %s candidate rows, "
            "%.3fs feature extraction (%.1f rows/s), %.3fs prediction",
            len(group_jobs),
            rows_featured,
            feature_seconds,
            rows_per_second,
            predict_seconds,
        )
        print(
            f"    Active ML timing: {rows_featured:,} feature rows in "
            f"{feature_seconds:.1f}s ({rows_per_second:.1f}/s); "
            f"{len(group_jobs)} classifier predictions in {predict_seconds:.3f}s",
            flush=True,
        )

    return scored


def find_matcher_disagreements(
    correspondences_list: List[Dict[str, Any]],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Find pairs where matchers disagree, ranked by prediction variance.

    This function takes correspondences from multiple matchers run on the same
    unlabeled candidate pairs, and finds pairs where they disagree most.
    Disagreement is measured by the variance of prediction scores/probabilities
    across matchers. Higher variance = more disagreement = more informative.

    Parameters
    ----------
    correspondences_list : list of dict
        List of matcher results, each containing:
        - 'matcher': matcher name (str)
        - 'f1': F1 score from validation (float)
        - 'threshold': decision threshold (float)
        - 'correspondences': DataFrame with id1, id2, score columns
    top_n : int
        Number of top matchers to compare (default: 5)

    Returns
    -------
    pd.DataFrame
        Pairs ranked by disagreement (variance), with columns [id1, id2, variance]
    """
    # Filter to results that have correspondences
    valid_results = [
        r for r in correspondences_list
        if r.get("correspondences") is not None
        and not (isinstance(r.get("correspondences"), pd.DataFrame) and r["correspondences"].empty)
    ]

    if len(valid_results) < 2:
        logger.warning(f"Need at least 2 matchers with correspondences, got {len(valid_results)}")
        return pd.DataFrame(columns=["id1", "id2"])

    # Sort by F1 and take top N
    sorted_results = sorted(valid_results, key=lambda x: x.get("f1", 0), reverse=True)[:top_n]

    logger.info(
        f"Finding disagreements between {len(sorted_results)} matchers: "
        f"{[r.get('matcher', 'unknown') for r in sorted_results]}"
    )

    # Collect scores/probabilities for each pair from each matcher
    pair_scores: Dict[tuple, List[float]] = {}

    for result in sorted_results:
        corr = result["correspondences"].copy()
        corr["score"] = pd.to_numeric(corr["score"], errors="coerce")

        for _, row in corr.iterrows():
            pair = (str(row["id1"]), str(row["id2"]))
            score = row["score"]
            if pd.notna(score):
                if pair not in pair_scores:
                    pair_scores[pair] = []
                pair_scores[pair].append(float(score))

    if not pair_scores:
        return pd.DataFrame(columns=["id1", "id2"])

    # Calculate variance for each pair (only if scored by multiple matchers)
    disagreements = []
    for pair, scores in pair_scores.items():
        if len(scores) >= 2:
            variance = float(np.var(scores))
            disagreements.append({
                "id1": pair[0],
                "id2": pair[1],
                "variance": variance,
            })

    if not disagreements:
        return pd.DataFrame(columns=["id1", "id2"])

    # Sort by variance descending (highest disagreement first)
    result_df = pd.DataFrame(disagreements)
    result_df = result_df.sort_values("variance", ascending=False).reset_index(drop=True)

    logger.info(
        f"Found {len(result_df)} pairs with variance, "
        f"top variance: {result_df['variance'].iloc[0]:.4f}, "
        f"median: {result_df['variance'].median():.4f}"
    )

    return result_df


def find_likely_negative_active_pairs(
    correspondences_list: List[Dict[str, Any]],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Find active-learning candidates that the current matcher ensemble considers likely negatives.

    Candidates are still drawn from the active candidate pool. We rank pairs by:
    1. How many top matchers score the pair below their selected threshold
    2. Low mean score across matchers
    3. High variance as an uncertainty tie-breaker
    """
    valid_results = [
        r for r in correspondences_list
        if r.get("correspondences") is not None
        and not (isinstance(r.get("correspondences"), pd.DataFrame) and r["correspondences"].empty)
    ]

    if len(valid_results) < 2:
        logger.warning(f"Need at least 2 matchers with correspondences, got {len(valid_results)}")
        return pd.DataFrame(columns=["id1", "id2"])

    sorted_results = sorted(valid_results, key=lambda x: x.get("f1", 0), reverse=True)[:top_n]
    pair_scores: Dict[tuple, List[float]] = {}
    pair_negative_votes: Dict[tuple, int] = {}

    for result in sorted_results:
        corr = result["correspondences"].copy()
        corr["score"] = pd.to_numeric(corr["score"], errors="coerce")
        try:
            threshold = float(result.get("threshold", 0.5))
        except (TypeError, ValueError):
            threshold = 0.5

        for _, row in corr.iterrows():
            score = row["score"]
            if pd.isna(score):
                continue
            pair = (str(row["id1"]), str(row["id2"]))
            pair_scores.setdefault(pair, []).append(float(score))
            if float(score) < threshold:
                pair_negative_votes[pair] = pair_negative_votes.get(pair, 0) + 1

    likely_negatives = []
    for pair, scores in pair_scores.items():
        if len(scores) < 2:
            continue
        likely_negatives.append({
            "id1": pair[0],
            "id2": pair[1],
            "mean_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "variance": float(np.var(scores)),
            "negative_votes": int(pair_negative_votes.get(pair, 0)),
            "n_scores": int(len(scores)),
        })

    if not likely_negatives:
        return pd.DataFrame(columns=["id1", "id2"])

    result_df = pd.DataFrame(likely_negatives)
    result_df = result_df.sort_values(
        ["negative_votes", "mean_score", "variance"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    logger.info(
        f"Found {len(result_df)} likely-negative active candidates; "
        f"top mean score: {result_df['mean_score'].iloc[0]:.4f}, "
        f"negative votes: {result_df['negative_votes'].iloc[0]}"
    )
    return result_df


def find_likely_positive_active_pairs(
    correspondences_list: List[Dict[str, Any]],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Find active-learning candidates that the current matcher ensemble considers likely positives.

    Candidates are still drawn from the active candidate pool. We rank pairs by:
    1. How many top matchers score the pair at or above their selected threshold
    2. High mean score across matchers
    3. Low variance as a consensus tie-breaker
    """
    valid_results = [
        r for r in correspondences_list
        if r.get("correspondences") is not None
        and not (isinstance(r.get("correspondences"), pd.DataFrame) and r["correspondences"].empty)
    ]

    if len(valid_results) < 2:
        logger.warning(f"Need at least 2 matchers with correspondences, got {len(valid_results)}")
        return pd.DataFrame(columns=["id1", "id2"])

    sorted_results = sorted(valid_results, key=lambda x: x.get("f1", 0), reverse=True)[:top_n]
    pair_scores: Dict[tuple, List[float]] = {}
    pair_positive_votes: Dict[tuple, int] = {}

    for result in sorted_results:
        corr = result["correspondences"].copy()
        corr["score"] = pd.to_numeric(corr["score"], errors="coerce")
        try:
            threshold = float(result.get("threshold", 0.5))
        except (TypeError, ValueError):
            threshold = 0.5

        for _, row in corr.iterrows():
            score = row["score"]
            if pd.isna(score):
                continue
            pair = (str(row["id1"]), str(row["id2"]))
            pair_scores.setdefault(pair, []).append(float(score))
            if float(score) >= threshold:
                pair_positive_votes[pair] = pair_positive_votes.get(pair, 0) + 1

    likely_positives = []
    for pair, scores in pair_scores.items():
        if len(scores) < 2:
            continue
        likely_positives.append({
            "id1": pair[0],
            "id2": pair[1],
            "mean_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "variance": float(np.var(scores)),
            "positive_votes": int(pair_positive_votes.get(pair, 0)),
            "n_scores": int(len(scores)),
        })

    if not likely_positives:
        return pd.DataFrame(columns=["id1", "id2"])

    result_df = pd.DataFrame(likely_positives)
    result_df = result_df.sort_values(
        ["positive_votes", "mean_score", "variance"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    logger.info(
        f"Found {len(result_df)} likely-positive active candidates; "
        f"top mean score: {result_df['mean_score'].iloc[0]:.4f}, "
        f"positive votes: {result_df['positive_votes'].iloc[0]}"
    )
    return result_df


# =============================================================================
# Random Pair Generation
# =============================================================================


def generate_completely_random_pairs(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    n_pairs: int,
    *,
    exclude_pairs: Optional[pd.DataFrame] = None,
    id_column: str = "id",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate random pairs from the cartesian product of two datasets.

    Creates truly random pairs with no similarity filtering. Useful for adding
    diverse negative examples to training sets.

    Parameters
    ----------
    df_left : pd.DataFrame
        Left dataset with id_column
    df_right : pd.DataFrame
        Right dataset with id_column
    n_pairs : int
        Number of random pairs to generate
    exclude_pairs : pd.DataFrame, optional
        Pairs to exclude (e.g., existing training/validation pairs).
        Must have id1, id2 columns.
    id_column : str
        Name of the ID column in both datasets
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Random pairs with columns [id1, id2] (unlabeled)
    """
    rng = np.random.default_rng(random_state)

    left_ids = df_left[id_column].astype(str).tolist()
    right_ids = df_right[id_column].astype(str).tolist()

    if not left_ids or not right_ids:
        return pd.DataFrame(columns=["id1", "id2"])

    # Build exclusion set
    excluded_set: set[tuple[str, str]] = set()
    if exclude_pairs is not None and not exclude_pairs.empty:
        excluded_set = _pairs_set(exclude_pairs)
        # Also add reversed pairs
        excluded_set |= {(id2, id1) for id1, id2 in excluded_set}

    # Generate random pairs with replacement, filtering exclusions
    max_attempts = n_pairs * 10  # Avoid infinite loop
    pairs: List[tuple[str, str]] = []
    attempts = 0

    while len(pairs) < n_pairs and attempts < max_attempts:
        batch_size = min(n_pairs * 2, 10000)
        id1_samples = rng.choice(left_ids, size=batch_size, replace=True)
        id2_samples = rng.choice(right_ids, size=batch_size, replace=True)

        for id1, id2 in zip(id1_samples, id2_samples):
            if (id1, id2) not in excluded_set and (id2, id1) not in excluded_set:
                # Also check we haven't already added this pair
                if (id1, id2) not in pairs:
                    pairs.append((id1, id2))
                    excluded_set.add((id1, id2))
                    if len(pairs) >= n_pairs:
                        break
        attempts += batch_size

    if len(pairs) < n_pairs:
        logger.warning(
            f"Could only generate {len(pairs)} random pairs "
            f"(requested {n_pairs}, max_attempts={max_attempts})"
        )

    result = pd.DataFrame(pairs, columns=["id1", "id2"])
    return result.reset_index(drop=True)


__all__ = [
    # Cache management
    "load_labeled_set_from_cache",
    "save_labeled_set_to_cache",
    # Utilities
    "drop_overlapping_pairs",
    "collect_candidates_from_blockers",
    "generate_completely_random_pairs",
    # Core generation
    "generate_labeled_set",
    # High-level functions
    "load_or_generate_validation_set",
    "load_or_generate_training_set",
    # Active learning
    "run_active_learning",
    "augment_training_set_with_disagreements",
    "find_matcher_disagreements",
    "find_likely_negative_active_pairs",
    "find_likely_positive_active_pairs",
]
