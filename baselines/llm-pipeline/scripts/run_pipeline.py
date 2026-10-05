"""
PyDI Data Integration Pipeline

Usage:
    python run_pipeline.py --data-dir <path> --schema <path> --output-dir <path>

Example:
    python run_pipeline.py \
        --data-dir usecases/input/music/data \
        --schema usecases/input/music/schemamatching/target_schema.json \
        --output-dir scripts/output/music
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from PyDI.pipeline import (
    run_pipeline,
    select_dataset_pairs,
    load_or_generate_similarity_validation_set,
    load_or_generate_similarity_training_set,
    run_active_learning,
    optimize_matching,
    FusionConfig,
    run_data_fusion,
    generate_faiss_candidates,
    generate_training_variants,
    compare_training_variants,
)
from PyDI.pipeline.fusion_validation_generation import (
    generate_fusion_validation_set,
    load_fusion_validation_from_cache,
    convert_to_tabular_format,
    identify_identifying_columns,
    build_entity_groups,
    select_well_known_entities,
)
from PyDI.pipeline.fusion_optimization import (
    run_fusion_case,
    run_all_fusion_cases,
    evaluate_fusion_accuracy,
    compute_source_accuracy_from_validation_set,
    convert_provided_validation_to_csv_format,
)
from PyDI.pipeline.end_to_end_metrics import (
    EndToEndMetrics,
    calculate_structural_metrics,
    generate_end_to_end_report,
    save_end_to_end_report,
)
from PyDI.pipeline.labeling_cost_estimation import save_cost_report
from PyDI.pipeline.normalization_stats import save_normalization_stats
from PyDI.pipeline.fusion_comparison import generate_fusion_comparison_report
from PyDI.pipeline.cluster_cleaning import (
    clean_oversized_clusters,
    save_cluster_cleaning_report,
    save_cluster_cleaning_latex,
)
from PyDI.io.loaders import load_xml
from PyDI.pipeline.reporting import (
    PipelineMetrics,
    create_step_tracker,
    print_source_overview,
    save_source_overview,
    generate_source_overview_table,
    generate_schema_matching_report_for_pipeline,
    save_training_comparison_summary,
    save_cluster_size_report,
)
from PyDI.pipeline.utils import report_correspondences
from PyDI.normalization import load_normalization_spec
from PyDI.normalization.transform import transform_dataframe
from PyDI.utils.columns import drop_excluded_columns, filter_target_schema

from pipeline_utils import (
    PipelineDirectories,
    check_validation_cache,
    check_training_cache,
    check_matching_cache,
    check_fusion_cache,
    load_json_cache,
    save_json_cache,
    normalize_blocking_cache,
    setup_logging,
    pair_key,
    find_training_file,
)

import time

start_time = time.time()

load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


# =============================================================================
# CONFIGURATION - Edit these values to customize the pipeline
# =============================================================================

# LLM Models
SCHEMA_LLM_MODEL = os.getenv("PYDI_SCHEMA_LLM_MODEL", os.getenv("PYDI_LLM_MODEL", "gpt-5.5"))
LABELING_LLM_MODEL = os.getenv("PYDI_LABELING_LLM_MODEL", "gpt-5.2")
FUSION_LLM_MODEL = os.getenv("PYDI_FUSION_LLM_MODEL", SCHEMA_LLM_MODEL)
LLM_REQUEST_TIMEOUT_SECONDS = _env_float("PYDI_LLM_REQUEST_TIMEOUT", 120.0)
LLM_MAX_RETRIES = _env_int("PYDI_LLM_MAX_RETRIES", 2)

# --- Validation Set Generation ---
# Target number of positive (matching) pairs
VALIDATION_TARGET_POSITIVES = 100
# Target number of negative (non-matching) pairs
VALIDATION_TARGET_NEGATIVES = 200
VALIDATION_K = 20                   # Number of FAISS neighbors to consider per query
# (positives_per_query, negatives_per_query)
VALIDATION_NEIGHBORS_PER_QUERY = (1, 3)
VALIDATION_BATCH_SIZE = 3           # Label batch size
VALIDATION_MAX_LABELS = 1_500        # Max LLM labeling calls

# --- Training Set Generation (small = step 2, large = active learning / comparison) ---
# Positives for small training set (step 2 / faiss_small)
TRAINING_SMALL_TARGET_POSITIVES = 34
TRAINING_SMALL_TARGET_NEGATIVES = 66    # Negatives for small training set
# Positives for large training set (active / faiss_large)
TRAINING_LARGE_TARGET_POSITIVES = 200
TRAINING_LARGE_TARGET_NEGATIVES = 400   # Negatives for large training set
TRAINING_K = 20                         # Number of FAISS neighbors to consider
# (positives_per_query, negatives_per_query)
TRAINING_NEIGHBORS_PER_QUERY = (1, 3)
TRAINING_BATCH_SIZE = 3                 # Label batch size
TRAINING_MAX_LABELS = 3_000               # Max LLM labeling calls

# --- FAISS Candidate Generation ---
FAISS_K = 20                        # Number of neighbors per query record

# --- Matcher Optimization ---
MATCHER_THRESHOLDS = [0.3, 0.4, 0.5, 0.6,
                      0.7, 0.8, 0.9]  # Thresholds to evaluate
INCLUDE_RULE_BASED = True           # Include rule-based matchers
INCLUDE_ML_BASED = True             # Include ML-based matchers
INCLUDE_LLM_BASED = False           # Include LLM-based matchers (expensive)

# --- Active Learning ---
ACTIVE_LEARNING_MAX_LABELS = 5000       # Max total labels in active learning
ACTIVE_LEARNING_LABELS_PER_ITER = 100   # Labels per iteration
ACTIVE_LEARNING_MAX_CANDIDATES = int(os.getenv("PYDI_ACTIVE_LEARNING_CANDIDATES", "20000"))  # Max candidates to score
ACTIVE_LEARNING_MAX_ITERATIONS = 30     # Max active learning iterations
ACTIVE_LEARNING_LOW_LABEL_PATIENCE = int(os.getenv("PYDI_ACTIVE_LOW_LABEL_PATIENCE", "4"))
ACTIVE_LEARNING_LOW_LABEL_THRESHOLD = int(os.getenv("PYDI_ACTIVE_LOW_LABEL_THRESHOLD", "10"))
ACTIVE_LEARNING_BATCH_SIZE = 25         # LLM labeling batch size
ACTIVE_ENSEMBLE_WORKERS = int(os.getenv("PYDI_ACTIVE_ENSEMBLE_WORKERS", "5"))
ACTIVE_TARGETED_POOL_MULTIPLIER = int(os.getenv("PYDI_ACTIVE_TARGETED_POOL_MULTIPLIER", "2"))

# --- Training Set Comparison ---
# Random pairs to add as fraction of base set (20%)
COMPARISON_RANDOM_SAMPLE_RATIO = 0.2

# --- Data Fusion ---
FUSION_INCLUDE_SINGLETONS = True    # Include unmatched records in fused output
FUSION_USE_LLM = True               # Use LLM for fusion strategy planning

# --- Fusion Validation ---
FUSION_VALIDATION_N_ENTITIES = 30   # Number of entities to validate
# Entities to sample for well-known selection
FUSION_VALIDATION_SAMPLE_SIZE = 100

# --- Diagnostics ---
DIAGNOSTICS_TOP_EXAMPLES = 10       # Number of examples to show in reports

# --- LLM Labeling ---
LLM_LABELING_WORKERS = int(os.getenv("PYDI_LLM_LABELING_WORKERS", "10"))


# =============================================================================
# CLI Arguments
# =============================================================================

def _resolve_manifest_relative_path(manifest_path: Path, value: str | Path) -> Path:
    """Resolve manifest paths that may be repo-root or manifest-relative."""
    path = Path(value)
    if path.is_absolute():
        return path

    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return REPO_ROOT / path


def _manifest_taxonomy_base_paths(manifest: dict, manifest_path: Path) -> list[str]:
    """Return extra taxonomy roots implied by a DI-Bench manifest."""
    raw_paths = []
    for key in ("taxonomy_base_path", "taxonomy_base_dir"):
        if manifest.get(key):
            raw_paths.append(manifest[key])
    raw_many = manifest.get("taxonomy_base_paths", [])
    if isinstance(raw_many, (str, Path)):
        raw_paths.append(raw_many)
    else:
        raw_paths.extend(raw_many)

    base_case = manifest.get("base_case")
    if base_case:
        raw_paths.append(
            REPO_ROOT / "DI-Bench" / "usecases" / str(base_case) / "input" / "schemamatching"
        )

    paths: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        if raw_path in (None, ""):
            continue
        path = _resolve_manifest_relative_path(manifest_path, raw_path)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(key)
    return paths

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PyDI data integration pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to a DI-Bench manifest JSON. Fills data-dir, schema, test-dir, output-dir, and pairs when omitted.",
    )
    parser.add_argument(
        "--excluded-columns", nargs="*", default=None,
        help="Target-schema columns to exclude from this run. DI-Bench manifests may also define excluded_columns.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to directory containing input CSV/XML files",
    )
    parser.add_argument(
        "--schema", type=str, default=None,
        help="Path to target schema JSON file",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Path to output directory for results",
    )
    parser.add_argument(
        "--test-dir", type=str, default=None,
        help="Path to directory containing held-out test sets",
    )
    parser.add_argument(
        "--fusion-val-generation-mode", type=str, default="llm",
        choices=["llm", "llm_omit", "web", "web_omit", "all"],
        help="Fusion validation mode: llm (LLM only), llm_omit (LLM + omit target), "
             "web (web search), web_omit (web search + omit target), all (generate all 4 variants)",
    )
    parser.add_argument(
        "--fusion-use-test-set", action="store_true", default=False,
        help="Filter fusion validation to entities in the test set",
    )
    parser.add_argument(
        "--fusion-case", type=str, default=None,
        choices=["heuristic", "llm_no_val", "llm_val",
                 "llm_val_stats", "heuristic_stats", "iterative", "all"],
        help="Fusion rule selection strategy: "
             "heuristic (data types only), "
             "llm_no_val (LLM without validation), "
             "llm_val (LLM + validation feedback), "
             "llm_val_stats (LLM + validation + accuracy stats), "
             "heuristic_stats (heuristic + accuracy stats), "
             "iterative (LLM iteratively improving), "
             "all (run all cases for comparison)",
    )
    parser.add_argument(
        "--fusion-iterations", type=int, default=3,
        help="Number of iterations for iterative fusion case (default: 3)",
    )
    parser.add_argument(
        "--fusion-primary-val-mode", type=str, default="llm_omit",
        choices=["llm", "llm_omit", "web", "web_omit"],
        help="Primary validation mode for computing source accuracy stats (default: llm_omit)",
    )
    # Training set construction
    parser.add_argument(
        "--training-set-construction",
        choices=["active", "compare", "faiss"],
        default="active",
        help=(
            "How to construct the training set used by matcher optimization. "
            "'active' expands the FAISS seed set with active learning; "
            "'compare' generates/evaluates all variants; "
            "'faiss' uses only the initial FAISS seed training set. Default: active."
        ),
    )
    parser.add_argument(
        "--compare-training-sets", action="store_true", default=False,
        help="Deprecated alias for --training-set-construction compare.",
    )
    parser.add_argument(
        "--comparison-pairs", type=str, nargs="*", default=None,
        help="Specific dataset pairs to compare. Supports 'left:right', 'left__right', 'left_2_right', "
             "or legacy 'left_right' when names do not contain underscores. If not specified, compares all pairs from step 2.",
    )
    # Provided validation comparison
    parser.add_argument(
        "--compare-provided-validation", action="store_true", default=False,
        help="Run fusion strategies with provided validation_set.xml and compare with auto-generated validation "
             "results when available. Requires --test-dir to point to directory containing validation_set.xml.",
    )
    parser.add_argument(
        "--official-validation-fusion-only", action="store_true", default=False,
        help="Run fusion optimization only with the official provided validation_set.xml. "
             "Skips generated fusion validation sets and implies --compare-provided-validation.",
    )
    args = parser.parse_args()
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text())
        case = manifest.get("case", manifest_path.stem)
        args.data_dir = args.data_dir or manifest.get("data_dir")
        args.schema = args.schema or manifest.get("schema")
        args.test_dir = args.test_dir or manifest.get("fusion_dir")
        args.output_dir = args.output_dir or manifest.get("output_dir") or f"DI-Bench/runs/{case}"
        if args.excluded_columns is None:
            args.excluded_columns = manifest.get("excluded_columns", [])
        args.fusion_alignment_columns = manifest.get("fusion_alignment_columns", [])
        args.schema_gold = manifest.get("schema_gold") or manifest.get("schema_gold_path")
        args.taxonomy_base_paths = _manifest_taxonomy_base_paths(manifest, manifest_path)
        args.entitymatching_tests = {
            f"{pair['left']}:{pair['right']}": pair.get("test_set") or pair.get("test_path")
            for pair in manifest.get("pairs", [])
            if pair.get("left") and pair.get("right") and (pair.get("test_set") or pair.get("test_path"))
        }
        args.entitymatching_tests_strict = bool(manifest.get("entitymatching_tests_strict", False))
        if not args.comparison_pairs:
            args.comparison_pairs = [
                f"{pair['left']}:{pair['right']}"
                for pair in manifest.get("pairs", [])
                if pair.get("left") and pair.get("right")
            ]
    else:
        args.schema_gold = None
        args.taxonomy_base_paths = []
        args.entitymatching_tests = {}
        args.entitymatching_tests_strict = False
    args.excluded_columns = [str(c) for c in (args.excluded_columns or []) if str(c).strip()]
    args.fusion_alignment_columns = [
        str(c) for c in getattr(args, "fusion_alignment_columns", []) if str(c).strip()
    ]

    if args.official_validation_fusion_only:
        args.compare_provided_validation = True
        args.fusion_case = args.fusion_case or "all"

    if args.compare_training_sets:
        args.training_set_construction = "compare"

    missing = [
        name
        for name, value in {
            "--data-dir": args.data_dir,
            "--schema": args.schema,
            "--output-dir": args.output_dir,
        }.items()
        if not value
    ]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


# =============================================================================
# Pipeline Steps
# =============================================================================

def load_runtime_target_schema(args) -> dict:
    """Load the target schema with manifest/runtime exclusions applied."""
    schema_path = Path(args.schema)
    with open(schema_path) as f:
        target_schema = json.load(f)
    target_schema, _ = filter_target_schema(target_schema, args.excluded_columns)
    return target_schema


def runtime_taxonomy_base_paths(args) -> list[str]:
    """Return ordered taxonomy roots for this run."""
    roots = [str(Path(args.schema).parent)]
    roots.extend(str(path) for path in getattr(args, "taxonomy_base_paths", []))

    unique: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def drop_runtime_excluded_columns(args, df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop columns excluded by the current manifest/runtime policy."""
    filtered, dropped = drop_excluded_columns(df, args.excluded_columns)
    if dropped:
        print(f"  Excluded columns from {label}: {', '.join(dropped)}")
    return filtered


def embedding_cache_suffix(args) -> str | None:
    """Return a cache suffix for embeddings generated with runtime exclusions."""
    if not args.excluded_columns:
        return None
    safe_names = [
        "".join(ch if ch.isalnum() else "_" for ch in str(col).strip().lower()).strip("_")
        for col in args.excluded_columns
    ]
    safe_names = [name for name in safe_names if name]
    return f"excluded_noem_{'_'.join(safe_names)}" if safe_names else None


def load_fusion_reference_file(path: str | Path) -> pd.DataFrame:
    """Load official fusion validation/test data from XML, CSV, JSON, or JSONL.

    Some augmented papers files are JSONL payloads with legacy .xml filenames.
    Detect by content instead of trusting the suffix.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        prefix = ""
        while True:
            ch = fh.read(1)
            if not ch:
                break
            if not ch.isspace():
                prefix = ch
                break

    if prefix == "<":
        return load_xml(path, nested_handling="aggregate")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if prefix in {"{", "["}:
        try:
            return pd.read_json(path, lines=(prefix == "{"))
        except ValueError:
            return pd.read_json(path)
    return load_xml(path, nested_handling="aggregate")


def _normalize_entitymatching_labels(labels: pd.Series) -> pd.Series:
    """Normalize boolean/string labels to TRUE/FALSE strings."""
    normalized = labels.astype(str).str.strip().str.upper()
    return normalized.replace({
        "1": "TRUE",
        "0": "FALSE",
        "YES": "TRUE",
        "NO": "FALSE",
        "Y": "TRUE",
        "N": "FALSE",
        "MATCH": "TRUE",
        "NON_MATCH": "FALSE",
        "NONMATCH": "FALSE",
    })


def _normalize_pair_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize id1/id2 columns to strings in a labeled/candidate pair frame."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in ("id1", "id2"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def _read_entitymatching_test_set(test_path: Path) -> pd.DataFrame:
    """Read an entity-matching test set and normalize to id1, id2, label."""
    with open(test_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().lower()

    has_header = any(token in first_line for token in ("id1", "label", "ltable_id", "rtable_id"))
    if has_header:
        test_set = pd.read_csv(test_path)
        test_set.columns = [str(c).lower().strip() for c in test_set.columns]
    else:
        test_set = pd.read_csv(test_path, header=None, names=["id1", "id2", "label"])

    rename_map = {
        "ltable_id": "id1",
        "rtable_id": "id2",
        "left_id": "id1",
        "right_id": "id2",
        "match": "label",
    }
    test_set = test_set.rename(columns={k: v for k, v in rename_map.items() if k in test_set.columns})

    missing = {"id1", "id2", "label"} - set(test_set.columns)
    if missing:
        raise ValueError(f"Entity-matching test set {test_path} missing columns: {sorted(missing)}")

    test_set = test_set[["id1", "id2", "label"]].copy()
    test_set["id1"] = test_set["id1"].astype(str)
    test_set["id2"] = test_set["id2"].astype(str)
    test_set["label"] = _normalize_entitymatching_labels(test_set["label"])
    test_set = test_set[test_set["label"].isin(["TRUE", "FALSE"])].reset_index(drop=True)
    test_set.attrs["source_path"] = str(test_path)
    return test_set


def _fix_entitymatching_pair_order(
    test_set: pd.DataFrame,
    df_left: pd.DataFrame | None,
    df_right: pd.DataFrame | None,
    *,
    id_column: str = "id",
) -> pd.DataFrame:
    """Ensure id1 belongs to the left dataframe and id2 to the right dataframe."""
    if test_set.empty or df_left is None or df_right is None:
        return test_set
    if id_column not in df_left.columns or id_column not in df_right.columns:
        return test_set

    left_ids = set(df_left[id_column].astype(str))
    right_ids = set(df_right[id_column].astype(str))
    sample = test_set.head(100)
    id1_in_left = sample["id1"].isin(left_ids).sum()
    id1_in_right = sample["id1"].isin(right_ids).sum()
    id2_in_left = sample["id2"].isin(left_ids).sum()
    id2_in_right = sample["id2"].isin(right_ids).sum()
    if id1_in_right > id1_in_left and id2_in_left > id2_in_right:
        swapped = test_set.copy()
        swapped["id1"], swapped["id2"] = test_set["id2"].copy(), test_set["id1"].copy()
        return swapped
    return test_set


def load_entitymatching_test_set(
    args,
    left_name: str,
    right_name: str,
    test_sets_dir: Path,
    *,
    df_left: pd.DataFrame | None = None,
    df_right: pd.DataFrame | None = None,
    quiet: bool = False,
) -> pd.DataFrame | None:
    """Load the official EM test set for a pair, preferring manifest paths."""
    manifest_tests = getattr(args, "entitymatching_tests", {}) or {}
    explicit_path = None
    for key in (
        f"{left_name}:{right_name}",
        f"{right_name}:{left_name}",
        f"{left_name}__{right_name}",
        f"{right_name}__{left_name}",
    ):
        if manifest_tests.get(key):
            explicit_path = Path(manifest_tests[key])
            break

    candidate_paths: list[Path] = []
    if explicit_path:
        candidate_paths.append(explicit_path)
    elif getattr(args, "entitymatching_tests_strict", False):
        return None

    test_sets_dir = Path(test_sets_dir)
    for pattern in [
        f"{left_name}_2_{right_name}_test_corner_filled.csv",
        f"{right_name}_2_{left_name}_test_corner_filled.csv",
        f"{left_name}_{right_name}_test_corner_filled.csv",
        f"{right_name}_{left_name}_test_corner_filled.csv",
        f"{left_name}_2_{right_name}_test.csv",
        f"{right_name}_2_{left_name}_test.csv",
        f"{left_name}_{right_name}_test.csv",
        f"{right_name}_{left_name}_test.csv",
    ]:
        candidate_paths.append(test_sets_dir / pattern)

    seen: set[str] = set()
    for test_path in candidate_paths:
        key = str(test_path)
        if key in seen:
            continue
        seen.add(key)
        if not test_path.exists():
            continue
        test_set = _read_entitymatching_test_set(test_path)
        test_set = _fix_entitymatching_pair_order(test_set, df_left, df_right)
        test_set.attrs["source_path"] = str(test_path)
        if not quiet:
            print(f"  Found test set: {test_path.name} ({len(test_set)} pairs)")
        return test_set

    return None


def _normalize_alignment_value(value) -> str:
    """Normalize an evaluation-only alignment key such as DOI."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    lower = text.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lower.startswith(prefix):
            lower = lower[len(prefix):].strip()
            break
    return lower


def _column_matches_alignment(column: str, alignment_columns: list[str]) -> bool:
    normalized = str(column).strip().lower().replace("-", "_").replace(" ", "_")
    for alignment_col in alignment_columns:
        key = str(alignment_col).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized == key or normalized.startswith(f"{key}_") or normalized.endswith(f"_{key}"):
            return True
    return False


def _value_matches_alignment(value, alignment_columns: list[str]) -> bool:
    """Return true when a sampled raw value looks like a requested alignment key."""
    normalized_keys = {
        str(col).strip().lower().replace("-", "_").replace(" ", "_")
        for col in alignment_columns
    }
    if "doi" not in normalized_keys:
        return False

    text = _normalize_alignment_value(value)
    return text.startswith("10.") and "/" in text


def _column_has_alignment_values(series: pd.Series, alignment_columns: list[str]) -> bool:
    """Detect obfuscated alignment columns, e.g. DOI stored as Attribute_2."""
    try:
        values = series.dropna().head(50)
    except Exception:
        return False
    if values.empty:
        return False
    return any(_value_matches_alignment(value, alignment_columns) for value in values)


def build_fusion_alignment_lookup(args) -> dict[str, dict[str, str]]:
    """Build evaluation-only alignment lookup: normalized key -> {source_name: source_id}."""
    alignment_columns = getattr(args, "fusion_alignment_columns", [])
    if not alignment_columns:
        return {}

    lookup: dict[str, dict[str, str]] = {}
    raw_sources = load_raw_sources(Path(args.data_dir))
    for source_name, df in raw_sources.items():
        if "id" not in df.columns:
            continue
        key_cols = [
            col for col in df.columns
            if (
                _column_matches_alignment(str(col), alignment_columns)
                or _column_has_alignment_values(df[col], alignment_columns)
            )
        ]
        if not key_cols:
            continue

        for _, row in df.iterrows():
            source_id = str(row.get("id", "")).strip()
            if not source_id:
                continue
            for key_col in key_cols:
                key = _normalize_alignment_value(row.get(key_col))
                if key:
                    lookup.setdefault(key, {})[source_name] = source_id

    if lookup:
        print(
            f"  Built fusion alignment lookup using {', '.join(alignment_columns)}: "
            f"{len(lookup):,} keys"
        )
    return lookup


def attach_fusion_source_ids_from_alignment(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    args,
    alignment_lookup: dict[str, dict[str, str]],
    label: str,
) -> pd.DataFrame:
    """Attach source_ids for fusion evaluation using private alignment columns."""
    alignment_columns = getattr(args, "fusion_alignment_columns", [])
    if df.empty or raw_df.empty or not alignment_columns or not alignment_lookup:
        return df

    key_cols = [
        col for col in raw_df.columns
        if _column_matches_alignment(str(col), alignment_columns)
    ]
    if not key_cols:
        return df

    source_ids: list[str] = []
    matched = 0
    for idx, _ in df.iterrows():
        source_id_values: list[str] = []
        if idx < len(raw_df):
            raw_row = raw_df.iloc[idx]
            for key_col in key_cols:
                key = _normalize_alignment_value(raw_row.get(key_col))
                if not key:
                    continue
                ids_by_source = alignment_lookup.get(key, {})
                for source_id in ids_by_source.values():
                    if source_id not in source_id_values:
                        source_id_values.append(source_id)
        if source_id_values:
            matched += 1
        source_ids.append(", ".join(source_id_values))

    out = df.copy()
    out["source_ids"] = source_ids
    print(f"  Attached source_ids to {matched}/{len(out)} {label} rows via fusion alignment")
    return out


def ensure_fusion_expected_id(df: pd.DataFrame, label: str, id_column: str = "id") -> pd.DataFrame:
    """Ensure expected fusion data has an ID column for evaluator alignment."""
    if df.empty:
        return df
    if id_column in df.columns and df[id_column].notna().any():
        return df
    if "source_ids" not in df.columns:
        return df

    out = df.copy()

    def _first_source_id(value) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return next((part.strip() for part in text.split(",") if part.strip()), "")

    out[id_column] = out["source_ids"].map(_first_source_id)
    populated = int((out[id_column].astype(str).str.strip() != "").sum())
    print(f"  Derived {id_column} for {populated}/{len(out)} {label} rows from source_ids")
    return out


def find_fusion_reference_path(test_dir: Path, candidates: list[str]) -> Path | None:
    """Return the first existing fusion reference file under a test directory."""
    for candidate in candidates:
        path = test_dir / candidate
        if path.exists():
            return path
    return None


def load_raw_sources(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load raw source files before any transformation.

    Args:
        data_dir: Directory containing data files (*.xml, *.csv)

    Returns:
        Dictionary mapping source names to raw DataFrames
    """
    data_dir = Path(data_dir)
    data_files = list(data_dir.glob("*.xml")) + list(data_dir.glob("*.csv"))

    sources = {}
    for data_path in data_files:
        name = data_path.stem
        if data_path.suffix == ".csv":
            sources[name] = pd.read_csv(data_path)
        elif data_path.suffix == ".xml":
            sources[name] = load_xml(data_path, nested_handling="aggregate")

    return sources


def print_and_save_source_overview(
    args,
    dirs: PipelineDirectories,
    target_columns: int = 9,
) -> dict[str, pd.DataFrame]:
    """Load raw sources, print overview table, and save to file.

    Args:
        args: Command line arguments (needs args.data_dir, args.schema)
        dirs: Pipeline directories
        target_columns: Number of columns in target schema

    Returns:
        Dictionary of raw source DataFrames
    """
    # Determine target schema size
    schema_path = Path(args.schema)
    if schema_path.exists():
        schema = load_runtime_target_schema(args)
        target_columns = len(schema.get("properties", {}))

    # Load raw sources
    sources = load_raw_sources(args.data_dir)
    if args.excluded_columns:
        sources = {
            name: drop_runtime_excluded_columns(args, df, f"source overview {name}")
            for name, df in sources.items()
        }

    if sources:
        # Print to console
        print_source_overview(sources, target_columns)

        # Save to reporting folder
        reporting_dir = dirs.output / "reporting"
        reporting_dir.mkdir(parents=True, exist_ok=True)
        save_source_overview(
            sources,
            reporting_dir,
            target_columns,
            filename="source_overview.csv",
        )
        print(f"  Saved source overview to: {reporting_dir / 'source_overview.csv'}")

    return sources


def step1_schema_matching(args, dirs: PipelineDirectories, llm, track_step):
    """Step 1: Schema Matching (1a) + Normalization (1b)"""
    print("\n=== Step 1: Schema Matching + Normalization ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Schema: {args.schema}")
    if args.excluded_columns:
        print(f"Excluded columns: {', '.join(args.excluded_columns)}")

    results = run_pipeline(
        data_dir=args.data_dir,
        schema_path=args.schema,
        chat_model=llm,
        output_dir=dirs.schema_matching,
        force_rematch=False,
        excluded_columns=args.excluded_columns,
        taxonomy_base_paths=getattr(args, "taxonomy_base_paths", []),
        track_step=track_step,
    )
    _normalize_runtime_ids(results)
    if args.excluded_columns:
        results = {
            name: drop_runtime_excluded_columns(args, df, f"normalized {name}")
            for name, df in results.items()
        }
        _normalize_runtime_ids(results)

    print("\n--- Normalized Datasets ---")
    for name, df in results.items():
        print(f"{name}: {len(df)} rows, {len(df.columns)} columns")

    return results


def _normalize_runtime_ids(results: dict[str, pd.DataFrame], id_column: str = "id") -> None:
    """Normalize runtime record IDs to strings for matching, fusion, and evaluation."""
    for df in results.values():
        if id_column in df.columns:
            df[id_column] = df[id_column].astype(str)


def _split_pair_spec(pair_str: str, available_names: set[str]) -> tuple[str, str] | None:
    """Split a user/manifest pair spec while allowing dataset names with underscores."""
    for separator in (":", "__", "->", ","):
        if separator in pair_str:
            left, right = pair_str.split(separator, 1)
            return left.strip(), right.strip()

    if "_2_" in pair_str:
        left, right = pair_str.split("_2_", 1)
        return left.strip(), right.strip()

    # Legacy format: left_right. Prefer the longest known dataset prefix so
    # names like open_alex are still handled when paired with another source.
    for left in sorted(available_names, key=len, reverse=True):
        prefix = f"{left}_"
        if pair_str.startswith(prefix):
            right = pair_str[len(prefix):]
            if right in available_names:
                return left, right
    for right in sorted(available_names, key=len, reverse=True):
        suffix = f"_{right}"
        if pair_str.endswith(suffix):
            left = pair_str[:-len(suffix)]
            if left in available_names:
                return left, right
    return None


def _parse_comparison_pairs(args, results):
    """Parse --comparison-pairs into list of (left, right) tuples."""
    if not args.comparison_pairs:
        return None

    parsed = []
    available_names = set(results.keys())

    for pair_str in args.comparison_pairs:
        split_pair = _split_pair_spec(pair_str, available_names)
        if split_pair:
            left, right = split_pair
            # Normalize order: larger dataset first (consistent with select_dataset_pairs)
            if left in available_names and right in available_names:
                if len(results[left]) >= len(results[right]):
                    parsed.append((left, right))
                else:
                    parsed.append((right, left))
            else:
                missing = [n for n in [left, right]
                           if n not in available_names]
                print(f"  Warning: dataset(s) not found: {missing}")
        else:
            print(
                f"  Warning: invalid pair format '{pair_str}', expected 'left:right', 'left__right', or 'left_2_right'")

    return parsed if parsed else None


def step2_validation_training(args, dirs: PipelineDirectories, results, llm, track_step):
    """Step 2: Validation & Training Set Generation"""
    print("\n=== Step 2: Validation & Training Set Generation ===")

    # Use custom pairs if specified, otherwise auto-select
    custom_pairs = _parse_comparison_pairs(args, results)
    if custom_pairs:
        pairs = custom_pairs
        print(f"Using custom pairs: {pairs}")
    else:
        pairs = select_dataset_pairs(results)
    validation_sets = {}
    training_sets = {}
    emb_cache_suffix = embedding_cache_suffix(args)

    # Check for pre-computed embeddings in input directory
    input_embeddings_dir = Path(
        args.data_dir).parent / "entitymatching" / "embeddings"
    if input_embeddings_dir.exists():
        if args.excluded_columns:
            print(
                "Ignoring pre-computed input embeddings because excluded columns are active; "
                "regenerating run-local embeddings without excluded columns."
            )
            input_embeddings_dir = None
        else:
            print(f"Found pre-computed embeddings: {input_embeddings_dir}")
    else:
        input_embeddings_dir = None

    val_cached = check_validation_cache(pairs, dirs.validation, cache_suffix=emb_cache_suffix)
    train_cached = check_training_cache(pairs, dirs.training, cache_suffix=emb_cache_suffix)

    with track_step("Step 2: Validation & Training", cached=val_cached and train_cached):
        for left_name, right_name in pairs:
            print(f"\n--- {left_name} <-> {right_name} ---")

            # Validation set
            print(
                "  [Validation] Loading or generating "
                f"(target={VALIDATION_TARGET_POSITIVES + VALIDATION_TARGET_NEGATIVES}, "
                f"positives={VALIDATION_TARGET_POSITIVES})..."
            )
            val_set = load_or_generate_similarity_validation_set(
                df_left=results[left_name],
                df_right=results[right_name],
                left_name=left_name,
                right_name=right_name,
                chat_model=llm,
                output_dir=dirs.validation,
                id_column="id",
                target_size=VALIDATION_TARGET_POSITIVES + VALIDATION_TARGET_NEGATIVES,
                target_positives=VALIDATION_TARGET_POSITIVES,
                k=VALIDATION_K,
                neighbors_per_query=VALIDATION_NEIGHBORS_PER_QUERY,
                batch_size=VALIDATION_BATCH_SIZE,
                query_order="similarity",
                force_regenerate=False,
                generate_guidelines=False,
                retrieval_method="faiss",
                max_labels=VALIDATION_MAX_LABELS,
                input_embeddings_dir=input_embeddings_dir,
                embedding_cache_suffix=emb_cache_suffix,
                cache_suffix=emb_cache_suffix,
                label_max_workers=LLM_LABELING_WORKERS,
            )
            val_set = _normalize_pair_ids(val_set)
            validation_sets[(left_name, right_name)] = val_set
            n_pos = (val_set["label"] == "TRUE").sum()
            n_neg = (val_set["label"] == "FALSE").sum()
            print(
                f"  Validation: {len(val_set)} pairs ({n_pos} pos, {n_neg} neg)")

            # Training set
            print(
                "  [Training] Loading or generating "
                f"(target={TRAINING_SMALL_TARGET_POSITIVES + TRAINING_SMALL_TARGET_NEGATIVES}, "
                f"positives={TRAINING_SMALL_TARGET_POSITIVES})..."
            )
            train_set = load_or_generate_similarity_training_set(
                df_left=results[left_name],
                df_right=results[right_name],
                left_name=left_name,
                right_name=right_name,
                chat_model=llm,
                output_dir=dirs.training,
                id_column="id",
                target_size=TRAINING_SMALL_TARGET_POSITIVES + TRAINING_SMALL_TARGET_NEGATIVES,
                target_positives=TRAINING_SMALL_TARGET_POSITIVES,
                k=TRAINING_K,
                neighbors_per_query=TRAINING_NEIGHBORS_PER_QUERY,
                batch_size=TRAINING_BATCH_SIZE,
                query_order="similarity",
                exclude_pairs=val_set,
                force_regenerate=False,
                generate_guidelines=False,
                retrieval_method="faiss",
                max_labels=TRAINING_MAX_LABELS,
                input_embeddings_dir=input_embeddings_dir,
                embedding_cache_suffix=emb_cache_suffix,
                cache_suffix=emb_cache_suffix,
                label_max_workers=LLM_LABELING_WORKERS,
            )
            train_set = _normalize_pair_ids(train_set)
            training_sets[(left_name, right_name)] = train_set
            n_pos = (train_set["label"] == "TRUE").sum()
            n_neg = (train_set["label"] == "FALSE").sum()
            print(
                f"  Training: {len(train_set)} pairs ({n_pos} pos, {n_neg} neg)")

    return pairs, validation_sets, training_sets


def step3_faiss_candidates(args, dirs: PipelineDirectories, pairs, results, llm, track_step):
    """Step 3: FAISS Candidate Generation"""
    print("\n=== Step 3: FAISS Candidate Generation ===")

    # Check for pre-computed embeddings in input directory
    input_embeddings_dir = Path(
        args.data_dir).parent / "entitymatching" / "embeddings"
    if args.excluded_columns:
        input_embeddings_dir = None
    elif not input_embeddings_dir.exists():
        input_embeddings_dir = None

    faiss_candidates = {}
    emb_cache_suffix = embedding_cache_suffix(args)

    with track_step("Step 3: FAISS Candidate Generation"):
        for left_name, right_name in pairs:
            candidates = generate_faiss_candidates(
                df_left=results[left_name],
                df_right=results[right_name],
                left_name=left_name,
                right_name=right_name,
                id_column="id",
                k=FAISS_K,
                output_dir=dirs.validation,
                chat_model=llm,
                input_embeddings_dir=input_embeddings_dir,
                embedding_cache_suffix=emb_cache_suffix,
            )
            faiss_candidates[(left_name, right_name)] = candidates
            print(f"{left_name} <-> {right_name}: {len(candidates)} candidates")

    return faiss_candidates


def _load_training_comparison_cache(
    comparison_dir: Path,
    training_dir: Path,
    left_name: str,
    right_name: str,
) -> tuple:
    """Load cached training comparison results if available.

    Returns
    -------
    tuple
        (best_training_set, best_variant, cached) or (None, None, False) if not cached
    """
    import json

    # Try both orderings of the pair names
    pair_orderings = [
        (left_name, right_name),
        (right_name, left_name),
    ]

    details_path = None
    actual_left, actual_right = left_name, right_name

    for l, r in pair_orderings:
        candidate_path = comparison_dir / f"{l}_{r}" / "comparison_details.json"
        if candidate_path.exists():
            details_path = candidate_path
            actual_left, actual_right = l, r
            break

    if details_path is None:
        return None, None, False

    try:
        with open(details_path) as f:
            details = json.load(f)

        best_variant = details.get("best_variant")
        if not best_variant:
            return None, None, False

        # Load the best variant's training set CSV - try multiple path patterns
        possible_paths = [
            training_dir / f"similarity_training_{best_variant}_{actual_left}_{actual_right}.csv",
            training_dir / f"similarity_training_{best_variant}_{actual_right}_{actual_left}.csv",
            training_dir / f"training_{actual_left}_{actual_right}_{best_variant}.csv",
            training_dir / f"training_{actual_left}_{actual_right}_augmented.csv",
            training_dir / f"training_{actual_left}_{actual_right}_latest.csv",
        ]

        for variant_path in possible_paths:
            if variant_path.exists():
                best_training_set = pd.read_csv(variant_path, dtype={"label": str})
                return best_training_set, best_variant, True

        return None, None, False

    except Exception as e:
        print(f"  Warning: Failed to load cache: {e}")
        return None, None, False


def step4_training_comparison(
    args,
    dirs: PipelineDirectories,
    pairs,
    results,
    validation_sets,
    training_sets,
    faiss_candidates,
    llm,
    track_step,
):
    """Step 4: Training Set Comparison (optional)

    Generates multiple training set variants (faiss_small, faiss_large, active, *_plus_random)
    and compares their performance on the provided test set (or validation set if no test set).

    Returns the best training sets for each pair.
    """
    if args.training_set_construction != "compare":
        return training_sets

    print("\n=== Step 4: Training Set Comparison ===")

    comparison_dir = dirs.training / "training_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Look for test sets in the entitymatching directory
    test_sets_dir = Path(args.data_dir).parent / "entitymatching"

    best_training_sets = {}

    # Check if all pairs are cached
    all_cached = True
    for (left_name, right_name) in validation_sets.keys():
        _, _, cached = _load_training_comparison_cache(
            comparison_dir, dirs.training, left_name, right_name
        )
        if not cached:
            all_cached = False
            break

    with track_step("Step 4: Training Set Comparison", cached=all_cached):
        for (left_name, right_name), val_set in validation_sets.items():
            print(f"\n--- {left_name} <-> {right_name} ---")

            # Check cache first
            cached_training_set, cached_best_variant, is_cached = _load_training_comparison_cache(
                comparison_dir, dirs.training, left_name, right_name
            )

            if is_cached:
                best_training_sets[(left_name, right_name)] = cached_training_set
                n_pos = (cached_training_set["label"].astype(str).str.upper() == "TRUE").sum()
                n_neg = (cached_training_set["label"].astype(str).str.upper() == "FALSE").sum()
                print(f"  Loaded from cache: {cached_best_variant}")
                print(f"  Training set: {len(cached_training_set)} pairs ({n_pos} pos, {n_neg} neg)")
                continue

            candidates = faiss_candidates.get((left_name, right_name))
            if candidates is None or candidates.empty:
                print("  Skipping: no FAISS candidates")
                best_training_sets[(left_name, right_name)] = training_sets.get(
                    (left_name, right_name), pd.DataFrame()
                )
                continue

            test_set = load_entitymatching_test_set(
                args,
                left_name,
                right_name,
                test_sets_dir,
                df_left=results[left_name],
                df_right=results[right_name],
            )

            if test_set is None:
                print("  No test set found, will evaluate on validation set")

            # Get existing training set from step 2 (to reuse for faiss_small)
            existing_train_set = training_sets.get((left_name, right_name))

            # Generate all variants
            variants = generate_training_variants(
                df_left=results[left_name],
                df_right=results[right_name],
                left_name=left_name,
                right_name=right_name,
                validation_set=val_set,
                faiss_candidates=candidates,
                chat_model=llm,
                output_dir=dirs.training,
                id_column="id",
                small_target_positives=TRAINING_SMALL_TARGET_POSITIVES,
                small_target_negatives=TRAINING_SMALL_TARGET_NEGATIVES,
                large_target_positives=TRAINING_LARGE_TARGET_POSITIVES,
                large_target_negatives=TRAINING_LARGE_TARGET_NEGATIVES,
                random_sample_ratio=COMPARISON_RANDOM_SAMPLE_RATIO,
                existing_training_set=existing_train_set,
                label_max_workers=LLM_LABELING_WORKERS,
            )

            # Compare variants (using test set if available)
            pair_comparison_dir = comparison_dir / f"{left_name}_{right_name}"
            pair_comparison_dir.mkdir(parents=True, exist_ok=True)

            comparison_df, best_variant = compare_training_variants(
                df_left=results[left_name],
                df_right=results[right_name],
                variants=variants,
                validation_set=val_set,
                output_dir=pair_comparison_dir,
                id_column="id",
                test_set=test_set,
                left_name=left_name,
                right_name=right_name,
                entitymatching_dir=test_sets_dir,
            )

            # Use the best variant's training set
            best_training_sets[(left_name, right_name)
                               ] = variants[best_variant].training_set

            print(f"  Best variant: {best_variant}")
            print(f"  Comparison saved to: {pair_comparison_dir}")

    # Save overall summary
    print("\n--- Training Comparison Summary ---")
    for (left_name, right_name), train_set in best_training_sets.items():
        n_pos = (train_set["label"].astype(str).str.upper() == "TRUE").sum()
        n_neg = (train_set["label"].astype(str).str.upper() == "FALSE").sum()
        print(
            f"  {left_name} <-> {right_name}: {len(train_set)} pairs ({n_pos} pos, {n_neg} neg)")

    # Generate Auto vs Provided comparison summary report
    summary_path = save_training_comparison_summary(dirs.output)
    if summary_path:
        print(f"\n  Saved Auto vs Provided summary to: {summary_path}")

    return best_training_sets


def step5_matcher_optimization(
    args,
    dirs: PipelineDirectories,
    pairs,
    results,
    validation_sets,
    training_sets,
    faiss_candidates,
    track_step,
):
    """Step 5: Matcher Optimization"""
    print("\n=== Step 5: Matcher Optimization ===")

    # Load matcher cache
    matcher_cache_path = dirs.matching / "matcher_configs.json"
    matcher_cache = load_json_cache(matcher_cache_path)
    blocking_cache = normalize_blocking_cache(
        load_json_cache(dirs.blocking / "blocking_configs.json")
    )

    fusion_correspondences = []
    best_matchers_summary = []

    cached = False if args.excluded_columns else check_matching_cache(pairs, dirs.matching)
    with track_step("Step 5: Matcher Optimization", cached=cached):
        for (left_name, right_name), val_set in validation_sets.items():
            cache_key = pair_key(left_name, right_name)
            corr_path = dirs.matching / \
                f"correspondences_{left_name}_{right_name}.csv"
            df_left = results[left_name]
            df_right = results[right_name]

            # Check cache
            cached_matcher = matcher_cache.get(cache_key)
            if cached_matcher and corr_path.exists() and not args.excluded_columns:
                has_test_metric = _official_entitymatching_result_exists(dirs, left_name, right_name)
                has_test_set = _has_official_entitymatching_test(
                    args,
                    left_name,
                    right_name,
                    df_left,
                    df_right,
                )
                if has_test_set and not has_test_metric:
                    print(
                        f"\n{left_name} <-> {right_name}: Cached matcher lacks official EM test metrics; re-optimizing"
                    )
                else:
                    print(f"\n{left_name} <-> {right_name}: Using cached matcher")
                    print(
                        f"  {cached_matcher.get('matcher')} (F1={cached_matcher.get('f1', 0):.3f})")

                    matched = pd.read_csv(corr_path)
                    if not matched.empty:
                        fusion_correspondences.append(matched)

                    best_matchers_summary.append({
                        "left": left_name, "right": right_name,
                        "best_matcher": cached_matcher.get("matcher"),
                        "f1": cached_matcher.get("f1"),
                        "precision": cached_matcher.get("precision"),
                        "recall": cached_matcher.get("recall"),
                        "threshold": cached_matcher.get("threshold"),
                    })
                    continue

            print(f"\nOptimizing matchers for {left_name} <-> {right_name}...")

            train_set = training_sets[(left_name, right_name)]
            train_pos, train_neg = _entity_label_counts(train_set)
            include_ml_based = INCLUDE_ML_BASED and train_pos > 0 and train_neg > 0
            if INCLUDE_ML_BASED and not include_ml_based:
                print(
                    f"  Warning: training set has {train_pos} positives and {train_neg} negatives; "
                    "skipping ML matchers for this pair."
                )

            # Get matching columns from blocking cache if available
            matching_columns = None
            cached_blocking = blocking_cache.get(cache_key)
            if isinstance(cached_blocking, dict):
                cols = cached_blocking.get("blocking_columns") or []
                if isinstance(cols, str):
                    cols = [cols]
                cols = [
                    c for c in cols if c in df_left.columns and c in df_right.columns and c != "id"]
                if cols:
                    matching_columns = cols

            match_opt = optimize_matching(
                df_left=df_left,
                df_right=df_right,
                validation_set=val_set,
                id_column="id",
                matching_columns=matching_columns,
                include_rule_based=INCLUDE_RULE_BASED,
                include_ml_based=include_ml_based,
                include_llm_based=INCLUDE_LLM_BASED,
                training_set=train_set if include_ml_based else None,
                thresholds=MATCHER_THRESHOLDS,
                out_dir=dirs.matching,
                out_path=dirs.matching /
                f"matcher_optimization_{left_name}_{right_name}.csv",
            )

            best = match_opt["best"]
            best_artifacts = match_opt["best_artifacts"]

            if best:
                print(f"  Best: {best['matcher']} (F1={best['f1']:.3f})")
                test_result = _evaluate_best_matcher_on_entitymatching_test(
                    args,
                    dirs,
                    best,
                    best_artifacts,
                    df_left,
                    df_right,
                    left_name,
                    right_name,
                )
                best_matchers_summary.append({
                    "left": left_name, "right": right_name,
                    "best_matcher": best.get("matcher"),
                    "f1": best.get("f1"),
                    "precision": best.get("precision"),
                    "recall": best.get("recall"),
                    "threshold": best.get("threshold"),
                    "test_f1": test_result.get("test_f1") if test_result else None,
                    "test_precision": test_result.get("test_precision") if test_result else None,
                    "test_recall": test_result.get("test_recall") if test_result else None,
                })

            # Run best matcher on FAISS candidates
            candidates_df = faiss_candidates.get((left_name, right_name))
            if candidates_df is None or candidates_df.empty:
                continue

            matched = _run_best_matcher(
                best, best_artifacts,
                df_left, df_right, candidates_df,
                dirs.matching, left_name, right_name,
            )

            if matched is not None and not matched.empty:
                fusion_correspondences.append(matched)
                # Update cache
                matcher_cache[cache_key] = {
                    "matcher": best.get("matcher"),
                    "threshold": best.get("threshold"),
                    "f1": best.get("f1"),
                    "precision": best.get("precision"),
                    "recall": best.get("recall"),
                }
                save_json_cache(matcher_cache_path, matcher_cache)

    # Save summary
    if best_matchers_summary:
        summary_df = pd.DataFrame(best_matchers_summary)
        summary_df.to_csv(dirs.matching / "matching_summary.csv", index=False)
        print("\n--- Best Matchers ---")
        for row in best_matchers_summary:
            print(
                f"  {row['left']} <-> {row['right']}: {row['best_matcher']} (F1={row['f1']:.3f})")

    return fusion_correspondences, best_matchers_summary


def _run_best_matcher(
    best, best_artifacts,
    df_left, df_right, candidates_df,
    matching_dir, left_name, right_name,
):
    """Run the best matcher on candidates and save correspondences."""
    corr = _score_best_matcher(best, best_artifacts, df_left, df_right, candidates_df)
    if corr is None or corr.empty:
        return None

    mname = str(best.get("matcher") or "")
    thr = float(best.get("threshold") or 0.5)
    display_cols = [
        c for c in df_left.columns if c in df_right.columns and c != "id"][:4]

    report_correspondences(
        title=mname, corr=corr, threshold=thr,
        df_left=df_left, df_right=df_right,
        id_column="id", display_cols=display_cols,
        top_examples=DIAGNOSTICS_TOP_EXAMPLES,
    )

    matched = corr[pd.to_numeric(corr["score"], errors="coerce") >= thr]
    matched = matched[["id1", "id2", "score"]].dropna(
    ).drop_duplicates().reset_index(drop=True)

    if not matched.empty:
        out_path = matching_dir / \
            f"correspondences_{left_name}_{right_name}.csv"
        matched.to_csv(out_path, index=False)
        return matched

    return None


def _score_best_matcher(best, best_artifacts, df_left, df_right, candidates_df):
    """Score candidate pairs with a trained best matcher artifact."""
    if not best or not best_artifacts:
        return None

    mname = str(best.get("matcher") or "")
    matcher = best_artifacts.get("matcher")

    if matcher is None:
        return None

    df_left = df_left.copy()
    df_right = df_right.copy()
    if "id" in df_left.columns:
        df_left["id"] = df_left["id"].astype(str)
    if "id" in df_right.columns:
        df_right["id"] = df_right["id"].astype(str)
    candidates_scored = candidates_df[["id1", "id2"]].copy()
    candidates_scored["id1"] = candidates_scored["id1"].astype(str)
    candidates_scored["id2"] = candidates_scored["id2"].astype(str)
    cand_batches = [candidates_scored]

    try:
        if mname == "RuleBasedMatcher":
            comparators = best_artifacts.get("comparators")
            if not comparators:
                return None
            corr = matcher.match(
                df_left=df_left, df_right=df_right,
                candidates=cand_batches, id_column="id",
                comparators=comparators, threshold=0.0,
            )
        elif mname == "MLBasedMatcher":
            classifier = best_artifacts.get("classifier")
            if not classifier:
                return None
            corr = matcher.match(
                df_left=df_left, df_right=df_right,
                candidates=cand_batches, id_column="id",
                trained_classifier=classifier, threshold=0.0,
                use_probabilities=bool(best.get("use_probabilities", True)),
            )
        elif mname == "LLMBasedMatcher":
            corr = matcher.match(
                df_left=df_left, df_right=df_right,
                candidates=cand_batches, id_column="id",
                threshold=0.0,
            )
        else:
            return None

        return corr

    except Exception as e:
        print(f"  Matcher failed: {e}")

    return None


def _evaluate_best_matcher_on_entitymatching_test(
    args,
    dirs: PipelineDirectories,
    best,
    best_artifacts,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> dict | None:
    """Evaluate the selected matcher on the official held-out EM test set."""
    test_sets_dir = Path(args.data_dir).parent / "entitymatching"
    test_set = load_entitymatching_test_set(
        args,
        left_name,
        right_name,
        test_sets_dir,
        df_left=df_left,
        df_right=df_right,
    )
    if test_set is None or test_set.empty:
        return None

    corr = _score_best_matcher(best, best_artifacts, df_left, df_right, test_set[["id1", "id2"]])
    if corr is None or corr.empty:
        return None

    scores = corr[["id1", "id2", "score"]].copy()
    scores["id1"] = scores["id1"].astype(str)
    scores["id2"] = scores["id2"].astype(str)
    scores = scores.drop_duplicates(["id1", "id2"], keep="first")
    scored = test_set.merge(scores, on=["id1", "id2"], how="left")
    scored["score"] = pd.to_numeric(scored["score"], errors="coerce").fillna(0.0)

    threshold = float(best.get("threshold") or 0.5)
    truth = scored["label"].astype(str).str.upper() == "TRUE"
    predicted = scored["score"] >= threshold
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    result = {
        "left": left_name,
        "right": right_name,
        "pair": f"{left_name}__{right_name}",
        "matcher": best.get("matcher"),
        "threshold": threshold,
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
        "test_pairs": len(scored),
        "test_positives": int(truth.sum()),
        "test_negatives": int((~truth).sum()),
        "test_set": test_set.attrs.get("source_path"),
    }

    out_path = dirs.matching / "entity_matching_official_test.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[
            ~(
                (existing["left"].astype(str) == left_name)
                & (existing["right"].astype(str) == right_name)
            )
        ]
        out_df = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)
    else:
        out_df = pd.DataFrame([result])
    out_df.to_csv(out_path, index=False)

    print(
        f"  Official EM test: F1={f1:.3f}, precision={precision:.3f}, "
        f"recall={recall:.3f} ({len(scored)} pairs)"
    )
    return result


def _official_entitymatching_result_exists(
    dirs: PipelineDirectories,
    left_name: str,
    right_name: str,
) -> bool:
    """Return True if the final official EM test metric already exists for a pair."""
    path = dirs.matching / "entity_matching_official_test.csv"
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty or not {"left", "right"}.issubset(df.columns):
        return False
    return bool(
        (
            (df["left"].astype(str) == left_name)
            & (df["right"].astype(str) == right_name)
        ).any()
    )


def _has_official_entitymatching_test(
    args,
    left_name: str,
    right_name: str,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
) -> bool:
    """Return True if an official EM test set can be loaded for this pair."""
    test_sets_dir = Path(args.data_dir).parent / "entitymatching"
    return load_entitymatching_test_set(
        args,
        left_name,
        right_name,
        test_sets_dir,
        df_left=df_left,
        df_right=df_right,
        quiet=True,
    ) is not None


def _drop_validation_overlap(training_set: pd.DataFrame, validation_set: pd.DataFrame) -> pd.DataFrame:
    """Remove validation pairs from a training set before matcher optimization."""
    if training_set.empty or validation_set.empty:
        return training_set.reset_index(drop=True)
    val_pairs = set(zip(validation_set["id1"].astype(str), validation_set["id2"].astype(str)))
    return training_set[
        ~training_set.apply(lambda r: (str(r["id1"]), str(r["id2"])) in val_pairs, axis=1)
    ].reset_index(drop=True)


def _entity_label_counts(labeled: pd.DataFrame) -> tuple[int, int]:
    """Return positive/negative counts for a labeled EM pair set."""
    if labeled is None or labeled.empty or "label" not in labeled.columns:
        return 0, 0
    labels = labeled["label"].astype(str).str.upper()
    return int((labels == "TRUE").sum()), int((labels == "FALSE").sum())


def _dedupe_labeled_pairs(labeled: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate labeled [id1, id2, label] rows while preserving order."""
    if labeled is None or labeled.empty:
        return pd.DataFrame(columns=["id1", "id2", "label"])
    out = labeled[["id1", "id2", "label"]].copy()
    out["id1"] = out["id1"].astype(str)
    out["id2"] = out["id2"].astype(str)
    out["label"] = _normalize_entitymatching_labels(out["label"])
    out = out[out["label"].isin(["TRUE", "FALSE"])].dropna(subset=["id1", "id2"])
    out["_pair_key"] = out["id1"] + "||" + out["id2"]
    out = out.drop_duplicates("_pair_key", keep="first")
    return out.drop(columns=["_pair_key"]).reset_index(drop=True)


def _load_unused_labeled_pairs(
    training_dir: Path,
    left_name: str,
    right_name: str,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
) -> pd.DataFrame:
    """Load unused LLM labels saved during similarity-set generation."""
    frames = []
    for left, right in [(left_name, right_name), (right_name, left_name)]:
        path = training_dir / f"{left}_{right}_unused_labels.csv"
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, dtype={"label": str})
        except Exception as exc:
            print(f"  Warning: could not read unused labels {path.name}: {exc}")
            continue
        if not {"id1", "id2", "label"}.issubset(frame.columns):
            continue
        frame = _dedupe_labeled_pairs(frame)
        frame = _fix_entitymatching_pair_order(frame, df_left, df_right)
        if not frame.empty:
            frames.append(frame[["id1", "id2", "label"]])

    if not frames:
        return pd.DataFrame(columns=["id1", "id2", "label"])
    return _dedupe_labeled_pairs(pd.concat(frames, ignore_index=True))


def _repair_training_with_unused_labels(
    training_set: pd.DataFrame,
    validation_set: pd.DataFrame,
    training_dir: Path,
    left_name: str,
    right_name: str,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Supplement a collapsed training set with unused LLM labels."""
    training_set = _dedupe_labeled_pairs(training_set)
    pos, neg = _entity_label_counts(training_set)
    if pos > 0 and neg > 0:
        return training_set, 0

    unused = _load_unused_labeled_pairs(training_dir, left_name, right_name, df_left, df_right)
    if unused.empty:
        return training_set, 0

    unused = _drop_validation_overlap(unused, validation_set)
    if unused.empty:
        return training_set, 0

    repaired = _dedupe_labeled_pairs(pd.concat([training_set, unused], ignore_index=True))
    repaired = _drop_validation_overlap(repaired, validation_set)
    added = max(0, len(repaired) - len(training_set))
    if added:
        r_pos, r_neg = _entity_label_counts(repaired)
        print(
            f"  Repaired collapsed training seed with {added} unused LLM labels: "
            f"{len(repaired)} pairs ({r_pos} pos, {r_neg} neg)"
        )
    return repaired, added


def step4_active_training_construction(
    args,
    dirs: PipelineDirectories,
    results,
    validation_sets,
    training_sets,
    faiss_candidates,
    llm,
    track_step,
):
    """Step 4: Expand FAISS seed training sets with active learning."""
    print("\n=== Step 4: Active Learning Training Construction ===")

    augmentation_summary = []
    active_cache_suffix = embedding_cache_suffix(args)
    active_cache_prefix = f"training_{active_cache_suffix}" if active_cache_suffix else "training"
    force_active_learning = os.getenv("PYDI_FORCE_ACTIVE_LEARNING", "0") == "1"
    active_cache_only = os.getenv("PYDI_ACTIVE_LEARNING_CACHE_ONLY", "0") == "1"
    active_summary_path = dirs.training / "active_learning_summary.csv"
    if active_summary_path.exists():
        try:
            cached_active_summary = pd.read_csv(active_summary_path)
        except Exception:
            cached_active_summary = pd.DataFrame()
    else:
        cached_active_summary = pd.DataFrame()

    def _cached_active_summary_row(left_name: str, right_name: str) -> dict | None:
        if cached_active_summary.empty:
            return None
        rows = cached_active_summary[
            (cached_active_summary["left"].astype(str) == left_name)
            & (cached_active_summary["right"].astype(str) == right_name)
        ]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    with track_step("Step 4: Active Learning Training Construction"):
        for (left_name, right_name), train_set in training_sets.items():
            print(f"\n{left_name} <-> {right_name}:")

            val_set = validation_sets.get(
                (left_name, right_name), pd.DataFrame())
            candidates = faiss_candidates.get((left_name, right_name))

            # Load training file
            training_path = None
            legacy_min_mtime = None
            if active_cache_suffix:
                suffix_seed_path = find_training_file(
                    dirs.training,
                    left_name,
                    right_name,
                    cache_suffix=active_cache_suffix,
                    allow_legacy_fallback=False,
                )
                if suffix_seed_path and suffix_seed_path.exists():
                    try:
                        legacy_min_mtime = suffix_seed_path.stat().st_mtime
                    except OSError:
                        legacy_min_mtime = None

            training_path = find_training_file(
                dirs.training,
                left_name,
                right_name,
                cache_suffix=active_cache_suffix,
                allow_legacy_fallback=(not active_cache_suffix or legacy_min_mtime is not None),
                legacy_min_mtime=legacy_min_mtime,
            )
            if training_path and training_path.exists():
                original_training = pd.read_csv(training_path)
                print(
                    f"  Loaded: {training_path.name} ({len(original_training)} rows)")
                loaded_active_checkpoint = training_path.name in {
                    f"{active_cache_prefix}_{left_name}_{right_name}_latest.csv",
                    f"{active_cache_prefix}_{right_name}_{left_name}_latest.csv",
                    f"{active_cache_prefix}_{left_name}_{right_name}_augmented.csv",
                    f"{active_cache_prefix}_{right_name}_{left_name}_augmented.csv",
                }
                loaded_legacy_checkpoint = bool(
                    active_cache_suffix and active_cache_suffix not in training_path.name
                )
                if loaded_legacy_checkpoint:
                    print(
                        f"  Adopting legacy active-learning checkpoint newer than the {active_cache_suffix} seed; "
                        "future checkpoints will use the suffixed name."
                    )
            else:
                original_training = train_set
                loaded_active_checkpoint = False
                loaded_legacy_checkpoint = False

            original_training = _drop_validation_overlap(original_training, val_set)
            original_training, repaired_unused_labels = _repair_training_with_unused_labels(
                original_training,
                val_set,
                dirs.training,
                left_name,
                right_name,
                results[left_name],
                results[right_name],
            )
            if loaded_legacy_checkpoint:
                original_training.to_csv(
                    dirs.training / f"{active_cache_prefix}_{left_name}_{right_name}_latest.csv",
                    index=False,
                )
                original_training.to_csv(
                    dirs.training / f"{active_cache_prefix}_{left_name}_{right_name}_augmented.csv",
                    index=False,
                )
                print(f"  Saved suffix-safe checkpoint: {active_cache_prefix}_{left_name}_{right_name}_latest.csv")
            elif repaired_unused_labels:
                original_training.to_csv(
                    dirs.training / f"{active_cache_prefix}_{left_name}_{right_name}_latest.csv",
                    index=False,
                )
                original_training.to_csv(
                    dirs.training / f"{active_cache_prefix}_{left_name}_{right_name}_augmented.csv",
                    index=False,
                )
            training_sets[(left_name, right_name)] = original_training

            original_pos = (original_training["label"].astype(
                str).str.upper() == "TRUE").sum()
            original_neg = (original_training["label"].astype(
                str).str.upper() == "FALSE").sum()
            print(
                f"  Current: {original_pos} pos, {original_neg} neg | Target: {TRAINING_LARGE_TARGET_POSITIVES} pos, {TRAINING_LARGE_TARGET_NEGATIVES} neg")

            cached_summary_row = _cached_active_summary_row(left_name, right_name)
            if active_cache_only and not force_active_learning:
                print(
                    "  Using existing training artifact; "
                    "PYDI_ACTIVE_LEARNING_CACHE_ONLY=1 prevents extending active learning"
                )
                if cached_summary_row:
                    augmentation_summary.append(cached_summary_row)
                continue

            if (
                loaded_active_checkpoint
                and cached_summary_row
                and not force_active_learning
            ):
                print("  Using cached active training set; set PYDI_FORCE_ACTIVE_LEARNING=1 to extend it")
                augmentation_summary.append(cached_summary_row)
                continue

            # Check if already have enough
            if original_pos >= TRAINING_LARGE_TARGET_POSITIVES and (TRAINING_LARGE_TARGET_NEGATIVES == 0 or original_neg >= TRAINING_LARGE_TARGET_NEGATIVES):
                print("  Skipping: targets already met")
                continue

            if candidates is None or candidates.empty:
                print("  Skipping active learning: no FAISS candidates")
                continue

            augmented, was_augmented, summary = run_active_learning(
                df_left=results[left_name],
                df_right=results[right_name],
                left_name=left_name,
                right_name=right_name,
                training_set=original_training,
                validation_set=val_set,
                candidates=candidates,
                chat_model=llm,
                output_dir=dirs.training,
                id_column="id",
                target_positives=TRAINING_LARGE_TARGET_POSITIVES,
                target_negatives=TRAINING_LARGE_TARGET_NEGATIVES if TRAINING_LARGE_TARGET_NEGATIVES > 0 else None,
                max_total_labels=ACTIVE_LEARNING_MAX_LABELS,
                labels_per_iteration=ACTIVE_LEARNING_LABELS_PER_ITER,
                max_candidates=ACTIVE_LEARNING_MAX_CANDIDATES,
                max_iterations=ACTIVE_LEARNING_MAX_ITERATIONS,
                low_label_patience=ACTIVE_LEARNING_LOW_LABEL_PATIENCE,
                low_label_threshold=ACTIVE_LEARNING_LOW_LABEL_THRESHOLD,
                label_batch_size=ACTIVE_LEARNING_BATCH_SIZE,
                label_max_workers=LLM_LABELING_WORKERS,
                ensemble_score_workers=ACTIVE_ENSEMBLE_WORKERS,
                targeted_pool_multiplier=ACTIVE_TARGETED_POOL_MULTIPLIER,
                cache_suffix=active_cache_suffix,
            )

            if was_augmented:
                final_training = _drop_validation_overlap(augmented, val_set)
                training_sets[(left_name, right_name)] = final_training

                # Save shuffled augmented set
                augmented_shuffled = final_training.sample(
                    frac=1, random_state=42).reset_index(drop=True)
                augmented_shuffled.to_csv(
                    dirs.training / f"{active_cache_prefix}_{left_name}_{right_name}_augmented.csv", index=False)
                augmented_shuffled.to_csv(
                    dirs.training / f"{active_cache_prefix}_{left_name}_{right_name}_latest.csv", index=False)
            else:
                final_training = original_training
                training_sets[(left_name, right_name)] = final_training

            final_pos = (final_training["label"].astype(str).str.upper() == "TRUE").sum()
            final_neg = (final_training["label"].astype(str).str.upper() == "FALSE").sum()
            print(f"  Active training set: {len(final_training)} pairs ({final_pos} pos, {final_neg} neg)")

            summary.setdefault("final_size", len(final_training))
            summary.setdefault("final_positives", int(final_pos))
            summary.setdefault("final_negatives", int(final_neg))
            augmentation_summary.append(summary)

    if augmentation_summary:
        aug_df = pd.DataFrame(augmentation_summary)
        aug_df.to_csv(dirs.training /
                      "active_learning_summary.csv", index=False)
        print("\n--- Active Learning Summary ---")
        print(aug_df.to_string(index=False))

    return training_sets


def step7_data_fusion(
    dirs: PipelineDirectories,
    results,
    fusion_correspondences,
    llm,
    track_step,
):
    """Step 7: Data Fusion"""
    print("\n=== Step 7: Data Fusion ===")

    cached = check_fusion_cache(dirs.fusion)
    with track_step("Step 7: Data Fusion", cached=cached):
        if cached:
            print("Fusion output cached; skipping.")
            return

        if not fusion_correspondences:
            print("No correspondences; skipping fusion.")
            return

        all_corr = pd.concat(fusion_correspondences, ignore_index=True)
        all_corr = all_corr.drop_duplicates(
            subset=["id1", "id2"]).reset_index(drop=True)
        all_corr.to_csv(dirs.matching / "correspondences_all.csv", index=False)
        print(f"Fusing with {len(all_corr)} correspondences")

        fused, strategy, _ = run_data_fusion(
            results,
            correspondences=all_corr,
            config=FusionConfig(
                id_column="id",
                include_singletons=FUSION_INCLUDE_SINGLETONS,
                debug=True,
                use_llm=FUSION_USE_LLM,
            ),
            chat_model=llm,
            output_dir=dirs.fusion,
        )
        print(f"Fused: {len(fused)} rows (strategy={strategy.name})")

        # Save clean version
        metadata_cols = ["_fusion_source_datasets",
                         "_fusion_confidence", "_fusion_metadata"]
        clean_cols = [c for c in fused.columns if c not in metadata_cols]
        fused[clean_cols].to_csv(dirs.fusion / "fused_clean.csv", index=False)


def step7_5_fusion_validation(
    args,
    dirs: PipelineDirectories,
    results,
    fusion_correspondences,
    llm,
    track_step,
):
    """Step 7.5: Fusion Validation Set Generation"""
    print("\n=== Step 7.5: Fusion Validation Set Generation ===")

    if not fusion_correspondences:
        print("No correspondences; skipping.")
        return

    # Load target entity IDs from test set if enabled
    target_entity_ids = None
    if args.fusion_use_test_set:
        fusion_test_set_path = Path(
            args.data_dir).parent / "fusion" / "test_set.xml"
        if fusion_test_set_path.exists():
            print(f"Loading test set from {fusion_test_set_path}...")
            test_set_df = load_fusion_reference_file(fusion_test_set_path)
            if "id" in test_set_df.columns:
                target_entity_ids = set(test_set_df["id"].astype(str).tolist())
                print(f"  >> {len(target_entity_ids)} entity IDs loaded")

    # Parse fusion mode into variants to generate
    mode_configs = {
        "llm": [(False, False, "validation_llm")],
        "llm_omit": [(False, True, "validation_llm_omit")],
        "web": [(True, False, "validation_web")],
        "web_omit": [(True, True, "validation_web_omit")],
        "all": [
            (False, False, "validation_llm"),
            (False, True, "validation_llm_omit"),
            (True, False, "validation_web"),
            (True, True, "validation_web_omit"),
        ],
    }
    variants = mode_configs.get(args.fusion_val_generation_mode, [
                                (False, False, "validation")])

    all_corr = pd.concat(fusion_correspondences, ignore_index=True)
    all_corr = all_corr.drop_duplicates(
        subset=["id1", "id2"]).reset_index(drop=True)

    # Pre-compute identifying columns once if any variant needs omit mode
    identifying_columns = None
    needs_omit = any(omit for _, omit, _ in variants)
    if needs_omit:
        all_attributes = set()
        for df in results.values():
            all_attributes.update(
                c for c in df.columns if c not in ("id", "_id"))
        print(
            f"Pre-computing identifying columns from {len(all_attributes)} attributes...")
        identifying_columns = identify_identifying_columns(
            list(all_attributes), llm)
        print(
            f"  >> Identified {len(identifying_columns)} identifying columns: {identifying_columns}")

    # Pre-select entity groups once if running multiple variants
    # This ensures the same entities are used across all validation variants
    selected_groups = None
    if len(variants) > 1:
        print("\nPre-selecting entity groups for all validation variants...")
        entity_groups = build_entity_groups(all_corr, results)
        print(
            f"  >> Found {len(entity_groups)} entity groups spanning multiple datasets")
        if entity_groups:
            if target_entity_ids is not None:
                # Filter to test set entities
                selected_groups = []
                for group in entity_groups:
                    for record in group.records:
                        record_id = str(record.get(
                            "_id", record.get("id", "")))
                        if record_id in target_entity_ids:
                            selected_groups.append(group)
                            break
                print(
                    f"  >> Filtered to {len(selected_groups)} groups matching test set IDs (will reuse for all variants)")
            else:
                # Select well-known entities
                selected_groups = select_well_known_entities(
                    entity_groups,
                    llm,
                    n_select=FUSION_VALIDATION_N_ENTITIES,
                    sample_size=FUSION_VALIDATION_SAMPLE_SIZE,
                )
                print(
                    f"  >> Selected {len(selected_groups)} well-known entities (will reuse for all variants)")

    for use_web_search, omit_target, subdir_name in variants:
        variant_dir = dirs.fusion_validation.parent / subdir_name
        variant_label = f"web_search={use_web_search}, omit_target={omit_target}"
        print(f"\n--- Generating variant: {subdir_name} ({variant_label}) ---")

        cached = load_fusion_validation_from_cache(variant_dir) is not None
        with track_step(f"Step 7.5: Fusion Validation ({subdir_name})", cached=cached):
            cached_val = load_fusion_validation_from_cache(variant_dir)
            if cached_val is not None:
                print(f"Loaded cached: {len(cached_val)} records")
                continue

            validation_set = generate_fusion_validation_set(
                correspondences=all_corr,
                datasets=results,
                chat_model=llm,
                n_entities=FUSION_VALIDATION_N_ENTITIES,
                sample_size=FUSION_VALIDATION_SAMPLE_SIZE,
                output_dir=variant_dir,
                use_web_search=use_web_search,
                omit_target_attribute=omit_target,
                target_entity_ids=target_entity_ids,
                identifying_columns=identifying_columns if omit_target else None,
                selected_groups=selected_groups,
            )
            print(f"Generated: {len(validation_set)} records")

            if not validation_set.empty:
                ground_truth_df, _ = convert_to_tabular_format(
                    validation_set, output_dir=variant_dir
                )
                print(f"Tabular format: {len(ground_truth_df)} entities")


def step7_6_end_to_end_metrics(
    dirs: PipelineDirectories,
    results,
    track_step,
):
    """Step 7.6: Calculate End-to-End Integration Metrics"""
    print("\n=== Step 7.6: End-to-End Integration Metrics ===")

    fused_path = dirs.fusion / "fused_clean.csv"
    if not fused_path.exists():
        print("No fused output found; skipping metrics.")
        return None

    # Look for fusion debug file (may be in optimization or main fusion dir)
    debug_file = None
    optimization_dir = dirs.fusion / "optimization"
    if optimization_dir.exists():
        # Find the most recent fusion_debug.jsonl in optimization subdirs
        for case_dir in optimization_dir.iterdir():
            if case_dir.is_dir():
                candidate = case_dir / "fusion_debug.jsonl"
                if candidate.exists():
                    debug_file = candidate
                    break
    if debug_file is None:
        # Check main fusion directory
        main_debug = dirs.fusion / "fusion_debug.jsonl"
        if main_debug.exists():
            debug_file = main_debug

    with track_step("Step 7.6: End-to-End Metrics"):
        fused_df = pd.read_csv(fused_path)

        # Also load the full fused file with metadata columns for metrics
        full_fused_path = dirs.fusion / "fused.csv"
        if not full_fused_path.exists():
            # Try optimization directory
            if optimization_dir.exists():
                for case_dir in optimization_dir.iterdir():
                    if case_dir.is_dir():
                        candidate = case_dir / "fused.csv"
                        if candidate.exists():
                            full_fused_path = candidate
                            break

        if full_fused_path.exists():
            fused_df_with_metadata = pd.read_csv(full_fused_path)
        else:
            fused_df_with_metadata = fused_df

        metrics = calculate_structural_metrics(
            datasets=results,
            fused_df=fused_df_with_metadata,
            debug_file=debug_file,
        )

        # Save metrics
        metrics_path = dirs.output / "end_to_end_metrics.json"
        metrics.save(metrics_path)
        print(f"Saved metrics to {metrics_path}")

        # Print and save end-to-end report
        report_text = generate_end_to_end_report(metrics)
        print(report_text)

        # Save report files
        txt_path, csv_path = save_end_to_end_report(metrics, dirs.output)
        print(f"\nSaved report to: {txt_path}")
        print(f"Saved report CSV to: {csv_path}")

        # Print per-source statistics
        print("\n--- Per-Source Statistics ---")
        print(metrics.per_source_table().to_string(index=False))

    return metrics


def step7_7_fusion_optimization(
    args,
    dirs: PipelineDirectories,
    results,
    fusion_correspondences,
    llm,
    track_step,
):
    """Step 7.7: Fusion Rule Optimization - runs multiple strategies and picks the best."""
    print("\n=== Step 7: Data Fusion (Optimization) ===")

    if not fusion_correspondences:
        print("No correspondences; skipping fusion optimization.")
        return None

    all_corr = pd.concat(fusion_correspondences, ignore_index=True)
    all_corr = all_corr.drop_duplicates(
        subset=["id1", "id2"]).reset_index(drop=True)

    # Load validation sets from all available modes
    validation_sets = {}
    mode_dirs = {
        "llm": dirs.fusion / "validation_llm",
        "llm_omit": dirs.fusion / "validation_llm_omit",
        "web": dirs.fusion / "validation_web",
        "web_omit": dirs.fusion / "validation_web_omit",
    }

    for mode, mode_dir in mode_dirs.items():
        val_path = mode_dir / "fusion_validation_set.csv"
        if val_path.exists():
            validation_sets[mode] = pd.read_csv(val_path)
            print(
                f"  Loaded validation set: {mode} ({len(validation_sets[mode])} records)")
        else:
            print(f"  Validation set not found: {mode}")

    if not validation_sets:
        print("No validation sets available; skipping fusion optimization.")
        print(
            "Run with --fusion-val-generation-mode all first to generate validation sets.")
        return None

    # Load test set if provided (for final evaluation only, not optimization)
    test_set = None
    test_set_raw = None
    if args.test_dir:
        test_dir = Path(args.test_dir)
        # Look for test_set.xml file (standard format)
        test_xml = test_dir / "test_set.xml"
        if test_xml.exists():
            test_set_raw = load_fusion_reference_file(test_xml)
            test_set = test_set_raw
            print(
                f"  Loaded test set: {len(test_set)} records from {test_xml}")
        else:
            # Fallback to CSV/JSON files
            test_files = list(test_dir.glob("*.csv")) + \
                list(test_dir.glob("*.json"))
            if test_files:
                test_dfs = []
                for tf in test_files:
                    if tf.suffix == ".csv":
                        test_dfs.append(pd.read_csv(tf))
                    elif tf.suffix == ".json":
                        test_dfs.append(pd.read_json(tf))
                if test_dfs:
                    test_set = pd.concat(test_dfs, ignore_index=True)
                    print(
                        f"  Loaded test set: {len(test_set)} records from {len(test_files)} files")

    # Normalize test set using same normalization as Step 1 (including taxonomy mapping)
    if test_set is not None:
        alignment_lookup = build_fusion_alignment_lookup(args)
        target_schema = load_runtime_target_schema(args)

        spec = load_normalization_spec(target_schema)
        result = transform_dataframe(
            test_set,
            spec,
            chat_model=llm,  # For taxonomy mapping
            taxonomy_cache_dir=str(dirs.schema_matching),  # Reuse Step 1's cache
            schema_base_path=runtime_taxonomy_base_paths(args),
        )
        test_set = drop_runtime_excluded_columns(args, result.dataframe, "fusion test set")
        if test_set_raw is not None:
            test_set = attach_fusion_source_ids_from_alignment(
                test_set,
                test_set_raw,
                args,
                alignment_lookup,
                "fusion test",
            )
            test_set = ensure_fusion_expected_id(test_set, "fusion test")
        print(f"  Normalized test set: {len(test_set)} records")

    config = FusionConfig(
        id_column="id",
        include_singletons=FUSION_INCLUDE_SINGLETONS,
        debug=False,  # Disable verbose fusion logging; evaluation logs are separate
        use_llm=True,
    )

    optimization_dir = dirs.fusion / "optimization"
    optimization_dir.mkdir(parents=True, exist_ok=True)

    # Determine which cases to run (default to all if not specified)
    fusion_case = args.fusion_case or "all"
    if fusion_case == "all":
        cases_to_run = ["heuristic", "llm_no_val", "llm_val",
                        "llm_val_stats", "heuristic_stats", "iterative"]
    else:
        cases_to_run = [fusion_case]

    with track_step(f"Step 7: Fusion Optimization ({fusion_case})"):
        case_results, best_case_key, best_case_dir = run_all_fusion_cases(
            datasets=results,
            validation_sets=validation_sets,
            correspondences=all_corr,
            chat_model=llm,
            config=config,
            output_dir=optimization_dir,
            iterations=args.fusion_iterations,
            primary_validation_mode=args.fusion_primary_val_mode,
            cases=cases_to_run,
            test_set=test_set,
        )

    # Print summary
    print("\n--- Fusion Optimization Summary ---")
    for case_name, result in case_results.items():
        print(f"\n{case_name}:")
        for mode, acc in result.accuracy_by_mode.items():
            print(f"  {mode}: {acc:.1%}")

    # Create fused_clean.csv from the best case
    if best_case_dir and best_case_dir.exists():
        best_fused_path = best_case_dir / "fused.csv"
        if best_fused_path.exists():
            print(f"\nCreating fused_clean.csv from best case: {best_case_key}")
            best_fused = pd.read_csv(best_fused_path)

            # Remove metadata columns for clean version
            metadata_cols = ["_fusion_source_datasets", "_fusion_confidence", "_fusion_metadata"]
            clean_cols = [c for c in best_fused.columns if c not in metadata_cols]
            best_fused[clean_cols].to_csv(dirs.fusion / "fused_clean.csv", index=False)

            # Also copy the fusion rules used
            best_rules_path = best_case_dir / "fusion_rules.json"
            if best_rules_path.exists():
                import shutil
                shutil.copy(best_rules_path, dirs.fusion / "fusion_rules.json")

            print(f"  Saved: {dirs.fusion / 'fused_clean.csv'}")
        else:
            print(f"Warning: Best case fused output not found at {best_fused_path}")

    return case_results


def step7_8_provided_validation_comparison(
    args,
    dirs: PipelineDirectories,
    results,
    fusion_correspondences,
    llm,
    track_step,
):
    """Step 7.8: Run fusion optimization with the provided validation set.

    This step:
    1. Loads the provided validation_set.xml from the test directory
    2. Converts it to the CSV format used by fusion optimization
    3. Runs fusion optimization using the provided validation set
    4. Compares results between auto-generated and provided validation approaches when available
    5. Saves comparison metrics and both fusion outputs
    """
    if not args.compare_provided_validation:
        return None

    print("\n=== Step 7.8: Provided Validation Comparison ===")

    if not args.test_dir:
        print("Warning: --test-dir required for --compare-provided-validation")
        return None

    # Check for provided validation set
    test_dir = Path(args.test_dir)
    provided_val_path = find_fusion_reference_path(
        test_dir,
        [
            "validation_set.xml",
            "validation_set_final.xml",
            "small/validation_set.xml",
        ],
    )
    if provided_val_path is None:
        print(f"Warning: No validation_set.xml found under {test_dir}")
        return None

    if not fusion_correspondences:
        print("No correspondences; skipping provided validation comparison.")
        return None

    all_corr = pd.concat(fusion_correspondences, ignore_index=True)
    all_corr = all_corr.drop_duplicates(subset=["id1", "id2"]).reset_index(drop=True)

    # Load and normalize the provided validation set
    print(f"Loading provided validation set from {provided_val_path}...")
    provided_val_raw = load_fusion_reference_file(provided_val_path)
    print(f"  Loaded {len(provided_val_raw)} entities from provided validation set")
    alignment_lookup = build_fusion_alignment_lookup(args)

    # Normalize using same normalization as Step 1
    target_schema = load_runtime_target_schema(args)

    spec = load_normalization_spec(target_schema)
    result = transform_dataframe(
        provided_val_raw,
        spec,
        chat_model=llm,
        taxonomy_cache_dir=str(dirs.schema_matching),
        schema_base_path=runtime_taxonomy_base_paths(args),
    )
    provided_val_normalized = drop_runtime_excluded_columns(
        args, result.dataframe, "provided validation set"
    )
    provided_val_normalized = attach_fusion_source_ids_from_alignment(
        provided_val_normalized,
        provided_val_raw,
        args,
        alignment_lookup,
        "provided validation",
    )
    provided_val_normalized = ensure_fusion_expected_id(
        provided_val_normalized, "provided validation"
    )
    print(f"  Normalized: {len(provided_val_normalized)} entities")

    # Convert to CSV format for fusion optimization
    print("Converting provided validation set to optimization format...")
    provided_val_csv = convert_provided_validation_to_csv_format(
        validation_xml_df=provided_val_normalized,
        datasets=results,
        correspondences=all_corr,
        id_column="id",
    )
    print(f"  Converted: {len(provided_val_csv)} validation records")

    if provided_val_csv.empty:
        print("Warning: No validation records could be extracted from provided validation set")
        return None

    # Save the converted validation set for inspection
    provided_val_dir = dirs.fusion / "validation_provided"
    provided_val_dir.mkdir(parents=True, exist_ok=True)
    provided_val_csv.to_csv(provided_val_dir / "fusion_validation_set.csv", index=False)

    # Also create ground truth in tabular format (for evaluation)
    from PyDI.pipeline.fusion_validation_generation import convert_to_tabular_format
    gt_df, _ = convert_to_tabular_format(provided_val_csv, output_dir=provided_val_dir)
    print(f"  Ground truth: {len(gt_df)} entities")

    # Create validation_sets dict for provided validation
    # We'll use "provided" as the mode name
    provided_validation_sets = {"provided": provided_val_csv}

    # Also load the auto-generated validation sets for comparison
    auto_validation_sets = {}
    mode_dirs = {
        "llm": dirs.fusion / "validation_llm",
        "llm_omit": dirs.fusion / "validation_llm_omit",
        "web": dirs.fusion / "validation_web",
        "web_omit": dirs.fusion / "validation_web_omit",
    }
    for mode, mode_dir in mode_dirs.items():
        val_path = mode_dir / "fusion_validation_set.csv"
        if val_path.exists():
            auto_validation_sets[mode] = pd.read_csv(val_path)

    # Load test set for final evaluation
    test_set = None
    test_xml = find_fusion_reference_path(
        test_dir,
        [
            "test_set.xml",
            "test_set_final.xml",
            "small/test_set.xml",
        ],
    )
    if test_xml is not None:
        test_set_raw = load_fusion_reference_file(test_xml)
        # Normalize test set
        test_result = transform_dataframe(
            test_set_raw,
            spec,
            chat_model=llm,
            taxonomy_cache_dir=str(dirs.schema_matching),
            schema_base_path=runtime_taxonomy_base_paths(args),
        )
        test_set = drop_runtime_excluded_columns(args, test_result.dataframe, "provided test set")
        test_set = attach_fusion_source_ids_from_alignment(
            test_set,
            test_set_raw,
            args,
            alignment_lookup,
            "provided test",
        )
        test_set = ensure_fusion_expected_id(test_set, "provided test")
        print(f"  Loaded test set: {len(test_set)} records")

    config = FusionConfig(
        id_column="id",
        include_singletons=FUSION_INCLUDE_SINGLETONS,
        debug=False,
        use_llm=True,
    )

    # Determine which cases to run
    fusion_case = args.fusion_case or "all"
    if fusion_case == "all":
        cases_to_run = ["heuristic", "llm_no_val", "llm_val",
                        "llm_val_stats", "heuristic_stats", "iterative"]
    else:
        cases_to_run = [fusion_case]

    # Run fusion optimization using provided validation set
    print("\n--- Running fusion optimization with PROVIDED validation set ---")
    provided_output_dir = dirs.fusion / "optimization_provided_val"
    provided_output_dir.mkdir(parents=True, exist_ok=True)

    with track_step("Step 7.8: Fusion Optimization (Provided Val)"):
        provided_results, provided_best_key, provided_best_dir = run_all_fusion_cases(
            datasets=results,
            validation_sets=provided_validation_sets,
            correspondences=all_corr,
            chat_model=llm,
            config=config,
            output_dir=provided_output_dir,
            iterations=args.fusion_iterations,
            primary_validation_mode="provided",
            cases=cases_to_run,
            test_set=test_set,
        )

    # Load auto-generated results from step 7.7
    auto_output_dir = dirs.fusion / "optimization"
    auto_best_case_path = auto_output_dir / "best_case.json"
    auto_best_info = None
    if not args.official_validation_fusion_only and auto_best_case_path.exists():
        with open(auto_best_case_path) as f:
            auto_best_info = json.load(f)

    # Create comparison summary
    print("\n--- Creating Comparison Summary ---")
    comparison_rows = []

    # Collect results from auto-generated validation
    auto_summary_path = auto_output_dir / "comparison_summary_validation.csv"
    if not args.official_validation_fusion_only and auto_summary_path.exists():
        auto_summary = pd.read_csv(auto_summary_path)
        for _, row in auto_summary.iterrows():
            comparison_rows.append({
                "validation_source": f"auto_{row['optimization_mode']}",
                "case": row["case"],
                "eval_mode": row["eval_mode"],
                "accuracy": row["accuracy"],
                "correct": row.get("correct", ""),
                "total": row.get("total", ""),
            })

    # Collect results from provided validation
    for result_key, result in provided_results.items():
        for mode, acc in result.accuracy_by_mode.items():
            comparison_rows.append({
                "validation_source": "provided",
                "case": result.case_name,
                "eval_mode": mode,
                "accuracy": acc,
                "correct": "",
                "total": "",
            })

    # Save comparison CSV
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_path = dirs.fusion / "validation_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Saved comparison to {comparison_path}")

    # Create best case comparison
    best_comparison = {
        "auto_generated": {
            "best_case_key": auto_best_info.get("best_case_key") if auto_best_info else None,
            "validation_accuracy": auto_best_info.get("best_validation_accuracy") if auto_best_info else None,
            "test_accuracy": auto_best_info.get("best_test_accuracy") if auto_best_info else None,
            "rules": auto_best_info.get("rules") if auto_best_info else None,
        },
        "provided": {
            "best_case_key": provided_best_key,
            "validation_accuracy": provided_results[provided_best_key].accuracy_by_mode.get("provided") if provided_best_key else None,
            "test_accuracy": provided_results[provided_best_key].accuracy_by_mode.get("test") if provided_best_key else None,
            "rules": provided_results[provided_best_key].rules if provided_best_key else None,
        },
    }

    best_comparison_path = dirs.fusion / "best_case_comparison.json"
    with open(best_comparison_path, "w") as f:
        json.dump(best_comparison, f, indent=2, default=str)
    print(f"Saved best case comparison to {best_comparison_path}")

    # Generate fusion comparison report with per-attribute accuracy
    auto_best_dir = Path(auto_best_info.get("best_case_dir")) if auto_best_info and auto_best_info.get("best_case_dir") else None
    fusion_report_dir = dirs.output / "reporting" / "fusion"
    comparison_report_path = generate_fusion_comparison_report(
        output_dir=fusion_report_dir,
        auto_dir=auto_best_dir,
        provided_dir=provided_best_dir,
        auto_best_info=auto_best_info,
        provided_best_info={
            "best_case_key": provided_best_key,
            "best_validation_accuracy": provided_results[provided_best_key].accuracy_by_mode.get("provided") if provided_best_key else None,
            "best_test_accuracy": provided_results[provided_best_key].accuracy_by_mode.get("test") if provided_best_key else None,
            "rules": provided_results[provided_best_key].rules if provided_best_key else None,
        } if provided_best_key else None,
    )
    if comparison_report_path:
        print(f"Saved fusion comparison report to {comparison_report_path}")

    # Save best provided validation fused output
    if provided_best_dir and provided_best_dir.exists():
        best_fused_path = provided_best_dir / "fused.csv"
        if best_fused_path.exists():
            import shutil
            # Copy to main fusion directory with distinct name
            dest_path = dirs.fusion / "fused_provided_val.csv"
            best_fused = pd.read_csv(best_fused_path)
            metadata_cols = ["_fusion_source_datasets", "_fusion_confidence", "_fusion_metadata"]
            clean_cols = [c for c in best_fused.columns if c not in metadata_cols]
            best_fused[clean_cols].to_csv(dest_path, index=False)
            print(f"Saved provided validation fused output to {dest_path}")

            if args.official_validation_fusion_only:
                canonical_path = dirs.fusion / "fused_clean.csv"
                best_fused[clean_cols].to_csv(canonical_path, index=False)
                print(f"Saved official-validation fused output to {canonical_path}")

            # Also copy fusion rules
            rules_path = provided_best_dir / "fusion_rules.json"
            if rules_path.exists():
                shutil.copy(rules_path, dirs.fusion / "fusion_rules_provided_val.json")
                if args.official_validation_fusion_only:
                    shutil.copy(rules_path, dirs.fusion / "fusion_rules.json")

    # Rename auto-generated output for clarity (if not already named)
    auto_fused_path = dirs.fusion / "fused_clean.csv"
    if not args.official_validation_fusion_only and auto_fused_path.exists():
        # Determine auto validation mode used
        auto_val_mode = args.fusion_primary_val_mode or "llm_omit"
        dest_auto_path = dirs.fusion / f"fused_auto_{auto_val_mode}.csv"
        if not dest_auto_path.exists():
            import shutil
            shutil.copy(auto_fused_path, dest_auto_path)
            print(f"Copied auto-generated fused output to {dest_auto_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPARISON SUMMARY")
    print("=" * 70)

    if auto_best_info:
        print(f"\nAuto-generated validation (mode: {auto_best_info.get('best_case_key', 'N/A')}):")
        print(f"  Validation accuracy: {auto_best_info.get('best_validation_accuracy', 'N/A'):.1%}" if auto_best_info.get('best_validation_accuracy') else "  Validation accuracy: N/A")
        print(f"  Test accuracy: {auto_best_info.get('best_test_accuracy', 'N/A'):.1%}" if auto_best_info.get('best_test_accuracy') else "  Test accuracy: N/A")

    if provided_best_key:
        prov_val_acc = provided_results[provided_best_key].accuracy_by_mode.get("provided")
        prov_test_acc = provided_results[provided_best_key].accuracy_by_mode.get("test")
        print(f"\nProvided validation (case: {provided_best_key}):")
        print(f"  Validation accuracy: {prov_val_acc:.1%}" if prov_val_acc else "  Validation accuracy: N/A")
        print(f"  Test accuracy: {prov_test_acc:.1%}" if prov_test_acc else "  Test accuracy: N/A")

    return provided_results


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Setup
    dirs = PipelineDirectories(Path(args.output_dir))
    dirs.create_all()
    setup_logging(dirs.output)

    # Load metrics. Timing harnesses can isolate metrics without changing the
    # pipeline output directory by setting PYDI_PIPELINE_METRICS_PATH.
    metrics_path = Path(
        os.getenv("PYDI_PIPELINE_METRICS_PATH", str(dirs.output / "pipeline_metrics.json"))
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = PipelineMetrics.load(metrics_path)
    _, track_step = create_step_tracker(metrics)

    # LLMs
    chat_kwargs = {
        "timeout": LLM_REQUEST_TIMEOUT_SECONDS,
        "max_retries": LLM_MAX_RETRIES,
    }
    schema_llm = ChatOpenAI(model=SCHEMA_LLM_MODEL, **chat_kwargs)
    labeling_llm = ChatOpenAI(model=LABELING_LLM_MODEL, **chat_kwargs)
    fusion_llm = ChatOpenAI(model=FUSION_LLM_MODEL, **chat_kwargs)
    print(
        "LLM models: "
        f"schema={SCHEMA_LLM_MODEL}, "
        f"labeling={LABELING_LLM_MODEL}, "
        f"fusion={FUSION_LLM_MODEL}; "
        f"timeout={LLM_REQUEST_TIMEOUT_SECONDS:g}s, retries={LLM_MAX_RETRIES}"
    )

    # Print source overview (raw data statistics before any transformation)
    print_and_save_source_overview(args, dirs)

    # Pipeline steps
    results = step1_schema_matching(args, dirs, schema_llm, track_step)

    # Generate schema matching report
    generate_schema_matching_report_for_pipeline(
        schema_path=args.schema,
        mappings_dir=dirs.schema_matching / "mappings",
        output_dir=dirs.output,
        normalized_results=results,
        raw_data_dir=args.data_dir,
        excluded_columns=args.excluded_columns,
        schema_gold_path=args.schema_gold,
    )

    # Generate normalization statistics (LLM from cache, builtin from pipeline_stats)
    norm_stats_path = save_normalization_stats(output_dir=dirs.output)
    if norm_stats_path:
        print(f"\n  Saved normalization stats to: {norm_stats_path}")

    pairs, validation_sets, training_sets = step2_validation_training(
        args, dirs, results, labeling_llm, track_step
    )

    faiss_candidates = step3_faiss_candidates(
        args, dirs, pairs, results, schema_llm, track_step)

    # Generate estimated labeling cost report
    print(
        "\n=== Labeling Cost Estimate ===\n"
        "Sampling 10,000 candidate prompts per pair, averaging tokens, and scaling linearly "
        "(set PYDI_COST_ESTIMATE_SAMPLE_SIZE=0 for exact counting)."
    )
    cost_report_path = save_cost_report(
        results=results,
        candidates_dict=faiss_candidates,
        output_dir=dirs.output,
        model=LABELING_LLM_MODEL,
    )
    if cost_report_path:
        print(f"\n  Saved labeling cost estimate to: {cost_report_path}")

    if args.training_set_construction == "compare":
        # Legacy path: generate all variants and select the best one.
        training_sets = step4_training_comparison(
            args, dirs, pairs, results, validation_sets, training_sets,
            faiss_candidates, labeling_llm, track_step
        )
    elif args.training_set_construction == "active":
        training_sets = step4_active_training_construction(
            args, dirs, results, validation_sets, training_sets,
            faiss_candidates, labeling_llm, track_step
        )
    else:
        print(
            "\n=== Step 4: Training Set Construction ===\n"
            "Using FAISS seed training sets; active learning and variant comparison skipped."
        )

    fusion_correspondences, _ = step5_matcher_optimization(
        args, dirs, pairs, results, validation_sets, training_sets,
        faiss_candidates, track_step
    )

    # Save cluster size distribution report
    cluster_report_path = save_cluster_size_report(
        fusion_correspondences, num_datasets=len(results), output_dir=dirs.output,
    )
    if cluster_report_path:
        print(f"  Saved cluster size distribution to: {cluster_report_path}")

    # Clean oversized clusters using post-clustering algorithms
    fusion_correspondences, cleaning_report = clean_oversized_clusters(
        fusion_correspondences, num_datasets=len(results),
    )
    if cleaning_report["best_strategy"] is not None:
        report_dir = save_cluster_cleaning_report(cleaning_report, dirs.output)
        tex_path = save_cluster_cleaning_latex(
            cleaning_report, num_datasets=len(results), output_dir=dirs.output,
        )
        print(f"  Cluster cleaning reports saved to: {report_dir}")
        print(f"  LaTeX table saved to: {tex_path}")

    # Step 7: Data Fusion with optimization
    # Runs multiple fusion strategies, evaluates with type-aware matching, and picks the best
    if args.official_validation_fusion_only:
        print(
            "\n=== Step 7: Official Validation Fusion Only ===\n"
            "Skipping generated fusion validation sets and optimizing with the provided validation_set.xml."
        )
    else:
        step7_5_fusion_validation(
            args, dirs, results, fusion_correspondences, fusion_llm, track_step)

        step7_7_fusion_optimization(
            args, dirs, results, fusion_correspondences, fusion_llm, track_step)

    # Step 7.8: Run fusion optimization with the provided validation set (optional)
    # In official-only mode, this is the primary fusion optimization path.
    step7_8_provided_validation_comparison(
        args, dirs, results, fusion_correspondences, fusion_llm, track_step)

    step7_6_end_to_end_metrics(dirs, results, track_step)

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(metrics.summary_table().to_string(index=False))
    metrics_csv_path = Path(
        os.getenv("PYDI_PIPELINE_METRICS_CSV_PATH", str(dirs.output / "pipeline_metrics.csv"))
    )
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.summary_table().to_csv(metrics_csv_path, index=False)
    print(f"\nTotal Runtime: {metrics.total_runtime():.1f}s")
    print(f"Total Tokens: {metrics.total_tokens():,}")
    print(f"Total Cost: ${metrics.total_cost():.4f}")
    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
    exectution_time = time.time() - start_time
    print(f"\nTotal execution time: {exectution_time:.1f} seconds")
