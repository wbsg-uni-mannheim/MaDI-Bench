"""MaDI-Bench evaluation: the fusion and normalization scorers the reference
numbers were produced with.

    from madi_bench.evaluation import score_fusion, score_normalization
    score_fusion("use cases/games/base", fused_df, membership_df)
    score_normalization("use cases/games/base", "output/normalization")

Command-line: ``madi-score-fusion`` and ``madi-score-normalization``.
"""

from .fusion_rules import STRICT_RULES, describe, rules_for
from .fusion_score import SubmissionError, load_table, score_fusion
from .fusion_gold import ANCHOR_PRIORITY, load_gold
from .normalization_score import NormalizedTables, score_normalization

__all__ = ["ANCHOR_PRIORITY", "NormalizedTables", "STRICT_RULES", "SubmissionError", "describe",
           "load_gold", "load_table", "rules_for", "score_fusion", "score_normalization"]
