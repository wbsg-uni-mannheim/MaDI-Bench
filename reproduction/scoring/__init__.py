"""Scoring code behind the paper's numbers (``results/scores_v2.json``
and the end-to-end metric panels).

Run the command-line tools from the repository root as modules, e.g.::

    python -m reproduction.scoring.rescore --out /tmp/rescore

Each module's docstring and ``--help`` describe its inputs and outputs.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Score with this checkout's ``madi_bench`` package even when another copy is
# installed in the environment.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SubmissionContractError(ValueError):
    """A submission file violates the submission contract (missing or empty
    file, missing columns, malformed rows). The stage scorers raise this
    instead of a raw KeyError / EmptyDataError so that a caller can record
    the stage as an error and go on with the other stages."""
