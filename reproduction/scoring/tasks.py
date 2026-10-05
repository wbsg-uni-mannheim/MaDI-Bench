"""The 20 MaDI-Bench tasks of this repository: ``use cases/<domain>/<tier>/``.

The task tree defaults to the repository's ``use cases/`` folder;
``--tasks-root`` on the command line or the environment variable
``MADI_BENCH_USECASES`` (the same variable the ``madi_bench`` tests and
command-line tools read) point the scorers at another copy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import REPO_ROOT

DOMAINS = ("companies", "games", "music", "products", "papers")
TIERS = ("base", "easy", "medium", "hard")

# Fusion gold file format at the base tier; every variant tier ships XML
# (papers variants: JSON lines inside a file named test_set.xml).
FUSION_GOLD_FORMAT_BASE = {
    "companies": "xml",
    "games": "xml",
    "music": "xml",
    "products": "csv",
    "papers": "jsonl",
}


@dataclass(frozen=True)
class TaskSpec:
    domain: str
    tier: str
    root: Path
    is_fixture: bool = False      # read by gold.discover_em_test_files; always False here

    @property
    def task_id(self) -> str:
        return f"{self.domain}_{self.tier}"

    @property
    def fusion_gold_format(self) -> str:
        if self.tier == "base":
            return FUSION_GOLD_FORMAT_BASE.get(self.domain, "xml")
        return "xml"


class UnknownTaskError(KeyError):
    pass


def default_tasks_root() -> Path:
    env = os.environ.get("MADI_BENCH_USECASES")
    return Path(env) if env else REPO_ROOT / "use cases"


def get_task(task_id: str, tasks_root: Path | None = None) -> TaskSpec:
    """``<domain>_<tier>`` (e.g. ``games_base``) -> its task folder."""
    domain, _, tier = task_id.rpartition("_")
    if domain not in DOMAINS or tier not in TIERS:
        raise UnknownTaskError(task_id)
    root = Path(tasks_root) if tasks_root is not None else default_tasks_root()
    task_root = root / domain / tier
    if not task_root.is_dir():
        raise FileNotFoundError(f"task folder not found: {task_root}")
    return TaskSpec(domain=domain, tier=tier, root=task_root)


def task_from_dir(task_dir: Path | str) -> TaskSpec:
    """A task folder (``.../<domain>/<tier>``) -> its TaskSpec."""
    task_dir = Path(task_dir)
    resolved = task_dir.resolve()
    domain, tier = resolved.parent.name, resolved.name
    if domain not in DOMAINS or tier not in TIERS:
        raise UnknownTaskError(f"not a task folder (<domain>/<tier>): {task_dir}")
    return TaskSpec(domain=domain, tier=tier, root=task_dir)
