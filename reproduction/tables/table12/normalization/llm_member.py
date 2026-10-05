#!/usr/bin/env python3
"""Table 12, columns Norm.: the LLM member (``llm_only``) of the generator's normalization committee, scored
on the normalization test set of each task. Completes the three-member committee whose two other members
``measure.py`` scores.

It measures the ``llm_only`` values behind Table 12's Norm. columns. The cache location and the output
folder are arguments; the per-domain results are written to ``<out>/llm_member/`` (``compare.py`` joins
them with the output of ``measure.py``).
A paid run needs an OpenAI key and calls the OpenAI API; ``--dry-run`` makes no call and needs no key.

How the member is built (exactly as the committee builds it)
------------------------------------------------------------
* Task data: ``usecases_synthetic.lib.variant_loader.load_variant(domain, level)``, as in ``measure.py``.
* Committee: ``NormCommitteeRunner(<normalization_committee_<domain>.yaml>, with_llm=True,
  scoring_surface="schema_constraints")``, with the rule lock of ``measure.py``'s default setting (the shipped
  lock, else chosen by the committee at the base level and reused by the variants; the lock is
  ``<out>/llm_member/locks/<domain>.json``; ``--locks-from`` as in ``measure.py``). ``runner.run(task)`` is
  called unchanged. The committee builds ``llm_only`` itself (YAML ``llm_normalizer.params``: model
  gpt-5.4-mini, number of in-context examples, temperature 0, max_tokens 2048, prompt version v2) and wires
  its in-context examples from the task's fusion VALIDATION gold values (Products: 5 examples per attribute;
  the other domains: none). The script checks at run time that every example is a validation value and none
  is a value that only the test gold has. The only substitution is the class of the member object, a
  subclass of the committee's ``LLMCanonicalizer`` that keeps its prompts, example wiring, cache keys and
  response parsing and changes only:
    - the cache directory: ``<--cache-root>/<cache_dir of the YAML>`` (the YAML path is relative);
    - no calls while the committee's own stage runs (the committee also scores the member on its own
      surface; those cells are not needed here, so the member abstains there without touching the cache);
    - API errors: the committee's class turns a transport or API error into an abstention that its cache
      would keep for good. Here the error is reported under ``api.errors``, nothing is cached, the cell
      abstains for this run, and a rerun asks again. Cached entries that hold such an error are asked again
      in a paid run. Answers that are not in the expected JSON form are the member's real behaviour: they are
      cached and counted (``parse_error_answers_cached``);
    - two audit fields added to each newly cached answer (``examples_source``, ``examples_sha256``), which
      the committee's reader ignores.
* Attributes: the committee applies ``llm_only`` to its eligible attributes only (attributes of the gold
  schema mapping with a target-schema constraint: the keys of the rule member's selection). Cells of other
  attributes keep their raw value, as for the rule member in ``measure.py``. The ``kind`` passed to the
  member comes from ``kind_map_for_domain(domain)`` (default ``long_string``), as in the committee.

Cells and scoring
-----------------
Only the (record, attribute) cells named in the task's ``input/normalization/test.csv`` are normalized; the
records are found with the scorer's own lookup, and the raw values are read through the gold schema mapping
as ``_export_member`` of ``reproduction/scoring/p2_norm_replay.py`` reads them (first mapped column; a later
column fills in only when the earlier ones gave nothing). The resulting table (one ``<source>.csv`` per
source, only the test cells filled) is scored with ``score_normalization(task_dir, tables, split="test")``
and deleted afterwards.

Cache
-----
The levels of a domain run in one process in the order baseline, easy, medium, hard, so a value asked at
one level is found in the cache at the next (the cache key is sha256 of domain|attribute|value|v2_n<number
of examples>|model; it contains neither the level nor the examples). A run with an empty cache asks every
prompt. Products needs a cache root of its own,
either empty or one this script started for Products (marker ``PRODUCTS_FRESH_CACHE.json``): cached Products
answers made with in-context examples from anywhere but the validation gold
must not be reused, and the cache key does not tell them apart. Cache roots inside the repository are
refused.

Modes
-----
``--dry-run`` (no key needed; every ``*_API_KEY`` variable is removed): the LLM client and the member's
call are replaced by stand-ins that raise on any API call (a BaseException, so nothing swallows it); cache
misses abstain without a call. Reports per level and per domain: normalize calls, distinct prompts (= cache
keys), cache hits, cached error entries, new misses, input tokens (tiktoken o200k_base over system + user
prompt + 9 framing tokens; ``null`` when the tokenizer file cannot be loaded) and cost bounds at the prices of
``PRICE_IN`` / ``PRICE_OUT``: visible output (``MEAN_VISIBLE_OUTPUT_TOKENS``), times 10 for hidden reasoning,
and the hard cap (every call at max_tokens). A level whose prompts are all in the cache is scored as in a paid
run.
Paid run (no ``--dry-run``; ``OPENAI_API_KEY`` set): each level's misses are asked once each on a thread pool
(``--workers``); before a level's calls, the domain's planned calls and their hard-cap cost are checked against
``--max-new-calls`` and ``--max-cap-usd`` (the run stops before calling). The export then reads the cache only.
Token usage reported by the API is appended to ``<out>/llm_member/<domain>.ledger.jsonl``. A rerun asks only
what is still missing.

Usage (from the repository root)::

    python reproduction/tables/table12/normalization/llm_member.py --domain music \\
        --cache-root /tmp/norm12_llm_cache --out /tmp/norm12 --dry-run
    OPENAI_API_KEY=... python reproduction/tables/table12/normalization/llm_member.py --domain music \\
        --cache-root /tmp/norm12_llm_cache --out /tmp/norm12
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import measure  # noqa: E402  (helpers shared with the non-LLM members)

DOMAINS = measure.DOMAINS
LEVELS = measure.LEVELS
SURFACE = measure.SURFACE
PRODUCTS_MARKER = "PRODUCTS_FRESH_CACHE.json"
PRICE_IN, PRICE_OUT = 0.75, 4.50          # USD per 1M tokens, gpt-5.4-mini (prices used for the paper's estimate)
REASONING_FACTOR = 10
CHAT_FRAMING_TOKENS = 9                   # 2 messages x 3 + 3 reply priming (OpenAI chat format)
# Mean visible output tokens of the cached llm_only answers per domain, measured over the LLM cache of the
# paper's run (2026-09-28); used for the cost estimate only.
MEAN_VISIBLE_OUTPUT_TOKENS = {"companies": 39.31, "games": 35.95, "music": 38.98, "papers": 39.76,
                              "products": 58.66}
EXPECTED_LLM_CLASS = ("usecases_synthetic.lib.llm_normalizer", "LLMCanonicalizer")

logger = logging.getLogger("table12_norm_llm")


class ApiCallRefused(BaseException):
    """An API call where none may happen (dry run, or outside the asking phase). BaseException so that
    ``LLMCanonicalizer._call_llm``'s ``except Exception`` cannot turn it into a cached abstention."""


class TransientLLMError(RuntimeError):
    """An API or transport error: raised past the cache so that it is not cached."""


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def guard_env(dry_run: bool) -> None:
    if dry_run:
        for key in list(os.environ):
            if key.endswith("API_KEY"):
                os.environ.pop(key)
    elif not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("a paid run needs OPENAI_API_KEY in the environment (or use --dry-run)")


def check_cache_root(root: Path, domain: str, *, dry_run: bool) -> Path:
    root = measure.outside_repo(root, "--cache-root")
    if domain == "products":
        has_entries = root.is_dir() and any(p.name != PRODUCTS_MARKER for p in root.rglob("*.json"))
        if has_entries and not (root / PRODUCTS_MARKER).exists():
            raise SystemExit(f"{root} holds cache entries this script did not start for Products; Products "
                             "needs a cache root of its own (empty, or started by this script for Products)")
        if not dry_run and not (root / PRODUCTS_MARKER).exists():
            root.mkdir(parents=True, exist_ok=True)
            (root / PRODUCTS_MARKER).write_text(json.dumps({
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "llm_only cache for products; in-context examples from the fusion validation gold",
                "script": "reproduction/tables/table12/normalization/llm_member.py"}, indent=1) + "\n")
    return root


class Ctl:
    """Shared run state: phase, API usage, failed keys."""

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.phase = "inert"            # inert | dry | prewarm | cache_only
        self.level = ""
        self.lock = threading.Lock()
        self.attempts = 0
        self.responses = 0
        self.usage: Counter = Counter()
        self.failed: dict[str, dict[str, Any]] = {}

    def record_usage(self, response: Any) -> None:
        um = getattr(response, "usage_metadata", None) or {}
        otd = um.get("output_token_details") or {}
        itd = um.get("input_token_details") or {}
        with self.lock:
            self.responses += 1
            self.usage["input_tokens"] += int(um.get("input_tokens") or 0)
            self.usage["output_tokens"] += int(um.get("output_tokens") or 0)
            self.usage["reasoning_tokens"] += int(otd.get("reasoning") or 0)
            self.usage["cached_input_tokens"] += int(itd.get("cache_read") or 0)


def usable(cache: Any, key: str) -> bool | None:
    """True: a cached answer; False: a cached API-error abstention; None: no entry."""
    payload = cache.get(key)
    if payload is None:
        return None
    res = payload.get("result")
    return isinstance(res, dict) and "error" not in res


def make_member_class(ln: Any, llm_client: Any, ctl: Ctl) -> type:
    class CommitteeLLM(ln.LLMCanonicalizer):
        """The committee's LLMCanonicalizer with the four changes of the module docstring."""

        def _ensure_llm_callable(self):  # noqa: D401
            if self._llm_callable is not None:
                return self._llm_callable
            if ctl.dry_run:
                def _refuse(system_prompt: str, user_prompt: str) -> str:
                    raise ApiCallRefused("dry run: an API call was attempted")

                self._llm_callable = _refuse
                return _refuse
            # as LLMCanonicalizer._ensure_llm_callable, plus the usage ledger
            from langchain_core.messages import HumanMessage, SystemMessage

            chat = llm_client.build_chat_openai(model=self.model_name, temperature=self.temperature,
                                                max_tokens=self.max_tokens)

            def _call(system_prompt: str, user_prompt: str) -> str:
                if ctl.phase != "prewarm":
                    raise ApiCallRefused(f"API call outside the asking phase ({ctl.phase})")
                with ctl.lock:
                    ctl.attempts += 1
                response = chat.invoke([SystemMessage(content=system_prompt),
                                        HumanMessage(content=user_prompt)])
                ctl.record_usage(response)
                content = getattr(response, "content", response)
                return str(content)

            self._llm_callable = _call
            return _call

        def _call_llm(self, *, value: str, attribute: str, kind: str, examples_text: str) -> dict[str, Any]:
            result = super()._call_llm(value=value, attribute=attribute, kind=kind, examples_text=examples_text)
            if isinstance(result, dict) and "error" in result:
                raise TransientLLMError(str(result["error"]))
            if isinstance(result, dict):
                result = {**result, "examples_source": "fusion_validation",
                          "examples_sha256": sha256_text(examples_text)}
            return result

        def normalize(self, value: Any, *, attribute: str, kind: str, domain: str) -> str | None:
            if ctl.phase == "inert":
                return None
            s = ln._stringify(value)
            if s is None:
                return None
            key = self._cache.make_cell_hash(source=domain, attribute=attribute, value=s)
            if key in ctl.failed:
                return None
            if not usable(self._cache, key):
                if ctl.phase == "dry":
                    return None
                if ctl.phase != "prewarm":
                    raise ApiCallRefused(f"cache miss outside the asking phase ({ctl.phase}, {attribute})")
            try:
                return super().normalize(value, attribute=attribute, kind=kind, domain=domain)
            except TransientLLMError as exc:
                with ctl.lock:
                    ctl.failed[key] = {"level": ctl.level, "attribute": attribute, "error": str(exc)[:400]}
                return None

    return CommitteeLLM


def costs(calls: int, in_tok: int | None, out_mean: float, max_tokens: int) -> dict[str, Any]:
    vis = calls * out_mean
    i = in_tok or 0
    usd = lambda i, o: round(i / 1e6 * PRICE_IN + o / 1e6 * PRICE_OUT, 4)  # noqa: E731
    d = {"calls": calls, "input_tokens": in_tok, "visible_output_tokens_est": round(vis),
         "usd_visible": usd(i, vis), "usd_reasoning_x10": usd(i, REASONING_FACTOR * vis),
         "usd_hard_cap": usd(i, calls * max_tokens)}
    if in_tok is None:
        d["note"] = "input tokens not counted (tokenizer not available): the amounts cover the output only"
    return d


def tokenizer() -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception as exc:  # noqa: BLE001 - no tokenizer file (offline): token counts are left out
        logger.warning("tiktoken o200k_base not available (%s): input tokens are not counted", exc)
        return None


# ---------------------------------------------------------------------------
# domain run
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    dry = bool(args.dry_run)
    guard_env(dry)
    out = measure.outside_repo(Path(args.out), "--out") / "llm_member"
    cache_root = check_cache_root(Path(args.cache_root), args.domain, dry_run=dry)
    locks_from = Path(args.locks_from).resolve() if args.locks_from else None
    levels = measure.parse_levels(args.levels)
    domain = args.domain
    measure.setup_repo()
    t_start = time.time()

    import madi_bench
    import usecases_synthetic
    from madi_bench.evaluation.normalization_score import NormalizedTables, source_stems, test_rows
    from usecases_synthetic.lib import committee_norm_c12 as c12
    from usecases_synthetic.lib import llm_client
    from usecases_synthetic.lib import llm_normalizer as ln
    from usecases_synthetic.lib.committee_norm import NormCommitteeRunner, _build_source_attribute_index
    from usecases_synthetic.lib.committee_paths import resolve_committee_path
    from usecases_synthetic.lib.protection import kind_map_for_domain
    from usecases_synthetic.lib.variant_loader import load_variant

    measure.check_modules(madi_bench, usecases_synthetic)
    c12_src = Path(c12.__file__).read_text(encoding="utf-8")
    fix_present = "for entity_attrs in val_targets.values():" in c12_src

    ctl = Ctl(dry)
    if dry:
        def _no_client(*a: Any, **k: Any) -> Any:
            raise ApiCallRefused("dry run: LLM client construction attempted")

        llm_client.build_chat_openai = _no_client

    # ---- rule lock + op-log folder outside the repository --------------------
    lock_path = out / "locks" / f"{domain}.json"
    lock_init = measure.init_lock(domain, lock_path, locks_from)
    if lock_init["source"] is None and levels[0] != "baseline":
        raise SystemExit("without a lock the committee chooses the rules at the first level it runs: start "
                         "--levels with baseline (or pass --locks-from)")
    measure.redirect_committee(c12, lock_path, out / "oplog")

    # ---- the committee builds llm_only; only the class is substituted --------
    CommitteeLLM = make_member_class(ln, llm_client, ctl)
    member_ref: dict[str, Any] = {}

    def _build_llm(llm_config: Any, *, op_log_path: Any) -> Any:
        if (llm_config.module, llm_config.cls_name) != EXPECTED_LLM_CLASS:
            raise SystemExit(f"unexpected llm_normalizer class {llm_config.module}.{llm_config.cls_name}")
        params = dict(llm_config.params)
        rel_dir = Path(str(params.get("cache_dir") or "usecases_synthetic/cache/llm_normalizer"))
        if rel_dir.is_absolute():
            raise SystemExit(f"YAML cache_dir is absolute: {rel_dir}")
        params["cache_dir"] = str(cache_root / rel_dir)
        inst = CommitteeLLM(**params)          # op_log_path left out: no operation log
        member_ref.update(llm=inst, yaml_params=dict(llm_config.params), cache_dir=params["cache_dir"])
        return inst

    c12._build_llm_instance = _build_llm

    norm_yaml = resolve_committee_path("normalization_committee", domain, committee_dir=measure.COMMITTEE_DIR)
    enc = tokenizer()
    sys_tokens = len(enc.encode_ordinary(ln._SYSTEM_PROMPT_V2)) if enc else None
    out_mean = MEAN_VISIBLE_OUTPUT_TOKENS[domain]

    part: dict[str, Any] = {
        "domain": domain, "mode": "dry_run" if dry else "paid", "levels": {},
        "setting": lock_init["setting"], "lock_init": lock_init,
        "norm_yaml": measure.rel(norm_yaml), "norm_yaml_sha256": measure.sha256_file(norm_yaml),
        "scoring_surface_of_the_committee_run": SURFACE, "cache_root": str(cache_root),
        "runner_takes_examples_from_validation_gold": fix_present,
        "code": {"usecases_synthetic": measure.rel(usecases_synthetic.__file__),
                 "madi_bench": measure.rel(madi_bench.__file__),
                 "committee_norm_c12_sha256": measure.sha256_file(Path(c12.__file__)),
                 "python": sys.version.split()[0]},
        "prices_usd_per_1M": {"input": PRICE_IN, "output": PRICE_OUT},
        "mean_visible_output_tokens": out_mean,
    }
    pending: dict[str, int | None] = {}   # dry run: keys a paid run would ask -> input tokens
    recall_keys: set[str] = set()         # cached API-error entries, asked again in a paid run
    dom_keys: set[str] = set()
    dom_hits: set[str] = set()            # usable in the cache before this run
    dom_tokens: dict[str, int | None] = {}  # every distinct prompt of the domain -> input tokens
    filled: set[str] = set()              # paid run: keys answered by this run
    dom_calls = 0
    max_tokens = None
    planned_calls = 0
    planned_in = 0

    def _sum(values: Any) -> int | None:
        values = list(values)
        return None if any(v is None for v in values) else int(sum(values))

    for level in levels:
        t_level = time.time()
        ctl.level = level
        task_dir = measure.TASKS / domain / measure.tier(level)
        bundle = load_variant(domain, level=level)
        dom = bundle.domain
        attr_index = _build_source_attribute_index(bundle.sm_mapping, bundle.knob_08_renames)
        kind_map = kind_map_for_domain(dom)

        # ---- the committee run: builds llm_only and its examples (member inert) --
        ctl.phase = "inert"
        member_ref.clear()
        runner = NormCommitteeRunner(norm_yaml, with_llm=True, scoring_surface=SURFACE)
        if not isinstance(runner, c12.C12NormCommitteeRunner) or "llm_only" not in runner.roster_names:
            raise RuntimeError(f"unexpected runner {type(runner).__name__} / roster {runner.roster_names}")
        result = runner.run(bundle)
        llm = member_ref["llm"]
        max_tokens = llm.max_tokens
        selection_map = dict(result.per_member["rule_per_attribute_optimal"].notes.get("selection_map", {}))
        eligible = set(selection_map)
        passthrough = c12._PassthroughNormalizer()

        # ---- where the in-context examples come from ---------------------------
        val_t, test_t = c12._load_eval_targets_for_bundle(dom, bundle)

        def _values(targets: dict[str, dict[str, list[str]]]) -> dict[str, set[str]]:
            vals: dict[str, set[str]] = {}
            for ent in targets.values():
                for a, vs in ent.items():
                    vals.setdefault(a, set()).update(str(v).strip() for v in vs if str(v).strip())
            return vals

        val_v, test_v = _values(val_t), _values(test_t)
        ex_info: dict[str, Any] = {}
        for (d, a), spec in sorted(llm._examples.items()):
            if d != dom:
                continue
            ex = set(spec.examples)
            not_val = ex - val_v.get(a, set())
            test_only = ex & (test_v.get(a, set()) - val_v.get(a, set()))
            if not_val or test_only:
                raise SystemExit(f"{domain}/{level} {a}: {len(not_val)} examples not from the validation gold, "
                                 f"{len(test_only)} only in the test gold; the committee must take its "
                                 "examples from the validation gold")
            if ex:
                ex_info[a] = {"n": len(spec.examples), "sha256": sha256_text("\n".join(spec.examples)),
                              "n_also_test_gold_values": len(ex & test_v.get(a, set()))}

        # ---- test cells -> records (the scorer's lookup over the task's records) --
        test = test_rows(task_dir, "test")
        nt = NormalizedTables.__new__(NormalizedTables)
        nt.aliases = {k: set(v) for k, v in source_stems(task_dir).items()}
        nt.rows = {}
        for src, df in bundle.sources.items():
            for i, rid in enumerate(df["id"].tolist()):
                nt.rows[(src, "" if rid is None else str(rid))] = {"_src": src, "_i": i}
        nt.skipped, nt.sources = [], set(bundle.sources)
        cells: dict[tuple[str, int, str], list[str]] = {}
        n_not_found = n_unmapped = 0
        for r in test:
            hit = nt.lookup(r["source"], r["source_id"])
            if hit is None:
                n_not_found += 1
                continue
            src, i, a = hit["_src"], hit["_i"], r["attribute"]
            df = bundle.sources[src]
            cols = [c for c in attr_index.get((src, a), []) if c in df.columns]
            if not cols:
                n_unmapped += 1
            cells[(src, i, a)] = cols
        scope = Counter("llm" if a in eligible else "passthrough" for (_, _, a) in cells)

        # ---- prompts of the LLM scope: accounting ------------------------------
        spec_text: dict[str, str] = {}
        plan: dict[str, tuple[str, str, Any]] = {}     # key -> (attribute, kind, raw value) to ask
        lv_keys: set[str] = set()
        lv_calls = 0
        st: Counter = Counter()
        in_tok_new: int | None = 0
        for (src, i, a), cols in cells.items():
            if a not in eligible:
                continue
            kind = kind_map.get(a, "long_string")
            df = bundle.sources[src]
            for c in cols:
                raw = df.iat[i, df.columns.get_loc(c)]
                s = ln._stringify(raw)
                if s is None:
                    continue
                lv_calls += 1
                key = llm._cache.make_cell_hash(source=dom, attribute=a, value=s)
                if key in lv_keys:
                    continue
                lv_keys.add(key)
                if key not in dom_tokens:          # the prompt as first rendered in this domain
                    if enc is None:
                        dom_tokens[key] = None
                    else:
                        if a not in spec_text:
                            spec = llm._examples.get((dom, a))
                            spec_text[a] = ln._format_examples(spec.examples if spec else [], llm.num_examples)
                        dom_tokens[key] = (sys_tokens + CHAT_FRAMING_TOKENS + len(enc.encode_ordinary(
                            ln._USER_PROMPT_TEMPLATE_V2.format(attribute=a, kind=kind, examples=spec_text[a],
                                                               value=s))))
                u = usable(llm._cache, key)
                if u and key not in filled:
                    st["hits"] += 1
                    dom_hits.add(key)
                    continue
                if u:
                    st["filled_at_earlier_level"] += 1
                    continue
                if key in ctl.failed:
                    st["api_error_at_earlier_level"] += 1
                    continue
                if u is False:
                    st["cached_api_error_entries"] += 1
                    recall_keys.add(key)
                if key in pending:
                    st["pending_from_earlier_level"] += 1
                    continue
                ntok = dom_tokens[key]
                pending[key] = ntok
                plan[key] = (a, kind, raw)
                st["new_misses"] += 1
                in_tok_new = None if (in_tok_new is None or ntok is None) else in_tok_new + ntok
        dom_keys |= lv_keys
        dom_calls += lv_calls
        planned_calls += len(plan)
        planned_in = None if (planned_in is None or in_tok_new is None) else planned_in + in_tok_new

        lv: dict[str, Any] = {
            "task": f"{domain}_{measure.tier(level)}", "variant_root": measure.rel(bundle.variant_root),
            "test_csv_sha256": measure.sha256_file(task_dir / "input" / "normalization" / "test.csv"),
            "test_rows": len(test), "test_rows_record_not_found": n_not_found,
            "test_cells_without_mapped_column": n_unmapped,
            "test_cells_by_scope": dict(scope),
            "eligible_attributes": sorted(eligible), "rule_selection_map": selection_map,
            "test_attributes_scope": {a: ("llm_only" if a in eligible else "not eligible -> raw value")
                                      for a in sorted({r["attribute"] for r in test})},
            "committee_entity_sets": {"validation": len(val_t), "test": len(test_t)},
            "llm_params": {k: v for k, v in member_ref["yaml_params"].items() if k != "cache_dir"},
            "llm_cache_dir": member_ref["cache_dir"],
            "cache_key_prompt_version": llm._cache.prompt_version,
            "examples": ex_info,
            "prompts": {"normalize_calls": lv_calls, "distinct": len(lv_keys), "hits": st["hits"],
                        "cached_api_error_entries": st["cached_api_error_entries"],
                        "pending_from_earlier_level": st["pending_from_earlier_level"],
                        "filled_at_earlier_level": st["filled_at_earlier_level"],
                        "api_error_at_earlier_level": st["api_error_at_earlier_level"],
                        "new_misses": st["new_misses"],
                        "cost_new_misses": costs(st["new_misses"], in_tok_new, out_mean, llm.max_tokens)},
        }
        logger.info("%s/%s: %d distinct prompts in the LLM scope, %d cache hits, %d new misses; cells %s",
                    domain, level, len(lv_keys), st["hits"], st["new_misses"], dict(scope))

        # ---- paid run: ask the misses once each ---------------------------------
        if not dry and plan:
            cap = costs(planned_calls, planned_in, out_mean, llm.max_tokens)["usd_hard_cap"]
            if planned_calls > args.max_new_calls or cap > args.max_cap_usd:
                raise SystemExit(f"{domain}: {planned_calls} planned calls / hard cap ${cap} exceed "
                                 f"--max-new-calls {args.max_new_calls} / --max-cap-usd {args.max_cap_usd}")
            for key in plan:
                if usable(llm._cache, key) is False:          # a cached API error: ask again
                    (llm._cache.cache_dir / f"{key}.json").unlink()
                    llm._cache._memory.pop(key, None)
            llm._ensure_llm_callable()
            ctl.phase = "prewarm"
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(lambda item: llm.normalize(item[2], attribute=item[0], kind=item[1], domain=dom),
                              plan.values()))
            lv["asking_s"] = round(time.time() - t0, 1)
            for key in plan:
                pending.pop(key, None)
                if key not in ctl.failed:
                    filled.add(key)

        # ---- export (cache only) + score ------------------------------------------
        ctl.phase = "dry" if dry else "cache_only"
        per_source: dict[str, dict[int, dict[str, str]]] = {}
        n_multi = abstain = 0
        abstain_by_attr: Counter = Counter()
        for (src, i, a), cols in cells.items():
            df = bundle.sources[src]
            kind = kind_map.get(a, "long_string")
            member = llm if a in eligible else passthrough
            value = ""
            n_multi += len(cols) > 1
            for ci, c in enumerate(cols):
                if ci > 0 and value:
                    break              # _export_member keeps the first non-empty value in mapping order
                raw = df.iat[i, df.columns.get_loc(c)]
                try:
                    res = member.normalize(raw, attribute=a, kind=kind, domain=dom)
                except Exception:      # noqa: BLE001 - the committee records such cells as None
                    res = None
                s_out = "" if res is None else str(res)
                if ci == 0:
                    value = s_out
                elif s_out and not value:
                    value = s_out
            if a in eligible and not value:
                abstain += 1
                abstain_by_attr[a] += 1
            per_source.setdefault(src, {}).setdefault(i, {})[a] = value

        def _parse_error(k: str) -> bool:
            res = (llm._cache.get(k) or {}).get("result")
            return isinstance(res, dict) and bool(res.get("parse_error"))

        parse_err = sum(map(_parse_error, lv_keys))
        table_dir = out / "tables" / f"{domain}_{level}"
        try:
            if table_dir.exists():
                shutil.rmtree(table_dir)
            table_dir.mkdir(parents=True)
            for src, recs in per_source.items():
                df = bundle.sources[src]
                ids = df["id"].tolist()
                attrs = sorted({a for (s_, a) in attr_index if s_ == src and a != "id"}
                               | {a for rec in recs.values() for a in rec})
                with open(table_dir / f"{src}.csv", "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["id", *attrs])
                    for i in sorted(recs):
                        rid = "" if ids[i] is None else str(ids[i])
                        w.writerow([rid, *(recs[i].get(a, "") for a in attrs)])
            sc = measure.score_tables(task_dir, table_dir)
        finally:
            if table_dir.exists():
                shutil.rmtree(table_dir)
        llm_attr = {a: v for a, v in sc["per_attribute"].items() if a in eligible}
        n_llm = sum(v["n"] for v in llm_attr.values())
        lv["export"] = {"llm_scope_cells_abstained": abstain, "abstained_by_attribute": dict(abstain_by_attr),
                        "multi_column_cells": n_multi,
                        "parse_error_answers_cached": parse_err,
                        "api_errors_this_level": sum(1 for f in ctl.failed.values() if f["level"] == level)}
        complete = not dry or (st["new_misses"] == 0 and st["pending_from_earlier_level"] == 0
                               and st["cached_api_error_entries"] == 0)
        if complete:
            lv["score"] = {
                **{k: sc[k] for k in ("accuracy", "accuracy_transformations", "record_coverage", "n_cells",
                                      "n_not_found", "n_correct", "n_transformation_cells")},
                "llm_scope": {"n": n_llm, "correct": sum(v["correct"] for v in llm_attr.values()),
                              "accuracy": round(sum(v["correct"] for v in llm_attr.values()) / n_llm, 4)
                              if n_llm else None},
                "per_category": sc["per_category"], "per_attribute": sc["per_attribute"],
            }
            if dry:
                lv["score"]["note"] = "dry run with every prompt of this level in the cache: the member's score"
            logger.info("%s/%s: accuracy %.4f transformations %s coverage %s n %d (API errors so far %d)",
                        domain, level, sc["accuracy"], sc["accuracy_transformations"], sc["record_coverage"],
                        sc["n_cells"], len(ctl.failed))
        else:
            lv["plumbing_check"] = {k: sc[k] for k in ("record_coverage", "n_cells", "n_not_found")}
            lv["plumbing_check"]["note"] = "dry run: cache misses abstain, so no accuracy is reported"
        lv["runtime_s"] = round(time.time() - t_level, 1)
        part["levels"][level] = lv

    # ---- domain totals -------------------------------------------------------------
    mt = max_tokens or 2048
    part["domain_totals"] = {
        "normalize_calls": dom_calls, "distinct_prompts": len(dom_keys), "hits": len(dom_hits),
        "cached_api_error_entries": len(recall_keys),
        "misses": planned_calls,
        "input_tokens_all_distinct": _sum(dom_tokens.values()),
        "cost_bounds": costs(planned_calls, planned_in, out_mean, mt),
        "cost_bounds_no_cache": costs(len(dom_tokens), _sum(dom_tokens.values()), out_mean, mt),
        "note": ("misses = distinct prompts a paid run sends (usable cache entries excluded, cached API-error "
                 "entries included); a prompt is asked at the first level that needs it; input tokens: tiktoken "
                 "o200k_base, system + user prompt + 9 framing tokens"),
    }
    part["api"] = {"attempts": ctl.attempts, "responses": ctl.responses, "usage": dict(ctl.usage),
                   "errors_not_cached": len(ctl.failed),
                   "errors_by_message": dict(Counter(f["error"][:120] for f in ctl.failed.values())),
                   "errors": [{"key": k, **v} for k, v in sorted(ctl.failed.items())]}
    if ctl.responses:
        use = ctl.usage
        part["api"]["usd_actual"] = round(use["input_tokens"] / 1e6 * PRICE_IN
                                          + use["output_tokens"] / 1e6 * PRICE_OUT, 4)
    part["repository_files_modified_during_run"] = measure.files_modified_since(measure.REPO, t_start)
    part["runtime_total_s"] = round(time.time() - t_start, 1)
    part["created_utc"] = datetime.now(timezone.utc).isoformat()
    out.mkdir(parents=True, exist_ok=True)
    part_path = out / (f"{domain}.dry_run.json" if dry else f"{domain}.json")
    part_path.write_text(json.dumps(part, indent=1, default=str) + "\n", encoding="utf-8")
    if not dry:
        with open(out / f"{domain}.ledger.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"created_utc": part["created_utc"], "attempts": ctl.attempts,
                                "responses": ctl.responses, "usage": dict(ctl.usage),
                                "usd_actual": part["api"].get("usd_actual", 0.0),
                                "errors_not_cached": len(ctl.failed)}) + "\n")
    if part["repository_files_modified_during_run"]:
        logger.warning("files modified in the repository during the run: %s",
                       part["repository_files_modified_during_run"])
    print(json.dumps({"domain": domain, "mode": part["mode"], "written": str(part_path),
                      "totals": part["domain_totals"],
                      "api": {k: part["api"][k] for k in ("attempts", "responses", "errors_not_cached")},
                      "scored_levels": [lv for lv, x in part["levels"].items() if "score" in x]},
                     indent=1, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--levels", default=",".join(LEVELS),
                    help="comma-separated subset of baseline,easy,medium,hard (always run in this order)")
    ap.add_argument("--cache-root", required=True,
                    help="folder (outside the repository) under which the YAML's relative cache_dir is resolved")
    ap.add_argument("--out", required=True, help="output folder outside the repository (results in <out>/llm_member/)")
    ap.add_argument("--locks-from", default=None, help="as in measure.py")
    ap.add_argument("--dry-run", action="store_true", help="no API call: cache hits, misses and cost bounds")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-new-calls", type=int, default=1500)
    ap.add_argument("--max-cap-usd", type=float, default=5.0,
                    help="stop before any call when the hard-cap cost of the domain's planned calls exceeds this")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="[%(asctime)s %(levelname)s] %(name)s - %(message)s")
    for noisy in ("httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
