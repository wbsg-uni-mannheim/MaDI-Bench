"""Utilities for unified LLM call logging.

This module provides a small, provider-agnostic logger that captures:
- prompts/messages sent to the model
- raw responses
- timing and basic request params
- token usage metadata when available (best-effort across providers)

It is designed to work in both blocking/synchronous call paths and to be
reusable across modules. It does not depend on LangChain directly; it only
expects a "chat model" object and a "response" object that may expose
metadata in a few common places.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert provider/LC messages to a simple serializable structure.

    Tries to read common attributes (type/role, content) and falls back to str().
    """
    serialized: List[Dict[str, Any]] = []
    for msg in messages or []:
        msg_type = _safe_getattr(msg, "type", None) or _safe_getattr(
            msg, "role", None) or type(msg).__name__
        content = _safe_getattr(msg, "content", None)
        if content is None:
            try:
                content = str(msg)
            except Exception:
                content = "<unserializable>"
        serialized.append({"type": str(msg_type), "content": content})
    return serialized


def _extract_usage_metadata(response: Any) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of token usage from diverse response objects.

    Looks in common locations used by LangChain/OpenAI/Anthropic wrappers.
    """
    # LangChain OpenAI new adapters often expose `usage_metadata`
    usage = _safe_getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return usage

    # Some responses have a nested `response_metadata` with `token_usage` or `usage`
    meta = _safe_getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        token_usage = meta.get("token_usage") or meta.get("usage")
        if isinstance(token_usage, dict):
            return token_usage

    # OpenAI python client v1 sometimes exposes `usage`
    usage2 = _safe_getattr(response, "usage", None)
    if isinstance(usage2, dict):
        return usage2

    return None


def _get_model_name(chat_model: Any) -> str:
    return (
        _safe_getattr(chat_model, "model_name", None)
        or _safe_getattr(chat_model, "model", None)
        or _safe_getattr(chat_model, "name", None)
        or "unknown"
    )


def _usage_count(usage: Optional[Dict[str, Any]], *keys: str) -> int:
    if not usage:
        return 0
    for key in keys:
        value = usage.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


class ThreadedLLMTokenTracker:
    """Track token usage from LLM calls made outside the main step thread."""

    _instance: "ThreadedLLMTokenTracker | None" = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._main_thread_id: int | None = None
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._requests = 0

    @classmethod
    def get_instance(cls) -> "ThreadedLLMTokenTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self, *, main_thread_id: int | None = None) -> None:
        with self._lock:
            self._main_thread_id = main_thread_id
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._total_tokens = 0
            self._requests = 0

    def record(self, usage: Optional[Dict[str, Any]]) -> None:
        if not usage:
            return
        thread_id = threading.get_ident()
        with self._lock:
            if self._main_thread_id is not None and thread_id == self._main_thread_id:
                return

            prompt_tokens = _usage_count(usage, "input_tokens", "prompt_tokens")
            completion_tokens = _usage_count(usage, "output_tokens", "completion_tokens")
            total_tokens = _usage_count(usage, "total_tokens")
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens

            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._total_tokens += total_tokens
            self._requests += 1

    def get_and_reset(self) -> tuple[int, int, int, int]:
        with self._lock:
            values = (
                self._prompt_tokens,
                self._completion_tokens,
                self._total_tokens,
                self._requests,
            )
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._total_tokens = 0
            self._requests = 0
            self._main_thread_id = None
            return values


def get_threaded_llm_token_tracker() -> ThreadedLLMTokenTracker:
    return ThreadedLLMTokenTracker.get_instance()


@dataclass
class LLMCallRecord:
    """Structured record for a single LLM call."""

    timestamp: str
    row_index: int
    attempt: int
    provider_class: str
    model: str
    duration_ms: float
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    usage: Optional[Dict[str, Any]] = None
    request_messages: List[Dict[str, Any]] = field(default_factory=list)
    response_text_preview: Optional[str] = None
    phase: Optional[str] = None  # Training phase identifier (e.g., "faiss_small", "active", "random")


class LLMCallLogger:
    """Aggregator for unified, provider-agnostic LLM call logging.

    Usage:
        logger = LLMCallLogger()
        logger.record_call(...)
        logger.flush(write_artifact)  # optional, writes JSON files when desired

    - Always emits a structured info log for each call
    - Optionally writes aggregated artifacts via the provided writer
    """

    def __init__(self) -> None:
        self._records: List[LLMCallRecord] = []

    def record_call(
        self,
        *,
        chat_model: Any,
        messages: List[Any],
        response: Any,
        row_index: int,
        attempt: int,
        duration_ms: float,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> None:
        """Record a single blocking call's details and emit a structured log.

        Parameters
        ----------
        phase : str, optional
            Training phase identifier for token tracking (e.g., "faiss_small", "active", "random").
        """
        model_name = _get_model_name(chat_model)
        provider_class = type(chat_model).__name__
        usage = _extract_usage_metadata(response)
        get_threaded_llm_token_tracker().record(usage)

        # Build safe response preview (avoid huge payloads)
        try:
            response_text = _safe_getattr(response, "content", None)
            if response_text is None:
                response_text = str(response)
            preview = str(response_text)[:2000]
        except Exception:
            preview = "<unserializable>"

        rec = LLMCallRecord(
            timestamp=datetime.utcnow().isoformat() + "Z",
            row_index=row_index,
            attempt=attempt,
            provider_class=provider_class,
            model=model_name,
            duration_ms=float(duration_ms),
            temperature=temperature,
            max_tokens=max_tokens,
            usage=usage,
            request_messages=_serialize_messages(messages),
            response_text_preview=preview,
            phase=phase,
        )

        self._records.append(rec)

        # Emit to Python logging immediately so it shows up even without artifacts
        try:
            logger.info("LLM call: %s", json.dumps(rec.__dict__))
        except Exception:
            # Fallback if json serialization fails for any reason
            logger.info("LLM call (non-JSON serializable)")

    def flush(self, write_artifact: Callable[[str, Any], Any]) -> None:
        """Write aggregated artifacts (JSON list and usage summary).

        The provided writer should accept (filename, data) and persist it.
        """
        if not self._records:
            return

        # Write all calls as JSON array
        data = [r.__dict__ for r in self._records]
        write_artifact("llm_calls.json", data)

        # Create a small usage summary
        total_calls = len(self._records)
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        for r in self._records:
            if not r.usage:
                continue
            # Try multiple common keys
            total_input_tokens += (
                r.usage.get("input_tokens")
                or r.usage.get("prompt_tokens")
                or 0
            )
            total_output_tokens += (
                r.usage.get("output_tokens")
                or r.usage.get("completion_tokens")
                or 0
            )
            total_tokens += (
                r.usage.get("total_tokens")
                or (r.usage.get("input_tokens") or r.usage.get("prompt_tokens") or 0)
                + (r.usage.get("output_tokens")
                   or r.usage.get("completion_tokens") or 0)
            )

        summary = {
            "total_calls": total_calls,
            "total_input_tokens": int(total_input_tokens),
            "total_output_tokens": int(total_output_tokens),
            "total_tokens": int(total_tokens),
        }
        write_artifact("llm_usage_summary.json", summary)

        # Write per-phase usage summary if any records have phase set
        phase_summary = self.aggregate_by_phase()
        if phase_summary:
            write_artifact("llm_usage_summary_by_phase.json", phase_summary)

    def aggregate_by_phase(self) -> Dict[str, Dict[str, int]]:
        """Aggregate token usage by training phase.

        Returns a dict mapping phase name to usage stats:
        {
            "faiss_small": {"calls": 100, "input_tokens": 5000, "output_tokens": 1000, "total_tokens": 6000},
            "active": {"calls": 50, ...},
            ...
        }
        """
        phase_stats: Dict[str, Dict[str, int]] = {}

        for r in self._records:
            phase = r.phase or "unspecified"
            if phase not in phase_stats:
                phase_stats[phase] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }

            phase_stats[phase]["calls"] += 1

            if r.usage:
                input_tokens = (
                    r.usage.get("input_tokens")
                    or r.usage.get("prompt_tokens")
                    or 0
                )
                output_tokens = (
                    r.usage.get("output_tokens")
                    or r.usage.get("completion_tokens")
                    or 0
                )
                total = (
                    r.usage.get("total_tokens")
                    or (input_tokens + output_tokens)
                )
                phase_stats[phase]["input_tokens"] += int(input_tokens)
                phase_stats[phase]["output_tokens"] += int(output_tokens)
                phase_stats[phase]["total_tokens"] += int(total)

        return phase_stats
