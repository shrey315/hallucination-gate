from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Literal

from bayesian_rag_evaluator.adapters import normalize_context
from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.models.schemas import EvaluateRequest, EvaluateResponse, ModelType
from hallucination_gate.evidence import Evidence

Mode = Literal["rag", "fine_tuned"]
RetrieveFn = Callable[[str], Any]
GenerateFn = Callable[..., Any]


@dataclass
class GatedAnswer:
    """Safe output for the calling app. ``text`` is what users should see.

    In shadow mode ``text`` is the original model answer; ``gated_text`` is the
    counterfactual safe answer, and ``released`` / ``action`` are the enforce
    decision you would have made.

    ``release_authority`` is always claim status. BN fusion scores are
    diagnostic only and are **not** calibrated probabilities.
    """

    text: str
    released: bool
    action: str
    original: str
    reason: str
    request_id: str | None = None
    latency_ms: float | None = None
    claims: list[dict[str, Any]] = field(default_factory=list)
    raw: EvaluateResponse | None = None
    evidence_gap: str = "none"
    retrieval_quality: float | None = None
    release_authority: str = "claim_status"
    scores_are_calibrated: bool = False
    gated_text: str | None = None
    shadow: bool = False

    def __str__(self) -> str:
        return self.text

    @property
    def diagnostics(self) -> list[dict[str, Any]]:
        """Compact claim↔chunk view for debugging RAG integrations."""
        out: list[dict[str, Any]] = []
        for claim in self.claims:
            out.append(
                {
                    "claim": claim.get("text"),
                    "status": claim.get("status"),
                    "source_id": claim.get("source_id"),
                    "reason": claim.get("reason"),
                    "citation": claim.get("citation"),
                    "grounding_kind": claim.get("grounding_kind"),
                    "hop_source_ids": claim.get("hop_source_ids") or [],
                    "logic_flags": claim.get("logic_flags") or [],
                    "reliability": claim.get("reliability"),
                    "chunk_hits": [
                        {
                            "source_id": h.get("source_id"),
                            "status": h.get("status"),
                            "support": h.get("support_score"),
                            "contradiction": h.get("contradiction_score"),
                            "coverage": h.get("coverage"),
                            "reason": h.get("reason"),
                        }
                        for h in (claim.get("chunk_hits") or [])
                    ],
                }
            )
        return out


class HallucinationGate:
    """Framework-agnostic gate for any RAG or fine-tuned generator.

    It does not generate answers and does not depend on a dataset, vector DB,
    or model vendor. You pass a query, the model answer, and whatever evidence
    you have (retrieved chunks, KB text, images, PDFs, OCR, tables, audio).

    Modes (dataset-agnostic):
      - quality_mode=\"ci\" → heuristic smoke backends
      - quality_mode=\"quality\" → MiniLM + DeBERTa-small (default neural)
      - quality_mode=\"quality_plus\" → mpnet + DeBERTa-base (opt-in, heavier)
    Policy:
      - \"strict\" → max false-release lock
      - \"balanced\" → fewer over-refusals (uncertain rewrite + slightly softer support)
    Shadow:
      - shadow=True → return the original answer as ``text``; inspect
        ``gated_text`` / ``released`` as the counterfactual enforce decision.
    """

    def __init__(
        self,
        mode: Mode = "rag",
        *,
        strict: bool = True,
        use_heuristic: bool | None = None,
        learned_model_path: str | Path | None = None,
        embed_model: str | None = None,
        nli_model: str | None = None,
        quality_mode: str | None = None,
        policy: str | None = "balanced",
        align_contexts: bool = True,
        warm: bool = False,
        shadow: bool = False,
    ) -> None:
        self.mode = mode
        self.strict = strict
        self.shadow = shadow
        self._evaluator = DiagnosticEvaluator(
            use_heuristic=use_heuristic,
            learned_model_path=Path(learned_model_path) if learned_model_path else None,
            embed_model=embed_model,
            nli_model=nli_model,
            mode=quality_mode,
            policy=policy,
            align_contexts=align_contexts,
        )
        if warm:
            self.warm()

    def warm(self) -> None:
        """Preload embed/NLI backends to avoid cold-start latency spikes."""
        self._evaluator.warm()

    def check(
        self,
        query: str,
        answer: str,
        context: Any = None,
        kb: Any = None,
        *,
        evidence: Evidence | None = None,
        mode: Mode | None = None,
        images: Any = None,
        tables: Any = None,
        documents: Any = None,
        audio: Any = None,
        pdfs: list[str] | None = None,
        strict: bool | None = None,
        debug: bool = False,
        source_reliability: dict[str, float] | None = None,
    ) -> GatedAnswer:
        """Verify an existing model answer against caller-supplied evidence."""
        ev = evidence or Evidence(
            context=context,
            kb=kb,
            images=images or [],
            tables=tables or [],
            documents=documents,
            pdfs=pdfs,
            audio=audio,
        )
        if context is not None and evidence is not None:
            ev.context = context
        if kb is not None and evidence is not None:
            ev.kb = kb

        reliability = source_reliability or getattr(ev, "source_reliability", None) or {}
        request = EvaluateRequest(
            query=query,
            answer=answer,
            context_chunks=normalize_context(ev.context),
            kb_chunks=normalize_context(ev.kb),
            images=ev.image_inputs(),
            tables=ev.table_inputs(),
            documents=normalize_context(ev.documents),
            audio_transcripts=normalize_context(ev.audio),
            pdf_paths=ev.pdfs or [],
            model_type=ModelType(mode or self.mode),
            strict=self.strict if strict is None else strict,
            source_reliability=reliability,
        )
        result = self._evaluator.evaluate(request)
        gated = result.safe_answer
        original = result.gate.original_answer
        return GatedAnswer(
            text=original if self.shadow else gated,
            released=result.gate.released,
            action=result.gate.action.value,
            original=original,
            reason=result.gate.reason,
            request_id=result.request_id,
            latency_ms=result.latency_ms,
            claims=[c.model_dump(mode="json") for c in result.claims],
            raw=result if debug else None,
            evidence_gap=result.gate.evidence_gap.value,
            retrieval_quality=result.evidence.retrieval_quality,
            release_authority=result.release_authority,
            scores_are_calibrated=result.scores_are_calibrated,
            gated_text=gated,
            shadow=self.shadow,
        )

    def evaluate(
        self,
        samples: Any,
        *,
        metrics: list[str] | tuple[str, ...] | None = None,
        latency_budget: Any = None,
        baseline_path: Any = None,
        save_baseline_path: Any = None,
        fail_on_regression: bool = False,
        fail_on_latency: bool = True,
    ):
        """Dataset-level RAG quality report (metrics + retrieval + latency + regression)."""
        from hallucination_gate.eval import DEFAULT_METRICS, RAGEval

        names = list(metrics) if metrics is not None else list(DEFAULT_METRICS)
        return RAGEval(
            evaluator=self._evaluator,
            metrics=names,
            latency_budget=latency_budget,
        ).evaluate(
            samples,
            metrics=names,
            latency_budget=latency_budget,
            baseline_path=baseline_path,
            save_baseline_path=save_baseline_path,
            fail_on_regression=fail_on_regression,
            fail_on_latency=fail_on_latency,
        )

    def run(
        self,
        query: str,
        generate: GenerateFn,
        *,
        retrieve: RetrieveFn | None = None,
        context: Any = None,
        kb: Any = None,
        evidence: Evidence | None = None,
        mode: Mode | None = None,
        **generate_kwargs: Any,
    ) -> GatedAnswer:
        """Your retrieve (optional) → your generate → this gate."""
        resolved_mode = mode or self.mode
        if retrieve is not None and context is None:
            context = retrieve(query)
        answer = _call_generate(generate, query, context, kb, generate_kwargs)
        return self.check(
            query,
            answer,
            context=context,
            kb=kb,
            evidence=evidence,
            mode=resolved_mode,
        )

    def wrap(
        self,
        generate: GenerateFn,
        *,
        retrieve: RetrieveFn | None = None,
        kb: Any = None,
        mode: Mode | None = None,
        text_only: bool = True,
    ) -> Callable[..., str | GatedAnswer]:
        """Drop-in wrapper around an existing generate(query, ...) function."""

        def wrapped(query: str, *args: Any, context: Any = None, **kwargs: Any):
            extra_context = args[0] if args else context
            result = self.run(
                query,
                lambda q, **kw: generate(q, *args, **kw) if args else generate(q, **kwargs),
                retrieve=retrieve if extra_context is None else None,
                context=extra_context,
                kb=kb,
                mode=mode,
            )
            return result.text if text_only else result

        return wrapped

    def protect(self, fn: GenerateFn | None = None, *, text_only: bool = True):
        """Decorator for any RAG/fine-tune function.

        The function may return a string, ``(answer, context)``,
        ``(answer, context, kb)``, or a dict with those keys.
        """

        def decorator(func: GenerateFn) -> Callable[..., str | GatedAnswer]:
            @wraps(func)
            def inner(query: str, *args: Any, **kwargs: Any):
                output = func(query, *args, **kwargs)
                answer, context, kb = _unpack_output(output)
                result = self.check(query, answer, context=context, kb=kb)
                return result.text if text_only else result

            return inner

        if fn is not None:
            return decorator(fn)
        return decorator


def _call_generate(
    generate: GenerateFn,
    query: str,
    context: Any,
    kb: Any,
    extra: dict[str, Any],
) -> str:
    import inspect

    try:
        params = inspect.signature(generate).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs = dict(extra)
    if "context" in params and context is not None:
        kwargs.setdefault("context", context)
    if "kb" in params and kb is not None:
        kwargs.setdefault("kb", kb)
    if "documents" in params and context is not None:
        kwargs.setdefault("documents", context)
    try:
        output = generate(query, **kwargs)
    except TypeError:
        output = generate(query)
    answer, _, _ = _unpack_output(output)
    return answer


def _unpack_output(output: Any) -> tuple[str, Any, Any]:
    if isinstance(output, GatedAnswer):
        return output.original or output.text, None, None
    if isinstance(output, str):
        return output, None, None
    if isinstance(output, dict):
        answer = (
            output.get("answer")
            or output.get("text")
            or output.get("output")
            or output.get("response")
            or ""
        )
        context = (
            output.get("context")
            or output.get("documents")
            or output.get("sources")
            or output.get("chunks")
        )
        kb = output.get("kb") or output.get("knowledge_base")
        return str(answer), context, kb
    if isinstance(output, (tuple, list)) and output:
        answer = str(output[0])
        context = output[1] if len(output) > 1 else None
        kb = output[2] if len(output) > 2 else None
        return answer, context, kb
    return str(output), None, None
