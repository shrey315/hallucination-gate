"""Optional LLM judge for the UNCERTAIN band only.

Off by default. Set HALLUCINATION_GATE_JUDGE=1 and an API key to escalate
borderline claims. Conservative NLI remains the default lock.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from bayesian_rag_evaluator.models.schemas import ClaimResult, ClaimVerdict, EvidenceUnit

_PROMPT = """You are a grounding judge. Decide if the CLAIM is supported by EVIDENCE.
Reply with JSON only: {{"verdict":"supported"|"unsupported"|"contradicted","confidence":0.0}}
CLAIM: {claim}
EVIDENCE: {evidence}
"""


def judge_enabled() -> bool:
    flag = os.getenv("HALLUCINATION_GATE_JUDGE", "").lower()
    if flag not in {"1", "true", "yes"}:
        return False
    return bool(
        os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("HALLUCINATION_GATE_JUDGE_URL")
    )


def refine_uncertain_claims(
    claims: list[ClaimResult],
    units: list[EvidenceUnit],
) -> list[ClaimResult]:
    if not judge_enabled():
        return claims
    evidence = "\n".join(u.content for u in units)[:6000]
    out: list[ClaimResult] = []
    for claim in claims:
        if claim.status != ClaimVerdict.UNCERTAIN:
            out.append(claim)
            continue
        verdict = _ask(claim.text, evidence)
        if verdict is None:
            out.append(claim)
            continue
        out.append(claim.model_copy(update={"status": verdict}))
    return out


def _ask(claim: str, evidence: str) -> ClaimVerdict | None:
    url = os.getenv("HALLUCINATION_GATE_JUDGE_URL")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    body_prompt = _PROMPT.format(claim=claim, evidence=evidence)
    try:
        if url:
            raw = _post_json(url, {"prompt": body_prompt}, {})
        elif anthropic_key:
            raw = _post_json(
                "https://api.anthropic.com/v1/messages",
                {
                    "model": os.getenv("HALLUCINATION_GATE_JUDGE_MODEL", "claude-sonnet-4-5"),
                    "max_tokens": 80,
                    "messages": [{"role": "user", "content": body_prompt}],
                },
                {
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                },
            )
        elif openai_key:
            raw = _post_json(
                "https://api.openai.com/v1/chat/completions",
                {
                    "model": os.getenv("HALLUCINATION_GATE_JUDGE_MODEL", "gpt-4o-mini"),
                    "messages": [{"role": "user", "content": body_prompt}],
                    "max_tokens": 80,
                },
                {"Authorization": f"Bearer {openai_key}"},
            )
        else:
            return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return None
    return _parse_verdict(raw)


def _post_json(url: str, payload: dict, headers: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_verdict(raw: dict) -> ClaimVerdict | None:
    text = ""
    if "content" in raw and raw["content"]:
        text = raw["content"][0].get("text", "")
    elif "choices" in raw:
        text = raw["choices"][0]["message"]["content"]
    elif "verdict" in raw:
        text = json.dumps(raw)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    data = json.loads(text[start : end + 1])
    value = str(data.get("verdict", "")).lower()
    mapping = {
        "supported": ClaimVerdict.SUPPORTED,
        "unsupported": ClaimVerdict.UNSUPPORTED,
        "contradicted": ClaimVerdict.CONTRADICTED,
    }
    return mapping.get(value)
