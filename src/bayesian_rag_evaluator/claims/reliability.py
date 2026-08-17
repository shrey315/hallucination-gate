"""Source reliability. Low-trust chunks cannot be the sole SUPPORTED citation."""

from __future__ import annotations

from bayesian_rag_evaluator.models.schemas import EvidenceUnit


def apply_source_reliability(
    units: list[EvidenceUnit],
    mapping: dict[str, float] | None,
) -> list[EvidenceUnit]:
    """Stamp caller-supplied reliability onto units. Unknown ids stay at 1.0."""
    if not mapping:
        return units
    out: list[EvidenceUnit] = []
    for unit in units:
        value = None
        if unit.source_id and unit.source_id in mapping:
            value = mapping[unit.source_id]
        elif unit.source_id and ":" in unit.source_id:
            prefix, _, rest = unit.source_id.partition(":")
            if rest in mapping:
                value = mapping[rest]
            elif unit.source_id.split(":")[-1] in mapping:
                value = mapping[unit.source_id.split(":")[-1]]
            elif prefix in mapping:
                value = mapping[prefix]
        if value is None:
            out.append(unit)
            continue
        clamped = max(0.0, min(1.0, float(value)))
        out.append(unit.model_copy(update={"reliability": clamped}))
    return out


def reliability_of(unit: EvidenceUnit) -> float:
    return max(0.0, min(1.0, float(unit.reliability)))
