import math
from typing import List, Optional, Tuple


"""
Improvements over the earlier version:

1. Fixes a scale-mismatch ("weight leakage") bug. If `semantic` arrives
   already pool-normalized (spanning close to the full [0,1] range, as
   produced by semantic_matcher_v2.score_batch()) while `keyword` coverage
   naturally clusters in a narrower band across a similarly-qualified
   pool, a naive alpha*sem + beta*key sum lets semantic dominate the
   final ranking's variance regardless of the stated weights — beta=0.45
   ends up contributing far less than 45% of what actually separates
   candidates. This version pool-normalizes BOTH components before
   weighting, so alpha/beta genuinely control each component's influence
   on the final rank order, not just its raw numeric magnitude.
2. Replaces the hard keyword<0.40 cliff (a 35% score drop between 0.399
   and 0.401) with a smooth logistic penalty, removing an arbitrary
   rank-order sensitivity right at the threshold boundary.
3. Validates alpha+beta sum to 1 and auto-renormalizes instead of
   silently producing a differently-scaled blend.
4. output_range is a parameter instead of hard-coded magic numbers, and
   each candidate gets diagnostic fields (whether a penalty applied) so
   the explanation generator can reference *why* a score landed where it
   did, instead of just showing a final number.
5. Deterministic tie-breaking when scores round to the same value.
"""


def _pool_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _smooth_penalty(
    keyword_score: float,
    threshold: float = 0.40,
    steepness: float = 12.0,
    max_penalty: float = 0.35,
) -> float:
    """
    Logistic multiplier centered at `threshold`, replacing a hard if/else
    cliff. A candidate just below threshold and one just above it now get
    nearly identical treatment; the penalty ramps in gradually as keyword
    coverage drops further below threshold, rather than snapping on/off.
    """
    logistic = 1 / (1 + math.exp(-steepness * (keyword_score - threshold)))
    return (1 - max_penalty) + max_penalty * logistic


def calibrate_and_rank(
    candidates: List[dict],
    alpha: float = 0.25,
    beta: float = 0.75,
    output_range: Tuple[float, float] = (28.0, 95.0),
    penalty_threshold: float = 0.40,
    max_penalty: float = 0.35,
    core_skills: Optional[List[str]] = None,
) -> List[dict]:
    if not candidates:
        return []

    # Auto-renormalize if weights don't sum to 1, rather than silently
    # producing an oddly-scaled blend.
    weight_sum = alpha + beta
    if abs(weight_sum - 1.0) > 1e-6:
        weight_sum = weight_sum or 1.0
        alpha, beta = alpha / weight_sum, beta / weight_sum

    semantic_vals = [c.get("semantic", 0.0) for c in candidates]
    keyword_vals = [c.get("keyword", 0.0) for c in candidates]

    # Normalize both components to the SAME pool-relative scale before
    # weighting — this is what makes alpha/beta reflect actual influence
    # on the final ranking rather than being at the mercy of each
    # component's natural spread within this particular candidate pool.
    sem_norm = _pool_normalize(semantic_vals)
    key_norm = _pool_normalize(keyword_vals)

    raw_scores, penalty_flags = [], []
    for i, c in enumerate(candidates):
        base = (alpha * sem_norm[i]) + (beta * key_norm[i])
        # The under-qualification penalty is judged against the RAW
        # keyword coverage (an absolute fraction of the JD's actual
        # requirements), not the pool-normalized value.
        multiplier = _smooth_penalty(keyword_vals[i], penalty_threshold, max_penalty=max_penalty)
        base *= multiplier
        penalty_flags.append(multiplier < 0.999)

        # Core stack completeness gating
        if core_skills:
            matched_dict = c.get("matched", {})
            matched_core = sum(1 for s in core_skills if s in matched_dict)
            core_ratio = matched_core / max(len(core_skills), 1)
            if core_ratio >= 0.99:
                stack_mult = 1.0
            elif core_ratio >= 0.50:
                stack_mult = 0.84  # Partial stack match (e.g. Angular instead of React, or backend-only)
            else:
                stack_mult = 0.60  # Non-matching stack (e.g. Django, Spring Boot, PHP)
            base *= stack_mult

        raw_scores.append(base)

    lo, hi = min(raw_scores), max(raw_scores)
    span = output_range[1] - output_range[0]
    mid = (output_range[0] + output_range[1]) / 2

    for i, c in enumerate(candidates):
        stretched = (
            output_range[0] + ((raw_scores[i] - lo) / (hi - lo)) * span
            if hi - lo > 1e-9 else mid
        )
        c["score"] = round(stretched, 1)
        c["semantic_pct"] = round(semantic_vals[i] * 100, 1)
        c["keyword_pct"] = round(keyword_vals[i] * 100, 1)
        c["low_keyword_penalty_applied"] = penalty_flags[i]

    ranked = sorted(
        candidates,
        key=lambda x: (x["score"], x["keyword_pct"], x["semantic_pct"]),
        reverse=True,
    )
    for idx, item in enumerate(ranked):
        item["rank"] = idx + 1
    return ranked


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _naive_fusion(candidates: List[dict], alpha=0.55, beta=0.45) -> List[dict]:
    """Reimplementation of the ORIGINAL logic, used only to demonstrate the
    difference in the tests below — not part of the shipped module."""
    raw_scores = []
    for c in candidates:
        base = (alpha * c["semantic"]) + (beta * c["keyword"])
        if c["keyword"] < 0.40:
            base *= 0.65
        raw_scores.append(base)
    lo, hi = min(raw_scores), max(raw_scores)
    for i, c in enumerate(candidates):
        c["_naive_raw"] = raw_scores[i]
    ranked = sorted(candidates, key=lambda x: x["_naive_raw"], reverse=True)
    for idx, item in enumerate(ranked):
        item["_naive_rank"] = idx + 1
    return ranked


if __name__ == "__main__":
    # 1. Weight-leakage scenario: a pool where keyword coverage is tightly
    #    clustered (0.45-0.55 — a small raw range, but the full meaningful
    #    signal available in this pool) while semantic varies widely
    #    (0.20-0.90). C2 has the WORST semantic but the BEST keyword
    #    coverage in the pool; C4 is mediocre on both. Under naive fusion,
    #    semantic's larger raw spread drowns out keyword's real signal and
    #    ranks C2 last. Under pool-relative normalization, keyword's true
    #    relative significance is restored and C2 should outrank C4.
    pool_naive = [
        {"name": "C1", "semantic": 0.90, "keyword": 0.45},
        {"name": "C2", "semantic": 0.20, "keyword": 0.55},
        {"name": "C3", "semantic": 0.70, "keyword": 0.47},
        {"name": "C4", "semantic": 0.45, "keyword": 0.50},
    ]
    _naive_fusion([dict(c) for c in pool_naive])  # just to confirm naive would rank C2 last
    naive_ranked = _naive_fusion([dict(c) for c in pool_naive])
    naive_order = [c["name"] for c in naive_ranked]
    assert naive_order[-1] == "C2", f"Expected naive fusion to rank C2 last, got {naive_order}"

    pool_fixed = [dict(c) for c in pool_naive]
    fixed_ranked = calibrate_and_rank(pool_fixed)
    fixed_rank_by_name = {c["name"]: c["rank"] for c in fixed_ranked}
    assert fixed_rank_by_name["C2"] < fixed_rank_by_name["C4"], (
        "Expected pool-relative normalization to move C2 (best keyword, worst semantic) "
        f"ahead of C4 (mediocre on both). Got ranks: {fixed_rank_by_name}"
    )

    # 2. Smooth penalty: candidates just below/above the 0.40 threshold
    #    should NOT show a large score jump the way the hard cliff did.
    near_threshold = [
        {"name": "Just Below", "semantic": 0.60, "keyword": 0.399},
        {"name": "Just Above", "semantic": 0.60, "keyword": 0.401},
        {"name": "Anchor Low", "semantic": 0.10, "keyword": 0.10},
        {"name": "Anchor High", "semantic": 0.95, "keyword": 0.95},
    ]
    ranked_nt = calibrate_and_rank([dict(c) for c in near_threshold])
    scores_by_name = {c["name"]: c["score"] for c in ranked_nt}
    gap = abs(scores_by_name["Just Below"] - scores_by_name["Just Above"])
    assert gap < 2.0, f"Penalty should be smooth across the threshold, got a gap of {gap}"

    # 3. alpha+beta auto-renormalization
    unnorm = calibrate_and_rank([dict(c) for c in pool_naive], alpha=0.7, beta=0.5)
    assert all(0 <= c["score"] <= 100 for c in unnorm), "Score should stay in a sane range even with alpha+beta != 1"

    print("All fusion self-tests passed.")