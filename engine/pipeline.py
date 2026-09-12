import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

# Ensure parent directory is in sys.path for robust imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from engine.parser import extract_text_from_pdf, parse_resume_content
from engine.keyword_matcher import evaluate_keywords
from engine.explainer import build_evidence_from_matcher_output, generate_explanation
from engine.semantic_matcher import LocalSemanticMatcher
from engine.hybrid_scorer import calibrate_and_rank


def _process_single_resume_worker(
    item: Tuple[str, Union[bytes, str], List[str], Optional[Dict[str, float]]]
) -> dict:
    """
    Top-level module worker function for ProcessPoolExecutor.
    Performs CPU-bound PDF extraction, section chunking, keyword matching,
    and evidence formatting independently per resume in a single pass.
    """
    name, pdf_source, required_skills, skill_weights = item
    try:
        raw_text, sections = parse_resume_content(pdf_source)
        kw_res = evaluate_keywords(raw_text, required_skills, weights=skill_weights)
        evidence = build_evidence_from_matcher_output(kw_res["matched"])
        return {
            "name": name,
            "raw_text": raw_text,
            "sections": sections,
            "keyword": kw_res["coverage_score"],
            "matched": kw_res["matched"],
            "missing": kw_res["missing"],
            "negated_mentions": kw_res.get("negated_mentions", []),
            "evidence": evidence,
        }
    except Exception as e:
        return {
            "name": name,
            "raw_text": "",
            "sections": {},
            "keyword": 0.0,
            "matched": {},
            "missing": required_skills,
            "negated_mentions": [],
            "evidence": {},
            "error": str(e),
        }


def parallel_parse_and_keyword_match(
    resume_items: List[Tuple[str, Union[bytes, str]]],
    required_skills: List[str],
    skill_weights: Optional[Dict[str, float]] = None,
    max_workers: Optional[int] = None,
) -> List[dict]:
    """
    Parallelize PDF parsing and keyword matching across resumes using ProcessPoolExecutor.
    Independent and strictly CPU-bound.
    """
    if not resume_items:
        return []

    workers = max_workers or os.cpu_count() or 1
    workers = max(1, min(workers, len(resume_items)))

    worker_payloads = [
        (name, source, required_skills, skill_weights)
        for name, source in resume_items
    ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_process_single_resume_worker, worker_payloads))

    return results


def run_smart_shortlist_pipeline(
    jd_source: Union[bytes, str],
    resume_sources: Dict[str, Union[bytes, str]],
    required_skills: List[str],
    skill_weights: Optional[Dict[str, float]] = None,
    semantic_matcher: Optional[LocalSemanticMatcher] = None,
    alpha: float = 0.25,
    beta: float = 0.75,
    core_skills: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[List[dict], str, Dict[str, float]]:
    """
    Full end-to-end pipeline:
    1. Extract JD text.
    2. Parallel CPU processing (PDF parse + section chunk + keyword match) across CPU cores.
    3. Single batched semantic embedding inference on main process (with disk cache).
    4. Calibrated score fusion, stack-aware gating, and normalization.
    5. Top candidates explanation generation.

    Returns:
        (ranked_candidates, jd_text, timing_stats)
    """
    t0 = time.perf_counter()
    matcher = semantic_matcher or LocalSemanticMatcher()

    # Determine core skills for stack gating if not explicitly provided
    if core_skills is None and skill_weights:
        # Skills with weight > 1.0 are designated must-haves
        core_skills = [s for s, w in skill_weights.items() if w > 1.0]

    # 1. Parse JD
    jd_text = extract_text_from_pdf(jd_source) if not isinstance(jd_source, str) or jd_source.endswith(".pdf") else jd_source
    t_jd = time.perf_counter()

    # 2. Parallel PDF parsing + Keyword matching
    resume_items = list(resume_sources.items())
    candidates = parallel_parse_and_keyword_match(
        resume_items,
        required_skills=required_skills,
        skill_weights=skill_weights,
    )
    t_cpu = time.perf_counter()

    # 3. Single batched semantic score calculation
    candidates_dict = {c["name"]: c["sections"] for c in candidates}
    sem_scores = matcher.score_batch(jd_text, candidates_dict)

    for c in candidates:
        c["semantic"] = sem_scores.get(c["name"], 0.0)
    t_sem = time.perf_counter()

    # 4. Calibrated Hybrid Fusion with stack gating
    ranked = calibrate_and_rank(candidates, alpha=alpha, beta=beta, core_skills=core_skills)
    t_rank = time.perf_counter()

    # 5. Explanations for top candidates
    for i, c in enumerate(ranked):
        c["explanation"] = generate_explanation(c, skill_weights=skill_weights)

    timing_stats = {
        "jd_parse_sec": round(t_jd - t0, 3),
        "parallel_cpu_sec": round(t_cpu - t_jd, 3),
        "batched_semantic_sec": round(t_sem - t_cpu, 3),
        "ranking_sec": round(t_rank - t_sem, 3),
        "total_sec": round(time.perf_counter() - t0, 3),
    }

    return ranked, jd_text, timing_stats
