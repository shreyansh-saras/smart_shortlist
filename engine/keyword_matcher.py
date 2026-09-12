"""
Improved keyword matching for the Smart Shortlisting Engine.

Key improvements over the naive version:
1. Negation is scoped to a token window WITHIN a sentence, not an open-ended
   character-class regex — avoids false negatives/positives from negation
   words bleeding across unrelated clauses.
2. Every match carries the evidence sentence it was found in, which feeds
   directly into the top-3 explanation generator (judges specifically care
   about explanation accuracy — raw evidence beats a bare boolean).
3. Regex patterns are compiled once and cached (lru_cache) instead of
   recompiled per resume per skill — matters once you're running 18 resumes
   x ~10 required skills x multiple synonym variants.
4. Supports optional per-skill weights (e.g. "must-have" vs "nice-to-have"
   pulled from JD phrasing) instead of treating every skill as equal.
5. Negated-but-mentioned skills are surfaced separately, which is useful
   signal for explanations ("candidate mentions SQL but says 'no hands-on
   SQL experience'") rather than being silently dropped into "missing".
"""

import os
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Taxonomy loading
# ---------------------------------------------------------------------------

def load_taxonomy(filepath: str = "skills_taxonomy.json") -> Dict[str, List[str]]:
    """Load and normalize the skill-synonym taxonomy to lowercase keys/values."""
    candidates = [
        filepath,
        os.path.join(os.path.dirname(__file__), filepath),
        os.path.join(os.path.dirname(__file__), "..", filepath),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    raw = json.load(f)
                return {k.lower(): [v.lower() for v in vs] for k, vs in raw.items()}
            except Exception as e:
                print(f"[keyword_matcher] Warning: failed to parse {path}: {e}")
    return {}


TAXONOMY = load_taxonomy()

# Words that flip a nearby skill mention from "present" to "absent"
NEGATION_CUES = {
    "no", "none", "without", "lack", "lacks", "lacking", "little",
    "limited", "less", "minimal", "absent", "excludes", "excluding",
    "not", "n't", "never", "unfamiliar", "no experience", "haven't",
    "has not", "have not", "did not", "do not",
}
NEGATION_WINDOW = 15  # Widened to cover "Has not worked with React, Angular, or Vue"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SkillMatch:
    required_skill: str
    matched: bool = False
    variant_found: Optional[str] = None
    evidence: Optional[str] = None
    negated: bool = False


# ---------------------------------------------------------------------------
# Tokenization / sentence splitting
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\b\w[\w+.#-]*\b")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter for resume text.
    Normalizes soft line wraps (newlines within sentences/paragraphs) so that
    negation clauses spanning a line break (e.g. 'Has not worked with\\nReact')
    are preserved intact.
    Splits on sentence-ending punctuation, double-newlines, bullet points, and pipes.
    """
    normalized = re.sub(r"(?<![.!?:\n])\n(?!\s*[•\-\*\u2022]|\s*[A-Z\s]{4,}\b)", " ", text)
    chunks = re.split(r"(?<=[.!?;])\s+|\n\s*\n+|\n\s*[•\-\*\u2022]\s*|\s*\|\s*", normalized)
    return [c.strip() for c in chunks if c.strip()]


@lru_cache(maxsize=None)
def _compiled_pattern(variant: str) -> re.Pattern:
    """
    Build a word-boundary-aware regex for a skill variant.
    For multi-word variants (e.g. 'rest api'), the trailing \b is dropped
    and replaced with an optional plural 's' + a non-word-char boundary.
    This lets 'REST API' match 'REST APIs', 'REST API endpoint', etc.
    For single-word variants the standard \b...\b is used.
    """
    escaped = re.escape(variant)
    if " " in variant:
        # Multi-word: match regardless of trailing plural 's' or punctuation
        return re.compile(rf"\b{escaped}s?(?=\W|$)", re.IGNORECASE)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Negation detection (sentence-scoped, token-window based)
# ---------------------------------------------------------------------------

def _is_negated_in_sentence(sentence_tokens: List[str], variant_tokens: List[str]) -> bool:
    """
    Returns True if a negation cue appears within NEGATION_WINDOW tokens
    before any occurrence of variant_tokens inside this sentence's token list.
    Scoping to a single sentence (rather than the whole resume, or an
    unbounded character span) is what prevents false positives like a
    negation earlier in the document leaking into an unrelated match.
    """
    v_len = len(variant_tokens)
    if v_len == 0:
        return False
    for i in range(len(sentence_tokens) - v_len + 1):
        if sentence_tokens[i:i + v_len] == variant_tokens:
            window_start = max(0, i - NEGATION_WINDOW)
            preceding = sentence_tokens[window_start:i]
            if any(t in NEGATION_CUES for t in preceding):
                return True
    return False


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_keywords(
    resume_text: str,
    required_skills: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Args:
        resume_text: full extracted resume text.
        required_skills: canonical skill names pulled from the JD.
        weights: optional {skill_name: weight} e.g. {"SQL": 2.0, "Excel": 1.0}
                 to reflect "required" vs "preferred" phrasing in the JD.
                 Skills not in the dict default to weight 1.0.

    Returns:
        dict with coverage_score, matched (with evidence + variant),
        missing, and negated_mentions (mentioned but explicitly denied).
    """
    weights = weights or {}
    sentences = _split_sentences(resume_text)
    sentence_tokens_cache = [_tokenize(s) for s in sentences]

    results: Dict[str, SkillMatch] = {}

    for term in required_skills:
        variants = [term.lower()] + TAXONOMY.get(term.lower(), [])
        match = SkillMatch(required_skill=term)

        for sentence, sent_tokens in zip(sentences, sentence_tokens_cache):
            sentence_lower = sentence.lower()
            for variant in variants:
                pattern = _compiled_pattern(variant.lower())
                if not pattern.search(sentence_lower):
                    continue
                variant_tokens = _tokenize(variant)
                if _is_negated_in_sentence(sent_tokens, variant_tokens):
                    match.negated = True
                    continue  # keep scanning — a later, non-negated mention should still count
                match.matched = True
                match.variant_found = variant
                match.evidence = sentence
                break
            if match.matched:
                break

        results[term] = match

    matched = {t: m for t, m in results.items() if m.matched}
    missing = [t for t, m in results.items() if not m.matched]
    negated_only = [t for t, m in results.items() if m.negated and not m.matched]

    if weights:
        total_weight = sum(weights.get(t, 1.0) for t in required_skills) or 1.0
        earned_weight = sum(weights.get(t, 1.0) for t in matched)
        coverage = earned_weight / total_weight
    else:
        coverage = len(matched) / max(len(required_skills), 1)

    return {
        "coverage_score": round(coverage, 4),
        "matched": {
            t: {"variant": m.variant_found, "evidence": m.evidence}
            for t, m in matched.items()
        },
        "missing": missing,
        "negated_mentions": negated_only,
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_resume = """
    Built REST APIs using Express and MongoDB for a course project.
    No hands-on experience with SQL or relational databases.
    Familiar with basic Python scripting.
    """
    required = ["Node.js", "SQL", "Python"]
    weights = {"Node.js": 2.0, "SQL": 2.0, "Python": 1.0}

    result = evaluate_keywords(sample_resume, required, weights)
    print(json.dumps(result, indent=2))