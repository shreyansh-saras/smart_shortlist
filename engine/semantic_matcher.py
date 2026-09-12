import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class LocalSemanticMatcher:
    """
    Offline local semantic matching engine using Sentence Transformers.

    Improvements over the earlier version (this revision):
    1. TRUE cross-candidate batching (carried over from the previous
       revision). score_batch() flattens every (candidate, section, text)
       triple across the WHOLE pool and calls model.encode() exactly once,
       instead of once per candidate.
    2. In-memory embedding cache keyed by a hash of the text (carried over).
    3. NEW — disk-persisted cache. Pass `cache_path` to __init__ and the
       cache is loaded from disk at startup and saved back after every
       score_batch() call. This means re-running the tool in a fresh
       process (app restart, a judge re-launching the demo, a second
       hackathon run against the same resume set) skips re-encoding any
       resume whose text hasn't changed — the single biggest recurring
       cost after the one-time model load.
    4. NEW — embeddings are now stored and compared as plain numpy arrays
       instead of torch tensors. This is what makes the disk cache safe:
       pickling a torch tensor ties the cache file to a specific torch
       build, and unpickling it requires torch to even be importable.
       Plain numpy arrays have neither problem, and cosine similarity is
       computed locally with `_cosine_sim()` instead of relying on
       sentence_transformers.util.cos_sim.
    5. Explicit batch_size passed to every encode() call.
    6. Fixes a crash carried over from earlier versions: section_parser.py
       can return "_unclassified_headers" as a List[str], not a string —
       explicitly filtered to text-only, resume-content sections.
    7. Pool-relative normalization (unchanged in spirit): min-max stretches
       scores across the actual candidate pool so the ranking has real
       separation, per the rubric's explicit call for score spread.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        section_weights: Optional[Dict[str, float]] = None,
        batch_size: int = 32,
        cache_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Default disk cache path: data/embeddings_cache.pkl in project directory
        if cache_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self._cache_path = os.path.join(base_dir, "data", "embeddings_cache.pkl")
        else:
            self._cache_path = cache_path

        # LAZY-LOADED: do not instantiate SentenceTransformer until actually needed for encoding
        self._model: Optional[SentenceTransformer] = None

        self._cached_jd_text: Optional[str] = None
        self._cached_jd_emb: Optional[np.ndarray] = None

        # text_hash -> np.ndarray embedding. Loaded from disk if cache file exists.
        self._embedding_cache: Dict[str, np.ndarray] = {}
        if self._cache_path:
            self._load_cache(self._cache_path)

        self.section_weights = section_weights or {
            "experience": 0.35,
            "projects": 0.35,   # Equal to experience — student resumes only have projects
            "skills": 0.20,
            "education": 0.05,
            "general": 0.15,
        }
        # Structural metadata keys that are not scorable resume content
        self._non_text_keys = {"_unclassified_headers"}

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the underlying SentenceTransformer model only when encoding uncached text."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Could not load '{self.model_name}' in offline mode. "
                    "Download it once with internet access before the event "
                    "(see pre-demo checklist) so it's cached locally."
                ) from e
        return self._model

    @model.setter
    def model(self, value: SentenceTransformer) -> None:
        self._model = value

    # ------------------------------------------------------------------
    # Disk cache persistence
    # ------------------------------------------------------------------

    def _load_cache(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict):
                self._embedding_cache = loaded
        except Exception as e:
            print(f"[semantic_matcher] Warning: failed to load cache '{path}': {e}. Starting empty.")

    def save_cache(self, path: Optional[str] = None) -> None:
        """
        Persist the in-memory embedding cache to disk atomically.
        Called automatically at the end of score_batch().
        """
        target = path or self._cache_path
        if not target:
            return
        try:
            target_dir = os.path.dirname(os.path.abspath(target))
            os.makedirs(target_dir, exist_ok=True)
            tmp_target = f"{target}.tmp"
            with open(tmp_target, "wb") as f:
                pickle.dump(self._embedding_cache, f)
            os.replace(tmp_target, target)
        except Exception as e:
            print(f"[semantic_matcher] Warning: failed to save cache '{target}': {e}")

    # ------------------------------------------------------------------
    # Similarity (numpy-only, no torch dependency for cache portability)
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float).flatten()
        b = np.asarray(b, dtype=float).flatten()
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    # ------------------------------------------------------------------
    # Encoding + cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _encode_with_cache(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Encodes only the texts NOT already in the (in-memory, possibly
        disk-loaded) cache, in a single batched call, then merges with
        cached hits. Returns text -> embedding for easy lookup.
        """
        unique_texts = list(dict.fromkeys(texts))  # de-dupe, preserve order
        hashes = {t: self._text_hash(t) for t in unique_texts}

        to_encode = [t for t in unique_texts if hashes[t] not in self._embedding_cache]

        if to_encode:
            new_embeddings = self.model.encode(
                to_encode,
                convert_to_numpy=True,
                batch_size=self.batch_size,
            )
            for t, emb in zip(to_encode, new_embeddings):
                self._embedding_cache[hashes[t]] = np.asarray(emb, dtype=float)

        return {t: self._embedding_cache[hashes[t]] for t in unique_texts}

    def _get_jd_embedding(self, jd_text: str) -> np.ndarray:
        """Cache the JD embedding — it's identical across every resume in a run."""
        if self._cached_jd_text != jd_text or self._cached_jd_emb is None:
            self._cached_jd_text = jd_text
            jd_hash = self._text_hash(jd_text)
            if jd_hash not in self._embedding_cache:
                emb = self.model.encode(jd_text, convert_to_numpy=True)
                self._embedding_cache[jd_hash] = np.asarray(emb, dtype=float)
            self._cached_jd_emb = self._embedding_cache[jd_hash]
        return self._cached_jd_emb

    # ------------------------------------------------------------------
    # Single-resume path (kept for callers scoring one resume ad hoc;
    # score_batch() below is the efficient path for a full run)
    # ------------------------------------------------------------------

    def _score_sections_dict(self, jd_emb: np.ndarray, sections: Dict[str, object]) -> float:
        text_sections = {
            k: v for k, v in sections.items()
            if k not in self._non_text_keys and isinstance(v, str) and v.strip()
        }
        if not text_sections:
            return 0.0

        names = list(text_sections.keys())
        texts = [text_sections[n] for n in names]
        emb_by_text = self._encode_with_cache(texts)

        weighted_sim, total_weight = 0.0, 0.0
        for name, text in zip(names, texts):
            weight = self.section_weights.get(name, 0.20)
            sim_clipped = max(0.0, min(1.0, self._cosine_sim(jd_emb, emb_by_text[text])))
            weighted_sim += sim_clipped * weight
            total_weight += weight

        return float(weighted_sim / total_weight) if total_weight > 0 else 0.0

    def compute_similarity(self, jd_text: str, sections: Union[str, Dict[str, object]]) -> float:
        """
        Args:
            jd_text: raw JD text.
            sections: either chunked sections dict (from section_parser.chunk_sections)
                      or a raw resume string.
        Returns: raw cosine similarity in [0, 1] for a SINGLE resume.
                 For ranking a batch, use score_batch() instead — it
                 encodes every candidate's sections in one shared batched
                 call, gives pool-relative normalization, and can persist
                 its cache to disk.
        """
        if not jd_text or not jd_text.strip():
            return 0.0

        jd_emb = self._get_jd_embedding(jd_text)

        if isinstance(sections, str):
            if not sections.strip():
                return 0.0
            emb_by_text = self._encode_with_cache([sections])
            return float(max(0.0, min(1.0, self._cosine_sim(jd_emb, emb_by_text[sections]))))

        if isinstance(sections, dict):
            return self._score_sections_dict(jd_emb, sections)

        return 0.0

    # ------------------------------------------------------------------
    # Batch path — the efficient entry point for a full ranking run
    # ------------------------------------------------------------------

    def score_batch(
        self, jd_text: str, candidates: Dict[str, Union[str, Dict[str, object]]]
    ) -> Dict[str, float]:
        """
        Score an entire batch of resumes against one JD.
        Every candidate's section text is flattened into ONE list and encoded
        in a single model.encode() call for efficiency. If `cache_path` was
        set at construction, the cache is saved to disk at the end of this call.

        Returns {candidate_name: raw cosine similarity in [0, 1]}.
        NOTE: scores are RAW (not pool-normalized). Pool normalization is done
        once in hybrid_scorer.calibrate_and_rank() alongside keyword scores,
        so both signals are stretched on the same relative scale together.
        """
        if not jd_text or not jd_text.strip() or not candidates:
            return {name: 0.0 for name in candidates}

        jd_emb = self._get_jd_embedding(jd_text)

        # --- Flatten every candidate's scorable text into one list ---
        flat: List[Tuple[str, str, str]] = []
        for name, sections in candidates.items():
            if isinstance(sections, str):
                if sections.strip():
                    flat.append((name, "general", sections))
            elif isinstance(sections, dict):
                for sec_name, sec_text in sections.items():
                    if sec_name in self._non_text_keys:
                        continue
                    if isinstance(sec_text, str) and sec_text.strip():
                        flat.append((name, sec_name, sec_text))

        if not flat:
            return {name: 0.0 for name in candidates}

        all_texts = [t for (_, _, t) in flat]
        emb_by_text = self._encode_with_cache(all_texts)

        # --- Fold section similarities back into one score per candidate ---
        weighted_sim: Dict[str, float] = {}
        total_weight: Dict[str, float] = {}
        for cand_name, sec_name, text in flat:
            weight = self.section_weights.get(sec_name, 0.20)
            sim_clipped = max(0.0, min(1.0, self._cosine_sim(jd_emb, emb_by_text[text])))
            weighted_sim[cand_name] = weighted_sim.get(cand_name, 0.0) + sim_clipped * weight
            total_weight[cand_name] = total_weight.get(cand_name, 0.0) + weight

        raw_scores = {
            name: (weighted_sim[name] / total_weight[name]) if total_weight.get(name, 0.0) > 0 else 0.0
            for name in candidates
        }
        for name in candidates:
            raw_scores.setdefault(name, 0.0)

        # Save cache if configured — do this before returning
        if self._cache_path:
            self.save_cache()

        return raw_scores

    @staticmethod
    def _normalize_pool(raw_scores: Dict[str, float]) -> Dict[str, float]:
        if not raw_scores:
            return {}
        values = np.array(list(raw_scores.values()), dtype=float)
        lo, hi = values.min(), values.max()
        if hi - lo < 1e-9:
            # Every candidate scored identically (degenerate case) —
            # nothing meaningful to stretch, so don't fabricate spread.
            return {k: 0.5 for k in raw_scores}
        return {k: float((v - lo) / (hi - lo)) for k, v in raw_scores.items()}


def semantic_score(
    resume_text: str, jd_text: str, matcher: Optional["LocalSemanticMatcher"] = None
) -> float:
    """
    Convenience wrapper for a single comparison. Pass a pre-built `matcher`
    when scoring more than one resume in the same run — building a new
    LocalSemanticMatcher per call reloads the full model from disk every
    time, which is the single biggest perf trap in a batch pipeline. For
    scoring an entire candidate pool, prefer matcher.score_batch()
    directly — it batches every resume's encode() calls into one and can
    persist its cache across runs.
    """
    matcher = matcher or LocalSemanticMatcher()
    return matcher.compute_similarity(jd_text, resume_text)


# ---------------------------------------------------------------------------
# Self-tests that don't require the actual model (offline sandbox has no
# network access to download all-MiniLM-L6-v2) — validate normalization
# math, section-filtering, batched-flattening, and disk-cache persistence
# using a mocked matcher / fake encoder.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # 1. Pool normalization should stretch a tight cluster of raw scores to [0, 1]
    raw = {"alice": 0.42, "bob": 0.51, "carol": 0.38, "dave": 0.55}
    normalized = LocalSemanticMatcher._normalize_pool(raw)
    assert min(normalized.values()) == 0.0
    assert max(normalized.values()) == 1.0
    assert normalized["carol"] < normalized["alice"] < normalized["bob"] < normalized["dave"]

    # 2. Degenerate case: identical scores shouldn't fabricate fake spread
    flat_scores = {"a": 0.4, "b": 0.4}
    flat_norm = LocalSemanticMatcher._normalize_pool(flat_scores)
    assert all(v == 0.5 for v in flat_norm.values())

    # 3. "_unclassified_headers" (a List[str], not a string) must not crash
    #    scoring and must be excluded from scorable content.
    fake = LocalSemanticMatcher.__new__(LocalSemanticMatcher)
    fake.section_weights = {"skills": 0.25, "general": 0.20}
    fake._non_text_keys = {"_unclassified_headers"}
    sections = {
        "skills": "Python, React, Node.js",
        "_unclassified_headers": ["Volunteering", "Certifications"],
        "general": "",
    }
    text_sections = {
        k: v for k, v in sections.items()
        if k not in fake._non_text_keys and isinstance(v, str) and v.strip()
    }
    assert text_sections == {"skills": "Python, React, Node.js"}

    # 4. Cosine similarity sanity check (numpy-only implementation)
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    assert abs(LocalSemanticMatcher._cosine_sim(a, b) - 1.0) < 1e-9
    assert abs(LocalSemanticMatcher._cosine_sim(a, c) - 0.0) < 1e-9
    assert LocalSemanticMatcher._cosine_sim(np.zeros(2), np.zeros(2)) == 0.0  # no div-by-zero crash

    # --- Stub model: deterministic embeddings, records call count ---
    class _StubModel:
        def __init__(self):
            self.encode_call_count = 0

        def encode(self, texts, convert_to_numpy=True, batch_size=32):
            self.encode_call_count += 1
            if isinstance(texts, str):
                return np.array([len(texts) % 7 + 1, len(texts) % 5 + 1], dtype=float)
            return [np.array([len(t) % 7 + 1, len(t) % 5 + 1], dtype=float) for t in texts]

    def _fresh_matcher(cache_path=None):
        m = LocalSemanticMatcher.__new__(LocalSemanticMatcher)
        m.model = _StubModel()
        m.batch_size = 32
        m._cache_path = cache_path
        m._cached_jd_text = None
        m._cached_jd_emb = None
        m._embedding_cache = {}
        if cache_path:
            m._load_cache(cache_path)
        m.section_weights = {
            "experience": 0.35, "projects": 0.30, "skills": 0.25,
            "education": 0.10, "general": 0.20,
        }
        m._non_text_keys = {"_unclassified_headers"}
        return m

    candidates = {
        "Aarav": {"experience": "Built cohort models for a SaaS product.",
                  "skills": "SQL, Python, Tableau, BigQuery, dbt, Amplitude"},
        "Priya": {"experience": "Owned product KPI dashboards and experiments.",
                  "skills": "SQL, Python, Power BI, GA4, Mixpanel"},
        "Rohan": {"experience": "Funnel analysis and dashboarding at a SaaS company.",
                  "skills": "SQL, Python, Tableau, Snowflake"},
    }
    jd_text = "Looking for a product analyst with SQL, Python, and experimentation experience."

    # 5. Exactly one encode() call for the JD + one batched call for the
    #    whole candidate pool (not one call per candidate).
    matcher = _fresh_matcher()
    result = matcher.score_batch(jd_text, candidates)
    assert matcher.model.encode_call_count == 2, (
        f"Expected 2 encode() calls total (1 JD + 1 batched pool), got {matcher.model.encode_call_count}"
    )
    assert set(result.keys()) == {"Aarav", "Priya", "Rohan"}
    assert all(0.0 <= v <= 1.0 for v in result.values())

    # 6. Re-running with identical text hits the in-memory cache: zero new calls.
    calls_before = matcher.model.encode_call_count
    _ = matcher.score_batch(jd_text, candidates)
    assert matcher.model.encode_call_count == calls_before, (
        "Re-scoring identical resume/JD text should be served entirely from cache"
    )

    # 7. Disk persistence: save to a temp file, build a BRAND NEW matcher
    #    instance (simulating a fresh process / app restart) that loads
    #    from that file, and confirm it serves the SAME candidates with
    #    ZERO new encode() calls because the cache was already on disk.
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "embeddings_cache.pkl")

        matcher_a = _fresh_matcher(cache_path=cache_file)
        matcher_a.score_batch(jd_text, candidates)  # writes cache to disk
        assert os.path.exists(cache_file), "score_batch() should have saved the cache to disk"

        matcher_b = _fresh_matcher(cache_path=cache_file)  # fresh instance, loads from disk
        assert len(matcher_b._embedding_cache) > 0, "New instance should have loaded the persisted cache"
        _ = matcher_b.score_batch(jd_text, candidates)
        assert matcher_b.model.encode_call_count == 0, (
            "A fresh process reusing a disk cache for unchanged text should need zero new encode() calls"
        )

        # 8. Changing ONE candidate's text after loading from disk should
        #    only trigger a new (batched) call for that changed text.
        candidates_modified = dict(candidates)
        candidates_modified["Rohan"] = {
            "experience": "Funnel analysis and dashboarding at a SaaS company.",
            "skills": "SQL, Python, Tableau, Snowflake, dbt",  # changed
        }
        _ = matcher_b.score_batch(jd_text, candidates_modified)
        assert matcher_b.model.encode_call_count == 1, (
            "Only the changed text should trigger a new (batched) encode() call, "
            f"got {matcher_b.model.encode_call_count}"
        )

    print("All offline self-tests passed.")