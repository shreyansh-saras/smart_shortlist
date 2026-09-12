# Smart Shortlisting Engine: Explainable Resume Ranking

An end-to-end, fully local hybrid NLP platform designed to parse, score, and rank batches of resumes against a target Job Description with complete transparency. Built for placement platforms like InternLoom, it bridges explicit keyword verification with contextual semantic understanding while defeating adversarial ATS exploits like keyword stuffing[cite: 1].

---

## 📌 Features

### 1. Hybrid Scoring Architecture (35% Core Criterion)[cite: 1]
* **Keyword Matching (Lexical Branch):** Employs tiered skill extraction and BM25/TF-IDF vector matching against exact tools, frameworks, and languages[cite: 1].
* **Semantic Retrieval (Dense Vector Branch):** Understands conceptual equivalents (e.g., mapping REST APIs with Express & MongoDB to Node.js backend roles) using local sentence transformers (`all-MiniLM-L6-v2`) with automatic fallback to dense TF-IDF[cite: 1].
* **Section-Aware Weighting:** Attributes greater weight to skills verified within `Projects` and `Experience` sections over raw keyword lists, effectively downranking keyword stuffers.

### 2. Explainability & Fit Auditing (20% Core Criterion)[cite: 1]
* **Top 3 Candidate Deep-Dive:** Generates concise, grounded explanations of matched qualifications and identified missing requirements[cite: 1].
* **Gated Scoring Formula:** Enforces prerequisite checks where candidates missing mandatory requirements encounter penalty gates, ensuring realistic score spread across the pool[cite: 1].

### 3. Bonus Capabilities[cite: 1]
* **JD Phrasing & Bias Auditor:** Scans job descriptions for exclusionary language, masculine-coded jargon, or unrealistic experience expectations for junior/intern roles[cite: 1].
* **Recruiter Delta Comparator:** Inspects the exact skill and semantic delta between any two candidates to answer: *"Why is Candidate X ranked above Candidate Y?"*[cite: 1]
* **Format-Resilient Normalization:** Ingests both plain-text formats and complex multi-column PDFs without layout corruption[cite: 1].

---

### Mathematical Formula

The final candidate fit score ($S_{\text{final}} \in [0, 100]$) is computed deterministically:

$$S_{\text{final}} = \text{Gate} \times \Big( 0.25 S_{\text{keyword}} + 0.25 S_{\text{semantic}} + 0.30 S_{\text{mandatory}} + 0.20 S_{\text{experience}} - 0.50 P_{\text{missing}} \Big)$$

* **$\text{Gate}$ Value:** Set to $1.00$ if all mandatory criteria are satisfied, $0.85$ if verified only semantically, and $0.55$ if mandatory skills are missing.
* **$P_{\text{missing}}$:** Deterministic penalty applied for missing core prerequisites.

---

## 📁 Repository Structure

```text
.
├── backend/
│   ├── app.py                  # FastAPI application & API endpoints
│   ├── config.py               # Configurable scoring weights and thresholds
│   ├── parser/                 # Document parsing and section normalization
│   ├── nlp/                    # JD requirement extraction & resume NLP
│   ├── matching/               # BM25 keyword and dense semantic matchers
│   ├── ranking/                # Deterministic scoring & gated rank aggregation
│   ├── explanations/           # Top-3 grounded explanation generators
│   └── bonus/                  # Bias auditing and candidate comparator logic
├── data/
│   ├── jd/                     # Target job descriptions (e.g., jd_fullstack.txt)
│   └── resumes/                # Candidate resumes (01 to 18 TXT/PDF files)
├── frontend/                   # UI dashboard (HTML, CSS, Vanilla JS)
├── tests/
│   └── run_pipeline.py         # Pipeline verification & CLI test script
├── requirements.txt            # System dependencies
└── README.md
