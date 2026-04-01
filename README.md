# HireVion — AI Resume Intelligence Platform
> Final Year Engineering Project | Flask + spaCy + Sentence-BERT + XAI + Bias Detection

**Theme:** Dark/Light Mode · Glassmorphism · Cyberpunk Cyan  
**Stack:** Python 3.11 · Flask · spaCy · Sentence-BERT · Plotly.js · langdetect  
**Default Login:** `admin` / `1234`

---

##  Features in hirevion

| Feature | Details |
|---|---|
|  Dark / Light Mode | Toggle in nav — persists via localStorage |
|  Anonymization Mode | Strip PII before scoring for blind hiring |
|  Candidate Comparison | Select 2–3 candidates → radar chart + skill table |
|  XAI Waterfall Chart | Per-candidate score breakdown bar chart |
|  Bias Report Page | Pool-level bias analysis with charts |
|  Research Page | Ablation study, threshold sweep, pipeline stats |
|  Multi-language | langdetect flags non-English resumes with notice |
|  Score Capped at 100 | Hard max enforced across all scoring paths |

---

##  Project Structure

```
hirevion/
├── app.py                  ← Flask backend
├── resume_parser.py        ← NLP engine (v2: anonymize, langdetect, score cap)
├── research_additions.py   ← Ablation, ground-truth eval, pipeline stats
├── hirevion_users.json     ← SHA-256 hashed user credentials
├── requirements.txt
├── Procfile                ← Gunicorn for Render/Railway
├── build.sh
└── templates/
    ├── welcome.html
    ├── login.html
    ├── upload.html         ← Anonymize toggle, dark/light, compact layout
    ├── dashboard.html      ← Comparison modal, XAI chart, lang badges
    ├── bias_report.html    ← NEW: pool-level bias analysis page
    └── research.html       ← NEW: ablation study, threshold sweep, stats
```



##  AI Pipeline

| Stage | Technology |
|---|---|
| Text Extraction | PyPDF2, python-docx |
| Language Detection | langdetect |
| NER | spaCy en_core_web_sm |
| Anonymization | Regex + spaCy PERSON entities |
| Skill Matching | Regex + 300+ skill taxonomy |
| Semantic Match | Sentence-BERT (all-MiniLM-L6-v2) / TF-IDF fallback |
| Scoring | Composite weighted formula — capped at 100 |
| Explainability | Waterfall chart (XAI breakdown) |
| Bias Detection | Rule-based pool + per-candidate flagging |
| Comparison | Multi-candidate radar chart |
| Research | Ablation study, threshold sweep, pipeline stats |

---

##  Application Flow

```
/ (Welcome) → /login → /upload
  ↓ [POST /process]
/dashboard   — KPIs · table · charts · candidate modal (6 tabs incl. XAI)
              Compare modal (radar + skills table)
/bias-report — Pool bias analysis
/research    — Ablation study · threshold sweep · pipeline stats
/export      — CSV download
```

---

*HireVion-built by Harsh Singh*
