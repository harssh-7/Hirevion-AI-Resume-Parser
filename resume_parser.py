"""
resume_parser.py  —  HireVion NLP Engine
Enhancements v2:
  • Score capped at 100
  • Resume anonymization mode
  • Multi-language detection (langdetect)
  • Bias pool-level stats helper
"""

import re
import time
import os
import math
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ── spaCy ──────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# ── Sentence-BERT ───────────────────────────────────────────────────
_FORCE_TFIDF  = os.environ.get("FORCE_TFIDF", "false").lower() == "true"
_SBERT_MODEL  = None
SBERT_AVAILABLE = False

if not _FORCE_TFIDF:
    try:
        from sentence_transformers import SentenceTransformer, util as sbert_util
        _CACHE_DIR = os.environ.get(
            "SENTENCE_TRANSFORMERS_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "sbert")
        )
        os.makedirs(_CACHE_DIR, exist_ok=True)
        _SBERT_MODEL    = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=_CACHE_DIR)
        SBERT_AVAILABLE = True
    except Exception as _e:
        print(f"[HireVion] SBERT unavailable ({_e}), falling back to TF-IDF.")

# ── langdetect ─────────────────────────────────────────────────────
LANGDETECT_AVAILABLE = False
try:
    from langdetect import detect as _langdetect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    pass

# ── Role keywords ───────────────────────────────────────────────────
_ROLE_KEYWORDS = [
    "engineer", "developer", "intern", "manager", "analyst",
    "consultant", "architect", "scientist", "researcher", "lead",
    "director", "officer", "specialist", "coordinator",
]

# ─────────────────────────────────────────────
#  LANGUAGE DETECTION
# ─────────────────────────────────────────────

SUPPORTED_LANGS = {"en": "English", "hi": "Hindi", "fr": "French",
                   "de": "German", "es": "Spanish", "pt": "Portuguese"}

def detect_language(text: str) -> dict:
    """Detect resume language. Returns {'code': 'en', 'name': 'English', 'supported': True}"""
    if not LANGDETECT_AVAILABLE:
        return {"code": "en", "name": "English", "supported": True, "notice": None}
    try:
        code = _langdetect(text[:2000])
        name = SUPPORTED_LANGS.get(code, f"Unknown ({code})")
        supported = code == "en"
        notice = None
        if not supported:
            notice = (f"Resume detected as {name}. NLP extraction is optimised for English. "
                      f"Results may be less accurate — consider providing an English translation.")
        return {"code": code, "name": name, "supported": supported, "notice": notice}
    except Exception:
        return {"code": "en", "name": "English", "supported": True, "notice": None}


# ─────────────────────────────────────────────
#  ANONYMIZATION
# ─────────────────────────────────────────────

def anonymize_text(text: str) -> str:
    """
    Strip PII from resume text for blind hiring:
      - Names (spaCy PERSON entities in first 500 chars)
      - Email addresses
      - Phone numbers
      - URLs / LinkedIn / GitHub profile links
      - Photo references
    Returns cleaned text with placeholders.
    """
    # Email
    text = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)
    # Phone
    text = re.sub(r"\+?[\d][\d\s\-\(\)]{8,15}", "[PHONE]", text)
    # URLs
    text = re.sub(r"https?://[^\s]+", "[URL]", text)
    text = re.sub(r"www\.[^\s]+", "[URL]", text)
    # LinkedIn / GitHub specific slugs
    text = re.sub(r"linkedin\.com/in/[A-Za-z0-9_\-]+", "[LINKEDIN]", text)
    text = re.sub(r"github\.com/[A-Za-z0-9_\-]+", "[GITHUB]", text)
    # Photo keywords
    text = re.sub(r"(?i)(photo|photograph|picture)\s*(attached|included|enclosed)?", "[PHOTO_REF]", text)
    # Name — use spaCy on first 500 chars
    try:
        doc = nlp(text[:500])
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
                text = text.replace(ent.text, "[NAME]")
    except Exception:
        pass
    # Address / location line patterns (city, pin)
    text = re.sub(r"\b\d{6}\b", "[PINCODE]", text)
    return text


# ─────────────────────────────────────────────
#  ENTITY EXTRACTION
# ─────────────────────────────────────────────

def _entities_from_doc(doc):
    persons, organizations, dates, locations = [], [], [], []
    for ent in doc.ents:
        if ent.label_ == "PERSON":    persons.append(ent.text)
        elif ent.label_ == "ORG":     organizations.append(ent.text)
        elif ent.label_ == "DATE":    dates.append(ent.text)
        elif ent.label_ == "GPE":     locations.append(ent.text)
    return {
        "persons":       list(set(persons)),
        "organizations": list(set(organizations)),
        "dates":         list(set(dates)),
        "locations":     list(set(locations)),
    }

def _experience_from_doc(doc):
    return [
        s.text.strip() for s in doc.sents
        if any(w in s.text.lower() for w in _ROLE_KEYWORDS)
    ][:7]


# ─────────────────────────────────────────────
#  SECTION EXTRACTOR
# ─────────────────────────────────────────────

def extract_section(text, section_names):
    if isinstance(section_names, str):
        section_names = [section_names]
    text_lower  = text.lower()
    start_index = -1
    for name in section_names:
        idx = text_lower.find(name)
        if idx != -1:
            start_index = idx
            break
    if start_index == -1:
        return ""
    headings = [
        "skills", "technical skills", "experience", "work experience",
        "projects", "education", "certifications", "certificates",
        "achievements", "awards", "publications", "interests",
    ]
    end_index = len(text)
    for heading in headings:
        idx = text_lower.find(heading, start_index + 10)
        if idx != -1 and idx < end_index:
            end_index = idx
    return text[start_index:end_index]


# ─────────────────────────────────────────────
#  NAME EXTRACTION
# ─────────────────────────────────────────────

_NAME_BLACKLIST = {
    "resume","curriculum","vitae","cv","summary","experience","education",
    "skills","projects","certifications","certificate","profile","contact",
    "objective","page","work","about","me","details","information","personal",
    "declaration","references","career","professional","technical","academic",
    "interests","activities","achievements","awards","publications","languages",
    "hobbies","volunteer","internship","training","course","skill","project",
    "qualification","strength","background","overview","introduction",
    "developer","engineer","analyst","manager","director","officer",
    "architect","consultant","specialist","coordinator","administrator",
    "executive","associate","intern","lead","head","chief","senior","junior",
    "principal","staff","distinguished","fellow","president","vice",
    "recruiter","scientist","researcher","professor","instructor","teacher",
    "designer","developer","technician","supervisor","assistant","representative",
    "ml","ai","ui","ux","it","api","sql","nlp","erp","crm",
    "sap","php","net","bi","qa","sde","swe","ios","git","css",
    "gcp","aws","rpa","etl","iot","ar","vr","nlp","ocr",
    "python","java","javascript","typescript","kotlin","swift","scala",
    "docker","kubernetes","linux","android","html","css","react",
    "mobile","dev","ops","sys","tech","app","biz","org","gov",
    "data","cloud","cyber","global","open","smart","full","stack",
    "limited","pvt","ltd","inc","corp","company","systems","group",
    "solutions","services","institute","agency","hospital","clinic",
}

_TITLES  = re.compile(r"^(mr\.?|mrs\.?|ms\.?|dr\.?|prof\.?|sir|er\.?|eng\.?|md\.?|shri|smt)\s+", re.IGNORECASE)
_NAME_LABEL   = re.compile(r"^(?:name|full\s*name|candidate\s*name)\s*[:\-]\s*(.+)$", re.IGNORECASE)
_NAME_TOKEN   = re.compile(r"^[A-Z][a-zA-Z\-\']{0,29}$")
_INITIAL_TOK  = re.compile(r"^[A-Z]\.$")

def _is_name_token(word):
    return bool(_NAME_TOKEN.match(word) or _INITIAL_TOK.match(word))

def _title_case_name(raw):
    return " ".join(w.capitalize() for w in raw.split())

def _is_valid_name(candidate):
    candidate = _TITLES.sub("", candidate).strip()
    if not candidate:
        return False
    words = candidate.split()
    if not (1 <= len(words) <= 5):
        return False
    if any(ch.isdigit() for ch in candidate):
        return False
    check_words = [w.capitalize() for w in words] if candidate == candidate.upper() else words
    if not all(_is_name_token(w) for w in check_words):
        return False
    non_initials = [w for w in check_words if not _INITIAL_TOK.match(w)]
    if not non_initials:
        return False
    if any(w.lower() in _NAME_BLACKLIST for w in check_words):
        return False
    return True

def _clean_line(line):
    c = re.sub(r"[\|•·\-–—/\\]", " ", line)
    c = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "", c)
    c = re.sub(r"https?://\S+", "", c)
    c = re.sub(r"www\.\S+",     "", c)
    c = re.sub(r"\+?\d[\d\s\-\(\)]{7,}", "", c)
    c = re.sub(r"\s{2,}", " ", c)
    return c.strip()

def _normalise(name):
    name = _TITLES.sub("", name).strip()
    if name == name.upper():
        name = _title_case_name(name)
    words = [w for w in name.split() if not _INITIAL_TOK.match(w)]
    return " ".join(words).strip()

def _name_from_text_fallback(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:15]:
        m = _NAME_LABEL.match(line)
        if m:
            candidate = " ".join(_clean_line(m.group(1).strip()).split()[:4])
            if _is_valid_name(candidate):
                return _normalise(candidate)
    if lines and _is_valid_name(lines[0]):
        return _normalise(lines[0])
    for line in lines[:12]:
        clean = _clean_line(line)
        if clean and _is_valid_name(clean):
            return _normalise(clean)
    header  = text[:400]
    pattern = re.compile(r"(?<![:\-/])\b([A-Z][a-z]{1,20}(?:\s[A-Z][a-z]{1,20}){1,3})\b")
    for match in pattern.finditer(header):
        candidate = match.group(1).strip()
        if _is_valid_name(candidate):
            return _normalise(candidate)
    return "Name Not Found"

def _name_from_doc(doc):
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.start_char < 500:
            candidate = _TITLES.sub("", ent.text).strip()
            if _is_valid_name(candidate):
                return _normalise(candidate)
    return _name_from_text_fallback(doc.text)

def extract_name(text):
    doc = nlp(text[:500])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            candidate = _TITLES.sub("", ent.text).strip()
            if _is_valid_name(candidate):
                return _normalise(candidate)
    return _name_from_text_fallback(text)


# ─────────────────────────────────────────────
#  CONTACT
# ─────────────────────────────────────────────

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group() if match else "Not Found"

def extract_phone(text):
    match = re.search(r"\+?\d[\d\s\-]{8,15}", text)
    return match.group().strip() if match else "Not Found"


# ─────────────────────────────────────────────
#  SKILLS
# ─────────────────────────────────────────────

SKILLS_DB = [
    "python","java","c","c++","c#","r","scala","go","rust",
    "kotlin","swift","typescript","javascript","perl","matlab",
    "ruby","php","dart","lua","haskell","elixir","clojure",
    "html","css","react","angular","vue","node.js","express",
    "nextjs","gatsby","tailwind","bootstrap","graphql","svelte",
    "webpack","vite","sass","less","jquery","three.js",
    "spring boot","django","flask","fastapi","laravel","rails",
    "asp.net","gin","fiber","nestjs","express.js",
    "machine learning","deep learning","data science","nlp",
    "natural language processing","computer vision","reinforcement learning",
    "generative ai","large language models","prompt engineering",
    "feature engineering","model deployment","mlops",
    "pandas","numpy","scipy","matplotlib","seaborn","plotly",
    "tensorflow","pytorch","keras","scikit-learn","xgboost","lightgbm",
    "hugging face","transformers","bert","gpt","opencv","spacy",
    "nltk","statsmodels","catboost","dask",
    "sql","mysql","postgresql","mongodb","redis","cassandra",
    "elasticsearch","firebase","oracle","sqlite","dynamodb",
    "neo4j","couchdb","mariadb","supabase",
    "aws","azure","gcp","docker","kubernetes","terraform",
    "ansible","jenkins","github actions","ci/cd",
    "cloudformation","pulumi","helm","istio","argocd",
    "git","linux","bash","jira","tableau","power bi",
    "excel","hadoop","spark","kafka","airflow","dbt",
    "snowflake","databricks","redshift","bigquery","etl",
    "data pipeline","data warehouse","data lake",
    "penetration testing","siem","vulnerability assessment",
    "network security","cryptography","ethical hacking",
    "flutter","react native","android","ios","swiftui",
    "selenium","cypress","jest","pytest","junit",
    "postman","unit testing","integration testing",
    "blockchain","solidity","web3","microservices","rest api",
    "grpc","websockets","rabbitmq","celery","nginx",
    "agile","scrum","kanban",
]

_SKILL_PATTERNS = [(skill, re.compile(r'\b' + re.escape(skill) + r'\b')) for skill in SKILLS_DB]

def extract_skills(text):
    text_lower = text.lower()
    return [skill for skill, pat in _SKILL_PATTERNS if pat.search(text_lower)]

def extract_skills_weighted(text):
    all_skills = extract_skills(text)
    skills_section = extract_section(text, ["skills", "technical skills", "core competencies"])
    section_lower = skills_section.lower() if skills_section else ""
    weights = {}
    for sk in all_skills:
        weights[sk] = 1.5 if sk in section_lower else 1.0
    return all_skills, weights


# ─────────────────────────────────────────────
#  EDUCATION / EXPERIENCE / PROJECTS / CERTS
# ─────────────────────────────────────────────

def extract_education(text):
    section = extract_section(text, ["education", "academic background", "qualification"])
    if not section:
        return []
    keywords = ["university","college","bachelor","master","b.tech","m.tech",
                "degree","diploma","phd","b.e","m.e","b.sc","m.sc","institute","school of"]
    education = []
    for line in section.split("\n"):
        clean = line.strip()
        if len(clean) < 5:
            continue
        if any(word in clean.lower() for word in keywords):
            education.append(clean)
    return education[:5]

def extract_experience(text):
    doc = nlp(text)
    return _experience_from_doc(doc)

def extract_projects(text):
    section = extract_section(text, ["projects", "project experience", "academic projects"])
    if not section:
        return []
    return [line.strip() for line in section.split("\n") if len(line.strip()) >= 15][:5]

def extract_certificates(text):
    cert_keywords = ["certified","certification","certificate","course","credential"]
    certificates  = []
    for line in text.split("\n"):
        clean = line.strip()
        if len(clean) < 5 or len(clean) > 120:
            continue
        if any(kw in clean.lower() for kw in cert_keywords):
            if len(clean.split()) <= 15:
                entry      = {"certificate": clean, "issuer": "Unknown", "year": "Unknown"}
                year_match = re.search(r"(20\d{2})", clean)
                if year_match:
                    entry["year"] = year_match.group()
                certificates.append(entry)
    return certificates[:5]

def extract_links(text):
    links     = {"GitHub": "Not Found", "LinkedIn": "Not Found", "Portfolio": "Not Found"}
    github    = re.search(r"github\.com/[A-Za-z0-9_-]+", text)
    linkedin  = re.search(r"linkedin\.com/in/[A-Za-z0-9_-]+", text)
    portfolio = re.search(r"https?://[A-Za-z0-9./_-]+", text)
    if github:    links["GitHub"]    = "https://" + github.group()
    if linkedin:  links["LinkedIn"]  = "https://" + linkedin.group()
    if portfolio: links["Portfolio"] = portfolio.group()
    return links


# ─────────────────────────────────────────────
#  SIMILARITY
# ─────────────────────────────────────────────

def resume_job_similarity(resume_text, job_description):
    if not job_description.strip():
        return 0.0, "none"
    if SBERT_AVAILABLE and _SBERT_MODEL is not None:
        try:
            from sentence_transformers import util as sbert_util
            emb   = _SBERT_MODEL.encode([resume_text, job_description],
                                        convert_to_tensor=True, device="cpu")
            score = float(sbert_util.cos_sim(emb[0], emb[1])[0][0])
            return round(max(score, 0.0) * 100, 2), "sbert"
        except Exception:
            pass
    try:
        tfidf  = TfidfVectorizer(stop_words="english")
        matrix = tfidf.fit_transform([resume_text, job_description])
        score  = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        return round(score * 100, 2), "tfidf"
    except Exception:
        return 0.0, "none"

def _batch_similarity(texts, job_description):
    n = len(texts)
    if not job_description.strip():
        return [0.0] * n, "none"
    if SBERT_AVAILABLE and _SBERT_MODEL is not None:
        try:
            from sentence_transformers import util as sbert_util
            all_texts  = texts + [job_description]
            embeddings = _SBERT_MODEL.encode(
                all_texts, batch_size=32, convert_to_tensor=True,
                device="cpu", show_progress_bar=False,
            )
            jd_emb = embeddings[-1]
            scores = [
                round(max(float(sbert_util.cos_sim(embeddings[i], jd_emb)[0][0]), 0.0) * 100, 2)
                for i in range(n)
            ]
            return scores, "sbert"
        except Exception:
            pass
    try:
        tfidf      = TfidfVectorizer(stop_words="english")
        all_texts  = texts + [job_description]
        matrix     = tfidf.fit_transform(all_texts)
        jd_vec     = matrix[-1]
        scores = [
            round(float(cosine_similarity(matrix[i:i+1], jd_vec)[0][0]) * 100, 2)
            for i in range(n)
        ]
        return scores, "tfidf"
    except Exception:
        return [0.0] * n, "none"


# ─────────────────────────────────────────────
#  DOMAIN / SKILL GAP / EXPERIENCE
# ─────────────────────────────────────────────

DOMAIN_MAP = {
    "Frontend Developer": ["html","css","javascript","react","angular","vue","typescript"],
    "Backend Developer":  ["python","java","django","flask","spring boot","sql","fastapi","node.js"],
    "Data Science / ML":  ["python","machine learning","deep learning","data science",
                           "pandas","numpy","tensorflow","pytorch","scikit-learn"],
    "DevOps / Cloud":     ["aws","azure","gcp","docker","kubernetes","terraform","ci/cd"],
    "Full Stack":         ["react","node.js","sql","django","javascript","python"],
    "Mobile Developer":   ["kotlin","swift","react","flutter","android","ios"],
    "Data Engineer":      ["spark","kafka","hadoop","sql","python","scala","aws"],
}

def detect_domain(skills):
    scores = {domain: 0 for domain in DOMAIN_MAP}
    for skill in skills:
        for domain, domain_skills in DOMAIN_MAP.items():
            if skill.lower() in domain_skills:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"

def skill_gap(candidate_skills, required_skills):
    candidate_lower = [s.lower() for s in candidate_skills]
    required_lower  = [s.lower() for s in required_skills]
    matched = [s for s in required_lower if s in candidate_lower]
    missing = [s for s in required_lower if s not in candidate_lower]
    pct     = round((len(matched) / len(required_lower)) * 100, 2) if required_lower else 0.0
    return matched, missing, pct

def estimate_experience_years(experience_list):
    years = []
    for exp in experience_list:
        found = re.findall(r"(20\d{2}|19\d{2})", exp)
        years.extend(int(y) for y in found)
    if len(years) >= 2:
        return max(years) - min(years)
    return len(experience_list)


# ─────────────────────────────────────────────
#  SCORING  — capped at 100
# ─────────────────────────────────────────────

# Scoring weights (tunable)
_W_SKILL       = 3    # pts per general skill
_W_REQ_SKILL   = 10   # pts per required skill matched
_W_PROJECT     = 4    # pts per project
_W_CERT        = 2    # pts per certificate
_W_EXPERIENCE  = 5    # pts per experience entry
_W_AI_BONUS    = 0.2  # multiplier on similarity %
_SCORE_MAX     = 100  # hard cap


def _raw_score(skills, projects, certificates, experience, required_skills, sim):
    skill_match_score = sum(_W_REQ_SKILL for s in required_skills
                            if s in [sk.lower() for sk in skills])
    base = (
        len(skills)       * _W_SKILL +
        skill_match_score +
        len(projects)     * _W_PROJECT +
        len(certificates) * _W_CERT +
        len(experience)   * _W_EXPERIENCE
    )
    return round(min(base + sim * _W_AI_BONUS, _SCORE_MAX), 2)


def compute_score(skills, projects, certificates, experience,
                  required_skills, job_description="", resume_text=""):
    t0 = time.time()
    similarity, sim_method = resume_job_similarity(resume_text, job_description) \
        if job_description else (0, "none")
    score = _raw_score(skills, projects, certificates, experience, required_skills, similarity)
    return score, round(similarity, 2), sim_method, round(time.time() - t0, 3)


def candidate_category(score):
    if score >= 80:   return "Highly Suitable"
    elif score >= 60: return "Suitable"
    elif score >= 40: return "Moderate"
    else:             return "Not Suitable"


# ─────────────────────────────────────────────
#  TEXT STATS / READABILITY
# ─────────────────────────────────────────────

def _count_syllables(word):
    word = word.lower().strip(".,!?;:\"'()[]{}") 
    if len(word) <= 2:
        return 1
    if word.endswith("e") and not word.endswith("le"):
        word = word[:-1]
    vowels = "aeiouy"
    count, prev_vowel = 0, False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(count, 1)

def flesch_reading_ease(text):
    words     = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_words   = len(words)
    n_sent    = max(len(sentences), 1)
    n_syl     = sum(_count_syllables(w) for w in words)
    if n_words == 0:
        return 0.0
    score = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (n_syl / n_words)
    return round(max(0, min(score, 100)), 2)

def flesch_kincaid_grade(text):
    words     = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_words   = len(words)
    n_sent    = max(len(sentences), 1)
    n_syl     = sum(_count_syllables(w) for w in words)
    if n_words == 0:
        return 0.0
    grade = 0.39 * (n_words / n_sent) + 11.8 * (n_syl / n_words) - 15.59
    return round(max(0, grade), 2)

def text_statistics(text):
    words        = text.split()
    sentences    = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    unique_words = len(set(w.lower() for w in words))
    return {
        "word_count":           len(words),
        "sentence_count":       len(sentences),
        "unique_words":         unique_words,
        "avg_word_length":      round(sum(len(w) for w in words) / max(len(words), 1), 2),
        "avg_sentence_length":  round(len(words) / max(len(sentences), 1), 2),
        "lexical_diversity":    round(unique_words / max(len(words), 1), 4),
        "flesch_reading_ease":  flesch_reading_ease(text),
        "flesch_kincaid_grade": flesch_kincaid_grade(text),
    }


# ─────────────────────────────────────────────
#  KEYWORD EXTRACTION
# ─────────────────────────────────────────────

def extract_keywords(text, top_n=10):
    try:
        bg_corpus = [
            "software engineer developer experience education skills projects",
            "data analysis machine learning python sql database management",
            "project manager agile scrum team leadership communication",
        ]
        corpus = bg_corpus + [text]
        tfidf  = TfidfVectorizer(stop_words="english", max_features=500,
                                 ngram_range=(1, 2), min_df=1)
        matrix = tfidf.fit_transform(corpus)
        feature_names = tfidf.get_feature_names_out()
        resume_vec = matrix[-1].toarray().flatten()
        top_indices = resume_vec.argsort()[-top_n:][::-1]
        return [
            (feature_names[i], round(float(resume_vec[i]), 4))
            for i in top_indices if resume_vec[i] > 0
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────
#  RECOMMENDATION
# ─────────────────────────────────────────────

def get_recommendation(score):
    if score >= 80:   return "✅ Hire"
    elif score >= 60: return "🟡 Consider"
    else:             return "❌ Reject"


# ─────────────────────────────────────────────
#  SCORE EXPLAINABILITY  — structured for waterfall chart
# ─────────────────────────────────────────────

def explain_score(skills, projects, certificates, experience,
                  required_skills, similarity, sim_method):
    cand_lower  = [s.lower() for s in skills]
    matched_req = [s for s in required_skills if s in cand_lower]
    breakdown   = []

    skill_pts = min(len(skills) * _W_SKILL, 30)
    breakdown.append({"label": "General Skills",
                      "points": skill_pts,
                      "detail": f"{len(skills)} skills × {_W_SKILL} pts"})

    req_pts = min(len(matched_req) * _W_REQ_SKILL, 40)
    if required_skills:
        breakdown.append({"label": "Required Skill Match",
                          "points": req_pts,
                          "detail": f"{len(matched_req)}/{len(required_skills)} matched × {_W_REQ_SKILL} pts"})

    proj_pts = min(len(projects) * _W_PROJECT, 16)
    breakdown.append({"label": "Projects",
                      "points": proj_pts,
                      "detail": f"{len(projects)} projects × {_W_PROJECT} pts"})

    cert_pts = min(len(certificates) * _W_CERT, 10)
    breakdown.append({"label": "Certificates",
                      "points": cert_pts,
                      "detail": f"{len(certificates)} certs × {_W_CERT} pts"})

    exp_pts = min(len(experience) * _W_EXPERIENCE, 20)
    breakdown.append({"label": "Experience",
                      "points": exp_pts,
                      "detail": f"{len(experience)} entries × {_W_EXPERIENCE} pts"})

    ai_pts = round(min(similarity * _W_AI_BONUS, 20), 2)
    method_label = "S-BERT" if sim_method == "sbert" else "TF-IDF"
    breakdown.append({"label": f"AI Similarity ({method_label})",
                      "points": ai_pts,
                      "detail": f"{similarity:.1f}% similarity × {_W_AI_BONUS}"})

    return breakdown


# ─────────────────────────────────────────────
#  BIAS DETECTION
# ─────────────────────────────────────────────

GENDERED_MALE    = ["he", "him", "his", "himself", "mr.", "mr "]
GENDERED_FEMALE  = ["she", "her", "hers", "herself", "ms.", "mrs.", "miss"]
NATIONALITY_INDICATORS = ["citizen", "visa", "nationality", "passport", "born in",
                           "native of", "resident of"]
AGE_INDICATORS   = ["age:", "born:", "dob:", "date of birth", "year of birth",
                    r"\b(19[5-9]\d|200[0-9])\b"]
PHOTO_INDICATORS = ["photo", "photograph", "picture attached", "image attached"]

def detect_bias_flags(text):
    text_lower = text.lower()
    flags      = {}
    male_hits   = [w for w in GENDERED_MALE   if w in text_lower]
    female_hits = [w for w in GENDERED_FEMALE if w in text_lower]
    if male_hits or female_hits:
        flags["Gender Language"] = {
            "level":      "warning",
            "detail":     f"Gendered terms found: {', '.join(male_hits + female_hits)}",
            "suggestion": "Gender-neutral language recommended for unbiased screening.",
        }
    nat_hits = [w for w in NATIONALITY_INDICATORS if w in text_lower]
    if nat_hits:
        flags["Nationality Info"] = {
            "level":      "warning",
            "detail":     f"Nationality-related terms: {', '.join(nat_hits)}",
            "suggestion": "Nationality info is irrelevant for skill-based screening.",
        }
    age_hits = []
    for pattern in AGE_INDICATORS:
        if re.search(pattern, text_lower):
            age_hits.append(pattern)
    if age_hits:
        flags["Age Disclosure"] = {
            "level":      "warning",
            "detail":     "Date of birth or age-related information detected.",
            "suggestion": "Age should not factor into candidate evaluation.",
        }
    photo_hits = [w for w in PHOTO_INDICATORS if w in text_lower]
    if photo_hits:
        flags["Photo Reference"] = {
            "level":      "info",
            "detail":     "Resume may include a photograph.",
            "suggestion": "Photos can introduce appearance bias.",
        }
    word_count = len(text.split())
    if word_count < 150:
        flags["Very Short Resume"] = {
            "level":      "info",
            "detail":     f"Only {word_count} words — may disadvantage non-native speakers.",
            "suggestion": "Consider minimum content thresholds instead of penalising brevity.",
        }
    elif word_count > 1200:
        flags["Very Long Resume"] = {
            "level":      "info",
            "detail":     f"{word_count} words — verbose resumes may obscure key info.",
            "suggestion": "Scoring is content-based; length alone should not favour candidates.",
        }
    return flags


def compute_pool_bias_report(candidates):
    """Aggregate bias flags across entire candidate pool for the Bias Report page."""
    flag_counts = Counter()
    candidates_with_flags = 0
    gender_flagged = nationality_flagged = age_flagged = photo_flagged = 0

    for c in candidates:
        flags = c.get("bias_flags", {})
        if flags:
            candidates_with_flags += 1
        for key in flags:
            flag_counts[key] += 1
        if "Gender Language"  in flags: gender_flagged      += 1
        if "Nationality Info" in flags: nationality_flagged += 1
        if "Age Disclosure"   in flags: age_flagged         += 1
        if "Photo Reference"  in flags: photo_flagged       += 1

    total = len(candidates)
    return {
        "total_candidates":       total,
        "candidates_with_flags":  candidates_with_flags,
        "flag_pct":               round(candidates_with_flags / max(total, 1) * 100, 1),
        "gender_flagged":         gender_flagged,
        "nationality_flagged":    nationality_flagged,
        "age_flagged":            age_flagged,
        "photo_flagged":          photo_flagged,
        "flag_counts":            dict(flag_counts),
        "top_flags":              flag_counts.most_common(5),
    }


# ─────────────────────────────────────────────
#  IMPROVEMENT SUGGESTIONS
# ─────────────────────────────────────────────

def generate_improvement_suggestions(skills, projects, certificates,
                                     education, experience, links,
                                     required_skills, similarity, text, domain):
    suggestions = []
    cand_lower  = [s.lower() for s in skills]
    if required_skills:
        missing = [s for s in required_skills if s not in cand_lower]
        if missing:
            suggestions.append({"priority": "high", "icon": "🚨",
                                 "title": "Add Missing Required Skills",
                                 "detail": f"Not mentioned: {', '.join(missing)}."})
    if len(projects) == 0:
        suggestions.append({"priority": "high", "icon": "🚨",
                             "title": "Add Projects Section",
                             "detail": "No projects detected. Projects significantly boost your score."})
    elif len(projects) < 2:
        suggestions.append({"priority": "medium", "icon": "⚠️",
                             "title": "Add More Projects",
                             "detail": "Only 1 project detected. Aim for 2–4 relevant projects."})
    if len(certificates) == 0:
        suggestions.append({"priority": "medium", "icon": "⚠️",
                             "title": "Add Certifications",
                             "detail": "No certifications detected. Industry certs strengthen your profile."})
    if 0 < similarity < 30:
        suggestions.append({"priority": "high", "icon": "🚨",
                             "title": "Improve JD Alignment",
                             "detail": f"AI similarity only {similarity:.1f}%. Mirror JD keywords."})
    elif 0 < similarity < 55:
        suggestions.append({"priority": "medium", "icon": "⚠️",
                             "title": "Better Keyword Alignment",
                             "detail": f"AI similarity {similarity:.1f}%. Tailor language to the JD."})
    if links.get("GitHub") == "Not Found":
        suggestions.append({"priority": "medium", "icon": "⚠️",
                             "title": "Add GitHub Profile",
                             "detail": "No GitHub link detected."})
    if links.get("LinkedIn") == "Not Found":
        suggestions.append({"priority": "low", "icon": "💡",
                             "title": "Add LinkedIn Profile",
                             "detail": "No LinkedIn detected."})
    if len(education) == 0:
        suggestions.append({"priority": "medium", "icon": "⚠️",
                             "title": "Add Education Section",
                             "detail": "No education entries detected."})
    if len(skills) < 8:
        suggestions.append({"priority": "medium", "icon": "⚠️",
                             "title": "Expand Skills Section",
                             "detail": f"Only {len(skills)} skills detected. List all relevant tools."})
    order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda x: order[x["priority"]])
    return suggestions


# ─────────────────────────────────────────────
#  EVALUATION METRICS
# ─────────────────────────────────────────────

def compute_evaluation_metrics(candidates, threshold_score=60):
    tp = fp = fn = tn = 0
    for c in candidates:
        predicted  = c["score"] >= threshold_score
        actual_pos = "Hire"   in c["recommendation"]
        actual_neg = "Reject" in c["recommendation"]
        if predicted and actual_pos:       tp += 1
        elif predicted and actual_neg:     fp += 1
        elif not predicted and actual_pos: fn += 1
        elif not predicted and actual_neg: tn += 1
    precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 1) \
                if (precision + recall) > 0 else 0.0
    total     = tp + fp + fn + tn
    accuracy  = round((tp + tn) / total * 100, 1) if total > 0 else 0.0
    return {
        "precision": precision, "recall": recall,
        "f1_score":  f1,        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold_score,
    }


# ─────────────────────────────────────────────
#  BATCH PROCESSOR
# ─────────────────────────────────────────────

def process_batch(texts, labels, required_skills=[], job_description="",
                  prefills=None, anonymize=False):
    """
    Process resumes in one fast batch pass.
    anonymize=True strips PII before NLP extraction (blind hiring mode).
    """
    if not texts:
        return []
    if prefills is None:
        prefills = [{}] * len(texts)
    while len(prefills) < len(texts):
        prefills.append({})

    # Optionally anonymize
    proc_texts = [anonymize_text(t) if anonymize else t for t in texts]

    # Batch spaCy parse
    docs = list(nlp.pipe(proc_texts, batch_size=16, disable=["lemmatizer"]))

    # Batch similarity
    sims, sim_method = _batch_similarity(proc_texts, job_description)

    results = []
    for i, (text, doc, label) in enumerate(zip(proc_texts, docs, labels)):
        t0      = time.time()
        prefill = prefills[i] if i < len(prefills) else {}
        original_text = texts[i]  # for language detection use original

        # Language detection on original text
        lang_info = detect_language(original_text[:3000])

        entities   = _entities_from_doc(doc)
        experience = _experience_from_doc(doc)

        name  = (prefill.get("name")  or "").strip() or _name_from_doc(doc)
        email = (prefill.get("email") or "").strip() or extract_email(text)
        phone = (prefill.get("phone") or "").strip() or extract_phone(text)

        # If anonymized, override with placeholders
        if anonymize:
            name  = "[ANONYMIZED]"
            email = "[ANONYMIZED]"
            phone = "[ANONYMIZED]"

        skills    = extract_skills(text)
        education = extract_education(text)
        projects  = extract_projects(text)
        certs     = extract_certificates(text)
        links     = extract_links(text)
        stats     = text_statistics(text)
        keywords  = extract_keywords(text)
        domain    = detect_domain(skills)

        matched, missing, match_pct = skill_gap(skills, required_skills)
        exp_years = estimate_experience_years(experience)

        sim = sims[i]
        score = _raw_score(skills, projects, certs, experience, required_skills, sim)

        category  = candidate_category(score)
        rec       = get_recommendation(score)
        breakdown = explain_score(skills, projects, certs, experience,
                                  required_skills, sim, sim_method)
        bias_flags  = detect_bias_flags(text)
        suggestions = generate_improvement_suggestions(
            skills, projects, certs, education, experience,
            links, required_skills, sim, text, domain
        )

        results.append({
            "name":             name,
            "email":            email,
            "phone":            phone,
            "skills":           skills,
            "education":        education,
            "experience":       experience,
            "projects":         projects,
            "certificates":     certs,
            "links":            links,
            "entities":         entities,
            "stats":            stats,
            "keywords":         keywords,
            "domain":           domain,
            "matched_skills":   matched,
            "missing_skills":   missing,
            "skill_match":      match_pct,
            "experience_count": len(experience),
            "experience_years": exp_years,
            "score":            score,
            "ai_similarity":    sim,
            "sim_method":       sim_method,
            "category":         category,
            "recommendation":   rec,
            "score_breakdown":  breakdown,
            "bias_flags":       bias_flags,
            "suggestions":      suggestions,
            "processing_time":  round(time.time() - t0, 3),
            "filename":         label,
            "language":         lang_info,
            "anonymized":       anonymize,
        })

    return results
