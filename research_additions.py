"""
research_additions.py  —  HireVion paper additions
==================================================
Three functions to add to resume_parser.py (or import alongside it).
No changes to your existing code needed.

Usage in app.py:
    from research_additions import (
        evaluate_against_ground_truth,
        run_ablation_study,
        compute_pipeline_stats,
    )
"""

import time
import statistics
from resume_parser import (
    process_batch,
    extract_skills, extract_education, extract_experience,
    extract_projects, extract_certificates, extract_links,
    text_statistics, detect_domain, skill_gap,
    estimate_experience_years, candidate_category,
    get_recommendation, explain_score, detect_bias_flags,
    generate_improvement_suggestions,
    resume_job_similarity, SBERT_AVAILABLE,
)


# ─────────────────────────────────────────────────────────────────
#  ADDITION 1 — Ground-truth evaluation
#  Fixes the circular evaluation in compute_evaluation_metrics().
#  Compares your system's predictions against HUMAN recruiter labels.
#
#  How to use:
#    1. Collect 50–100 resumes and ask a recruiter to label each
#       as "shortlist" (True) or "reject" (False).
#    2. Run process_batch() on those same resumes to get candidates.
#    3. Call evaluate_against_ground_truth(candidates, human_labels).
#
#  Example:
#    human_labels = [True, False, True, True, False, ...]  # recruiter decisions
#    results = evaluate_against_ground_truth(candidates, human_labels, threshold=60)
# ─────────────────────────────────────────────────────────────────

def evaluate_against_ground_truth(candidates, human_labels, threshold=60):
    """
    Evaluate system predictions against human recruiter labels.

    Args:
        candidates   : list of candidate dicts from process_batch()
        human_labels : list of bools — True = recruiter shortlisted, False = rejected
                       Must be same length and order as candidates.
        threshold    : score threshold above which system predicts "shortlist"

    Returns:
        dict with precision, recall, F1, accuracy, confusion matrix, and
        per-threshold analysis (useful for finding the best threshold).
    """
    if len(candidates) != len(human_labels):
        raise ValueError(
            f"candidates ({len(candidates)}) and human_labels "
            f"({len(human_labels)}) must be the same length."
        )

    tp = fp = fn = tn = 0
    per_candidate = []

    for cand, human_positive in zip(candidates, human_labels):
        system_positive = cand["score"] >= threshold
        correct = system_positive == human_positive

        if system_positive and human_positive:      tp += 1
        elif system_positive and not human_positive: fp += 1
        elif not system_positive and human_positive: fn += 1
        else:                                        tn += 1

        per_candidate.append({
            "name":           cand["name"],
            "score":          cand["score"],
            "system_predict": "shortlist" if system_positive else "reject",
            "human_label":    "shortlist" if human_positive  else "reject",
            "correct":        correct,
        })

    total     = tp + fp + fn + tn
    precision = round(tp / (tp + fp) * 100, 2) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 2) \
                if (precision + recall) > 0 else 0.0
    accuracy  = round((tp + tn) / total * 100, 2) if total > 0 else 0.0

    # ── Sweep thresholds 40–80 to find the best F1 ──────────────────
    threshold_sweep = []
    for t in range(40, 85, 5):
        _tp = _fp = _fn = _tn = 0
        for cand, human_positive in zip(candidates, human_labels):
            sp = cand["score"] >= t
            if sp and human_positive:       _tp += 1
            elif sp and not human_positive: _fp += 1
            elif not sp and human_positive: _fn += 1
            else:                           _tn += 1
        _p  = _tp / (_tp + _fp) * 100 if (_tp + _fp) > 0 else 0
        _r  = _tp / (_tp + _fn) * 100 if (_tp + _fn) > 0 else 0
        _f1 = 2 * _p * _r / (_p + _r) if (_p + _r) > 0 else 0
        threshold_sweep.append({"threshold": t, "precision": round(_p,2),
                                 "recall": round(_r,2), "f1": round(_f1,2)})

    best_t = max(threshold_sweep, key=lambda x: x["f1"])

    return {
        # ── Primary metrics (report these in your paper's Table) ──
        "precision":        precision,
        "recall":           recall,
        "f1_score":         f1,
        "accuracy":         accuracy,
        # ── Confusion matrix ──────────────────────────────────────
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_samples":    total,
        "threshold_used":   threshold,
        # ── Threshold analysis ────────────────────────────────────
        "threshold_sweep":  threshold_sweep,
        "best_threshold":   best_t,
        # ── Per-candidate breakdown (for error analysis) ──────────
        "per_candidate":    per_candidate,
    }


# ─────────────────────────────────────────────────────────────────
#  ADDITION 2 — Ablation study
#  Shows the contribution of each component to overall performance.
#  This is standard in NLP papers and reviewers will expect it.
#
#  How to use:
#    results = run_ablation_study(texts, labels, human_labels,
#                                 required_skills, job_description)
#
#  It runs 5 variants of your pipeline and reports metrics for each:
#    - Full system (your baseline)
#    - Without SBERT (TF-IDF only)
#    - Without bias detection
#    - Without skill gap analysis
#    - Without improvement suggestions (scores only)
# ─────────────────────────────────────────────────────────────────

def run_ablation_study(texts, labels, human_labels,
                       required_skills=None, job_description="", threshold=60):
    """
    Run ablation experiments — disable one component at a time and
    measure the impact on evaluation metrics vs. human ground truth.

    Args:
        texts            : list of resume text strings
        labels           : list of display labels (filenames / IDs)
        human_labels     : list of bools from recruiter (ground truth)
        required_skills  : list of required skill strings
        job_description  : job description string
        threshold        : shortlist score threshold

    Returns:
        list of dicts, one per ablation variant, with metrics.
        Print or export this as your paper's ablation table.
    """
    required_skills = required_skills or []

    def _score_variant(candidates):
        return evaluate_against_ground_truth(candidates, human_labels, threshold)

    ablation_results = []

    # ── Variant 1: Full system ────────────────────────────────────
    t0   = time.time()
    full = process_batch(texts, labels, required_skills, job_description)
    elapsed_full = round(time.time() - t0, 3)
    m    = _score_variant(full)
    ablation_results.append({
        "variant":   "Full system",
        "precision": m["precision"], "recall": m["recall"],
        "f1":        m["f1_score"],  "accuracy": m["accuracy"],
        "time_s":    elapsed_full,
    })

    # ── Variant 2: No SBERT — force TF-IDF ───────────────────────
    import os
    os.environ["FORCE_TFIDF"] = "true"
    # Re-import to pick up env flag (cheap — only changes similarity fn)
    import importlib, resume_parser as rp
    importlib.reload(rp)
    t0      = time.time()
    no_sbert = rp.process_batch(texts, labels, required_skills, job_description)
    elapsed_tfidf = round(time.time() - t0, 3)
    m = _score_variant(no_sbert)
    ablation_results.append({
        "variant":   "No SBERT (TF-IDF only)",
        "precision": m["precision"], "recall": m["recall"],
        "f1":        m["f1_score"],  "accuracy": m["accuracy"],
        "time_s":    elapsed_tfidf,
    })
    os.environ["FORCE_TFIDF"] = "false"
    importlib.reload(rp)

    # ── Variant 3: No bias detection ─────────────────────────────
    no_bias = [dict(c, bias_flags={}) for c in full]
    m = _score_variant(no_bias)
    ablation_results.append({
        "variant":   "No bias detection",
        "precision": m["precision"], "recall": m["recall"],
        "f1":        m["f1_score"],  "accuracy": m["accuracy"],
        "time_s":    None,   # same processing time as full
    })

    # ── Variant 4: No skill gap — zero matched/missing skills ─────
    no_skillgap = []
    for c in full:
        c2 = dict(c)
        # Recompute score without skill_match_score contribution
        sim = c["ai_similarity"]
        base_no_skill = (len(c["skills"]) * 3 +
                         len(c["projects"]) * 4 +
                         len(c["certificates"]) * 2 +
                         len(c["experience"]) * 5)
        c2["score"]          = round(base_no_skill + sim * 0.2, 2)
        c2["matched_skills"] = []
        c2["missing_skills"] = required_skills[:]
        c2["skill_match"]    = 0
        no_skillgap.append(c2)
    m = _score_variant(no_skillgap)
    ablation_results.append({
        "variant":   "No skill gap analysis",
        "precision": m["precision"], "recall": m["recall"],
        "f1":        m["f1_score"],  "accuracy": m["accuracy"],
        "time_s":    None,
    })

    # ── Variant 5: Score-only — no JD similarity ─────────────────
    no_jd = process_batch(texts, labels, required_skills, job_description="")
    m     = _score_variant(no_jd)
    ablation_results.append({
        "variant":   "No JD similarity (score only)",
        "precision": m["precision"], "recall": m["recall"],
        "f1":        m["f1_score"],  "accuracy": m["accuracy"],
        "time_s":    round(time.time() - t0, 3),
    })

    return ablation_results


# ─────────────────────────────────────────────────────────────────
#  ADDITION 3 — Pipeline performance stats
#  Aggregates the processing_time already stored per candidate into
#  the summary statistics your paper's results section needs.
#
#  How to use:
#    stats = compute_pipeline_stats(candidates)
#    # then log or display stats["avg_time_ms"], stats["throughput"], etc.
# ─────────────────────────────────────────────────────────────────

def compute_pipeline_stats(candidates):
    """
    Aggregate processing time and pipeline statistics across a batch.
    Uses the processing_time field already stored by process_batch().

    Args:
        candidates : list of candidate dicts from process_batch()

    Returns:
        dict of stats suitable for a paper's results / performance table.
    """
    if not candidates:
        return {}

    times_ms = [c["processing_time"] * 1000 for c in candidates]  # convert s → ms

    # ── Skill stats ───────────────────────────────────────────────
    skill_counts   = [len(c["skills"])      for c in candidates]
    project_counts = [len(c["projects"])    for c in candidates]
    cert_counts    = [len(c["certificates"])for c in candidates]
    scores         = [c["score"]            for c in candidates]
    bias_counts    = [len(c["bias_flags"])  for c in candidates]

    # ── Similarity method breakdown ───────────────────────────────
    methods = [c.get("sim_method", "unknown") for c in candidates]
    method_counts = {m: methods.count(m) for m in set(methods)}

    n = len(candidates)

    return {
        # ── Timing (report avg + std in your paper) ───────────────
        "n_resumes":           n,
        "avg_time_ms":         round(statistics.mean(times_ms), 2),
        "std_time_ms":         round(statistics.stdev(times_ms), 2) if n > 1 else 0,
        "min_time_ms":         round(min(times_ms), 2),
        "max_time_ms":         round(max(times_ms), 2),
        "total_time_s":        round(sum(times_ms) / 1000, 3),
        "throughput_per_min":  round(n / (sum(times_ms) / 1000) * 60, 1),

        # ── Score distribution ────────────────────────────────────
        "avg_score":           round(statistics.mean(scores), 2),
        "std_score":           round(statistics.stdev(scores), 2) if n > 1 else 0,
        "min_score":           round(min(scores), 2),
        "max_score":           round(max(scores), 2),

        # ── Extraction quality indicators ─────────────────────────
        "avg_skills_extracted":   round(statistics.mean(skill_counts), 1),
        "avg_projects_extracted": round(statistics.mean(project_counts), 1),
        "avg_certs_extracted":    round(statistics.mean(cert_counts), 1),

        # ── Bias summary ──────────────────────────────────────────
        "resumes_with_bias_flags": sum(1 for b in bias_counts if b > 0),
        "avg_bias_flags":          round(statistics.mean(bias_counts), 2),

        # ── Similarity method used ────────────────────────────────
        "similarity_methods":  method_counts,
        "sbert_available":     SBERT_AVAILABLE,
    }


# ─────────────────────────────────────────────────────────────────
#  QUICK DEMO — run this file directly to see sample output
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # ── Tiny synthetic test — replace with real resumes + labels ──
    sample_texts = [
        "John Smith\nPython developer with 3 years experience.\nSkills: Python, Django, Docker, SQL\nProjects: E-commerce API, Chat bot\nEducation: B.Tech Computer Science 2021",
        "Jane Doe\nData Scientist\nSkills: Python, TensorFlow, PyTorch, Pandas, NumPy\nProjects: Image classifier, NLP pipeline\nCertificates: AWS ML 2023\nEducation: M.Tech AI 2022",
        "Bob Johnson\nFront-end developer\nSkills: HTML, CSS, JavaScript, React\nEducation: BCA 2020",
        "Alice Wang\nBackend Engineer 5 years experience\nSkills: Java, Spring, Kubernetes, Docker, MySQL, Redis\nProjects: Payment gateway, Microservices platform\nEducation: B.E. 2019",
    ]
    sample_labels       = ["resume_1", "resume_2", "resume_3", "resume_4"]
    sample_human_labels = [True, True, False, True]   # ← recruiter decisions
    sample_skills       = ["python", "docker", "sql"]
    sample_jd           = "Looking for a backend developer with Python and Docker experience."

    print("Running process_batch()...")
    candidates = process_batch(sample_texts, sample_labels,
                               sample_skills, sample_jd)

    print("\n── Addition 1: Ground-truth evaluation ──")
    eval_result = evaluate_against_ground_truth(candidates, sample_human_labels)
    print(f"  Precision : {eval_result['precision']}%")
    print(f"  Recall    : {eval_result['recall']}%")
    print(f"  F1        : {eval_result['f1_score']}%")
    print(f"  Accuracy  : {eval_result['accuracy']}%")
    print(f"  Best threshold for F1: {eval_result['best_threshold']}")
    print("\n  Per-candidate:")
    for row in eval_result["per_candidate"]:
        status = "✓" if row["correct"] else "✗"
        print(f"    {status} {row['name']:<20} score={row['score']:>6}  "
              f"system={row['system_predict']:<12} human={row['human_label']}")

    print("\n── Addition 3: Pipeline stats ──")
    stats = compute_pipeline_stats(candidates)
    print(f"  Resumes processed  : {stats['n_resumes']}")
    print(f"  Avg time/resume    : {stats['avg_time_ms']} ms")
    print(f"  Throughput         : {stats['throughput_per_min']} resumes/min")
    print(f"  Avg score          : {stats['avg_score']}")
    print(f"  Avg skills found   : {stats['avg_skills_extracted']}")
    print(f"  SBERT available    : {stats['sbert_available']}")

    print("\n── Addition 2: Ablation study ──")
    print("  (runs 5 pipeline variants — may take a moment...)")
    ablation = run_ablation_study(sample_texts, sample_labels,
                                  sample_human_labels, sample_skills, sample_jd)
    print(f"\n  {'Variant':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}")
    print("  " + "-" * 60)
    for row in ablation:
        print(f"  {row['variant']:<35} {row['precision']:>5}% "
              f"{row['recall']:>5}% {row['f1']:>5}% {row['accuracy']:>5}%")