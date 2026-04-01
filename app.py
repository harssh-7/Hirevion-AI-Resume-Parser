"""
HireVion — AI-Powered Resume Intelligence Platform  v2
Flask Backend | Fixed: upload freeze, model cold-start, error handling
"""

import os, json, uuid, hashlib, tempfile, threading
from pathlib import Path
from functools import wraps

import numpy as np
from flask import (Flask, render_template, request, redirect,
                   url_for, session, Response, jsonify)
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd

# ── Lazy import guard: resume_parser loads SBERT on import, which can
#    take 10-30 s on a cold start.  We import it once at module level
#    (happens before the first request is served), so the server start
#    takes the hit — NOT the upload request. ─────────────────────────
from resume_parser import (
    process_batch, compute_evaluation_metrics,
    compute_pool_bias_report, SBERT_AVAILABLE
)
from research_additions import compute_pipeline_stats, run_ablation_study

# ─────────────────────────────────────────────────────────────
#  APP CONFIG
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'hirevion-x9k2m7p4q1-2025')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, 'uploads')
RESULT_DIR  = os.path.join(BASE_DIR, 'results')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

USERS_FILE  = os.path.join(BASE_DIR, 'hirevion_users.json')
ALLOWED_EXT = {'pdf', 'docx', 'txt', 'csv'}


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _load_users():
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {'admin': hashlib.sha256(b'1234').hexdigest()}

def _save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def _allowed(fname: str) -> bool:
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def _extract_text(path: str) -> str:
    """Extract plain text from PDF, DOCX or plain text file."""
    ext = path.rsplit('.', 1)[-1].lower()
    try:
        if ext == 'pdf':
            r = PdfReader(path)
            return '\n'.join(p.extract_text() or '' for p in r.pages)
        elif ext == 'docx':
            d = Document(path)
            return '\n'.join(p.text for p in d.paragraphs)
        else:
            with open(path, errors='ignore') as f:
                return f.read()
    except Exception:
        return ''

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

def _clean(obj):
    return json.loads(json.dumps(obj, cls=_NpEncoder))


# ─────────────────────────────────────────────────────────────
#  ROUTES — Auth
# ─────────────────────────────────────────────────────────────

@app.route('/')
def welcome():
    if session.get('logged_in'):
        return redirect(url_for('upload'))
    return render_template('welcome.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('upload'))
    error = None
    if request.method == 'POST':
        action = request.form.get('action', 'login')
        u  = request.form.get('username', '').strip()
        p  = request.form.get('password', '')
        p2 = request.form.get('confirm_password', '')
        users = _load_users()
        if action == 'login':
            if users.get(u) == _hash(p):
                session['logged_in'] = True
                session['username']  = u
                return redirect(url_for('upload'))
            error = 'Invalid credentials — please try again.'
        elif action == 'register':
            if not u or not p:
                error = 'Username and password are required.'
            elif p != p2:
                error = 'Passwords do not match.'
            elif u in users:
                error = 'Username already exists.'
            else:
                users[u] = _hash(p)
                _save_users(users)
                session['logged_in'] = True
                session['username']  = u
                return redirect(url_for('upload'))
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('welcome'))


# ─────────────────────────────────────────────────────────────
#  ROUTES — Core
# ─────────────────────────────────────────────────────────────

@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html', username=session.get('username', 'Admin'))


@app.route('/process', methods=['POST'])
@login_required
def process():
    jd         = request.form.get('job_description', '').strip()
    rs_raw     = request.form.get('required_skills', '').strip()
    req_skills = [s.strip().lower() for s in rs_raw.split(',') if s.strip()]
    anonymize  = request.form.get('anonymize', 'false') == 'true'

    texts, labels, prefills = [], [], []

    # ── Handle individual file uploads ───────────────────────
    for f in request.files.getlist('resumes'):
        if f and f.filename and _allowed(f.filename):
            fname = secure_filename(f.filename)
            path  = os.path.join(UPLOAD_DIR, fname)
            try:
                f.save(path)
                txt = _extract_text(path)
                if txt.strip():
                    texts.append(txt)
                    labels.append(fname)
                    prefills.append({})
            except Exception as e:
                app.logger.warning(f'Failed to process {fname}: {e}')

    # ── Handle CSV bulk upload ────────────────────────────────
    csv_f = request.files.get('csv_file')
    if csv_f and csv_f.filename and csv_f.filename.lower().endswith('.csv'):
        fname = secure_filename(csv_f.filename)
        path  = os.path.join(UPLOAD_DIR, fname)
        try:
            csv_f.save(path)
            df = pd.read_csv(path)
            text_col = next(
                (c for c in df.columns if any(k in c.lower()
                 for k in ('resume', 'text', 'content', 'summary'))),
                df.columns[0]
            )
            for idx, row in df.iterrows():
                txt = str(row.get(text_col, ''))
                if txt.strip():
                    texts.append(txt)
                    labels.append(str(row.get('filename', f'record_{idx+1}')))
                    prefills.append({
                        'name':  str(row.get('name',  row.get('Name',  ''))),
                        'email': str(row.get('email', row.get('Email', ''))),
                        'phone': str(row.get('phone', row.get('Phone', ''))),
                    })
        except Exception as e:
            app.logger.warning(f'Failed to process CSV: {e}')

    if not texts:
        return render_template('upload.html',
                               username=session.get('username', 'Admin'),
                               error='No valid resumes found. Upload PDF, DOCX, or CSV.')

    # ── Run NLP pipeline ─────────────────────────────────────
    # FIX: wrap in try/except so any parser error returns a
    # clean error page instead of a silent freeze / 500.
    try:
        candidates = process_batch(texts, labels, req_skills, jd,
                                   prefills=prefills, anonymize=anonymize)
    except Exception as e:
        app.logger.error(f'process_batch failed: {e}')
        return render_template('upload.html',
                               username=session.get('username', 'Admin'),
                               error=f'Processing error: {str(e)[:200]}. Try fewer files or check your PDF is not encrypted.')

    candidates.sort(key=lambda x: x['score'], reverse=True)
    for i, c in enumerate(candidates):
        c['rank'] = i + 1

    # ── Metrics & stats ───────────────────────────────────────
    eval_metrics   = compute_evaluation_metrics(candidates)
    pipeline_stats = compute_pipeline_stats(candidates)
    bias_report    = compute_pool_bias_report(candidates)

    result = _clean({
        'candidates':      candidates,
        'job_description': jd,
        'required_skills': req_skills,
        'eval_metrics':    eval_metrics,
        'pipeline_stats':  pipeline_stats,
        'bias_report':     bias_report,
        'sbert_available': bool(SBERT_AVAILABLE),
        'anonymized':      anonymize,
    })

    rid   = str(uuid.uuid4())
    rpath = os.path.join(RESULT_DIR, f'{rid}.json')
    with open(rpath, 'w') as fh:
        json.dump(result, fh)
    session['result_id'] = rid

    return redirect(url_for('dashboard'))


@app.route('/dashboard')
@login_required
def dashboard():
    rid = session.get('result_id')
    if not rid:
        return redirect(url_for('upload'))
    rpath = os.path.join(RESULT_DIR, f'{rid}.json')
    if not os.path.exists(rpath):
        return redirect(url_for('upload'))
    with open(rpath) as f:
        data = json.load(f)
    return render_template('dashboard.html',
                           data=data,
                           username=session.get('username', 'Admin'))


@app.route('/bias-report')
@login_required
def bias_report():
    rid = session.get('result_id')
    if not rid:
        return redirect(url_for('upload'))
    rpath = os.path.join(RESULT_DIR, f'{rid}.json')
    if not os.path.exists(rpath):
        return redirect(url_for('upload'))
    with open(rpath) as f:
        data = json.load(f)
    return render_template('bias_report.html',
                           data=data,
                           username=session.get('username', 'Admin'))


@app.route('/research')
@login_required
def research():
    rid = session.get('result_id')
    if not rid:
        return redirect(url_for('upload'))
    rpath = os.path.join(RESULT_DIR, f'{rid}.json')
    if not os.path.exists(rpath):
        return redirect(url_for('upload'))
    with open(rpath) as f:
        data = json.load(f)
    return render_template('research.html',
                           data=data,
                           username=session.get('username', 'Admin'))


@app.route('/export')
@login_required
def export():
    rid = session.get('result_id')
    if not rid:
        return redirect(url_for('upload'))
    rpath = os.path.join(RESULT_DIR, f'{rid}.json')
    if not os.path.exists(rpath):
        return redirect(url_for('upload'))
    with open(rpath) as f:
        data = json.load(f)

    rows = []
    for c in data['candidates']:
        rows.append({
            'Rank':             c.get('rank', ''),
            'Name':             c.get('name', ''),
            'Email':            c.get('email', ''),
            'Phone':            c.get('phone', ''),
            'Score':            c.get('score', 0),
            'Category':         c.get('category', ''),
            'Recommendation':   c.get('recommendation', ''),
            'Domain':           c.get('domain', ''),
            'Skills Count':     len(c.get('skills', [])),
            'Matched Skills':   ', '.join(c.get('matched_skills', [])),
            'Missing Skills':   ', '.join(c.get('missing_skills', [])),
            'Skill Match %':    round(c.get('skill_match', 0), 1),
            'Experience Years': c.get('experience_years', 0),
            'AI Similarity %':  round(c.get('ai_similarity', 0), 1),
            'Bias Flags':       len(c.get('bias_flags', {})),
            'Language':         c.get('language', {}).get('name', 'English'),
            'Anonymized':       c.get('anonymized', False),
            'File':             c.get('filename', ''),
        })

    csv_str = pd.DataFrame(rows).to_csv(index=False)
    return Response(
        csv_str,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=hirevion_results.csv'}
    )


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    # threaded=True is CRITICAL: without it Flask handles one request
    # at a time — the browser's prefetch + the form POST compete and
    # one of them blocks forever, causing the "upload page freeze".
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
