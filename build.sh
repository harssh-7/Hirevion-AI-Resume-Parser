#!/usr/bin/env bash
# build.sh — run this once before starting the server
set -e
pip install -r requirements.txt
python -m spacy download en_core_web_sm
echo "✓ HireVion build complete"
