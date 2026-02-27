#!/usr/bin/env bash
set -euo pipefail

RAW_DIR=${1:-data/raw}

python src/preprocess.py --raw-dir "$RAW_DIR"
python src/build_kg.py
python src/train_cf.py
python src/train_svd.py
python src/train_kg.py
python src/evaluate.py
python src/visualize.py --model-dir kg_gcn
