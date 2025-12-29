# Mercor Cheating Detection — 5‑file pipeline

## Overview
This pipeline trains a cost‑aware ensemble using:
- LightGBM on tabular + graph features
- Positive–Unlabeled (PU) weighting with `high_conf_clean`
- Lightweight GNN embeddings
- Isotonic calibration for probability stability
- Local cost evaluation with threshold grid search

## Files
- `main.py` — orchestrates training, evaluation, submission
- `losses.py` — cost metric + PU weighting
- `graph.py` — graph stats + GNN embeddings
- `models.py` — LightGBM + calibration
- `README.md` — usage guide

## Usage
1. Place `train.csv`, `test.csv`, `edges.csv` in working directory.
2. Install deps:  
   `pip install numpy pandas scikit-learn lightgbm networkx torch`
3. Train:  
   `python main.py --mode train --train train.csv --test test.csv --edges edges.csv --out artifacts`
4. Submit:  
   `python main.py --mode submit --test test.csv --edges edges.csv --artifacts artifacts --out submission.csv`
5. Upload `submission.csv` on Kaggle competition page.
