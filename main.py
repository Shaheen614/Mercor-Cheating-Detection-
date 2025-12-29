

import argparse
import os
import numpy as np
import pandas as pd

from graph import build_graph_augmented_features
from models import train_lgbm, predict_lgbm, IsotonicCalibrator
from losses import assign_pu_weights, grid_search_cost

SEED = 42

BASE_FEATURES = [f"feature_{i:03d}" for i in range(1, 19)]
ID_COL = "user_hash"


def _build_feature_frames(train_df, test_df, edges_df):
    # Graph modules (stats + embeddings)
    stats_train, emb_train = build_graph_augmented_features(edges_df, train_df[ID_COL].tolist())
    stats_test, emb_test = build_graph_augmented_features(edges_df, test_df[ID_COL].tolist())

    # Assemble train features
    add_cols = [c for c in BASE_FEATURES if c in train_df.columns]
    if "high_conf_clean" in train_df.columns:
        add_cols.append("high_conf_clean")

    X_train = train_df[[ID_COL] + add_cols].merge(stats_train, on=ID_COL, how="left")
    X_train = X_train.merge(emb_train, on=ID_COL, how="left")

    # Assemble test features
    add_cols_test = [c for c in BASE_FEATURES if c in test_df.columns]
    if "high_conf_clean" in test_df.columns:
        add_cols_test.append("high_conf_clean")

    X_test = test_df[[ID_COL] + add_cols_test].merge(stats_test, on=ID_COL, how="left")
    X_test = X_test.merge(emb_test, on=ID_COL, how="left")

    # Fill NA
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Feature cols (exclude ID)
    feature_cols = [c for c in X_train.columns if c != ID_COL]
    return X_train, X_test, feature_cols


def run_train(train_path, test_path, edges_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    edges_df = pd.read_csv(edges_path)

    # Build feature matrices
    X_train, X_test, feature_cols = _build_feature_frames(train_df, test_df, edges_df)

    # Targets and PU weights
    y = train_df["is_cheating"]
    w = assign_pu_weights(train_df)

    # Train LightGBM (CV ensemble)
    models, oof, auc = train_lgbm(X_train, y, feature_cols, w, seed=SEED, n_splits=5)
    print(f"OOF AUC: {auc:.5f}")

    # Calibrate on labeled OOF
    labeled_idx = y.notna().values
    cal = IsotonicCalibrator()
    cal.fit(oof[labeled_idx], y.loc[labeled_idx])

    # Local cost sanity check
    y_lab = y.loc[labeled_idx].astype(int).values
    oof_cal = cal.transform(oof[labeled_idx])
    best = grid_search_cost(y_lab, oof_cal, grid=101)
    print(f"Local best cost: {best}")

    # Save artifacts
    # Feature matrices (CSV for portability)
    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)

    # IDs and labels
    train_df[[ID_COL, "is_cheating"]].to_csv(os.path.join(out_dir, "train_labels.csv"), index=False)

    # OOF + labeled targets for recalibration at submit
    np.save(os.path.join(out_dir, "oof.npy"), oof)
    np.save(os.path.join(out_dir, "y_lab.npy"), y_lab)

    # Save models
    import lightgbm as lgb  # local import to avoid hard dependency when only submitting
    for i, m in enumerate(models):
        m.save_model(os.path.join(out_dir, f"lgbm_{i}.txt"))

    # Save feature columns
    with open(os.path.join(out_dir, "feature_cols.txt"), "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    print(f"Artifacts saved to: {out_dir}")


def run_evaluate(artifacts_dir):
    # Load artifacts
    X_train = pd.read_csv(os.path.join(artifacts_dir, "X_train.csv"))
    labels = pd.read_csv(os.path.join(artifacts_dir, "train_labels.csv"))
    y = labels["is_cheating"]

    feature_cols = [c for c in X_train.columns if c != ID_COL]

    # Load models
    import lightgbm as lgb
    models = []
    i = 0
    while True:
        path = os.path.join(artifacts_dir, f"lgbm_{i}.txt")
        if not os.path.exists(path):
            break
        models.append(lgb.Booster(model_file=path))
        i += 1
    if len(models) == 0:
        raise RuntimeError("No LightGBM models found in artifacts.")

    # OOF and labels
    oof = np.load(os.path.join(artifacts_dir, "oof.npy"))
    y_lab = np.load(os.path.join(artifacts_dir, "y_lab.npy"))

    # Calibrate and compute local cost
    cal = IsotonicCalibrator()
    cal.fit(oof[:len(y_lab)], y_lab)
    best = grid_search_cost(y_lab, cal.transform(oof[:len(y_lab)]), grid=101)
    print(f"Local best cost (re-evaluated): {best}")


def run_submit(test_path, edges_path, artifacts_dir, out_csv):
    # Load artifacts
    X_test = pd.read_csv(os.path.join(artifacts_dir, "X_test.csv"))
    feature_cols_path = os.path.join(artifacts_dir, "feature_cols.txt")
    if os.path.exists(feature_cols_path):
        with open(feature_cols_path, "r") as f:
            feature_cols = [ln.strip() for ln in f if ln.strip()]
    else:
        feature_cols = [c for c in X_test.columns if c != ID_COL]

    # Load models
    import lightgbm as lgb
    models = []
    i = 0
    while True:
        path = os.path.join(artifacts_dir, f"lgbm_{i}.txt")
        if not os.path.exists(path):
            break
        models.append(lgb.Booster(model_file=path))
        i += 1
    if len(models) == 0:
        raise RuntimeError("No LightGBM models found in artifacts.")

    # Raw predictions
    preds_raw = np.mean([m.predict(X_test[feature_cols], num_iteration=m.best_iteration) for m in models], axis=0)

    # Fit calibrator using stored labeled OOF pairs
    oof = np.load(os.path.join(artifacts_dir, "oof.npy"))
    y_lab = np.load(os.path.join(artifacts_dir, "y_lab.npy"))
    cal = IsotonicCalibrator()
    cal.fit(oof[:len(y_lab)], y_lab)

    preds = cal.transform(preds_raw)
    sub = pd.DataFrame({ID_COL: X_test[ID_COL], "prediction": np.clip(preds, 0.0, 1.0)})
    sub.to_csv(out_csv, index=False)
    print(f"Saved submission: {out_csv}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["train", "evaluate", "submit"])
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--edges", default="edges.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--artifacts", default="artifacts")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_train(args.train, args.test, args.edges, args.out)
    elif args.mode == "evaluate":
        run_evaluate(args.artifacts)
    elif args.mode == "submit":
        run_submit(args.test, args.edges, args.artifacts, args.out)
