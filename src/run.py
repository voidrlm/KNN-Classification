#!/usr/bin/env python3
"""
Simple NN (1-NN) and KNN classifiers from scratch (no scikit-learn).
- Supports Euclidean and Manhattan distances
- Optional Z-score normalization
- Train/test split or K-fold cross-validation
- Confusion matrix and basic metrics

Usage examples:
  python src/run.py --data data/iris.csv --target target --algo nn --normalize
  python src/run.py --data data/iris.csv --target target --algo knn --k 5 --metric manhattan --test_size 0.3
  python src/run.py --data data/iris.csv --target target --algo knn --k 7 --cv 5

CSV requirements:
  - A header row
  - All feature columns must be numeric
  - One column must be the class label (use --target to name it, default='target')
"""

import csv
import argparse
import math
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import os

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # plotting is optional

# ----------------------------
# Utilities
# ----------------------------
def read_csv(path: str) -> Tuple[List[List[float]], List[str], List[str]]:
    """
    Returns (X, y, feature_names)
    - X: list of list of floats (n_samples x n_features)
    - y: list of class labels (strings)
    - feature_names: list of feature column names
    Assumes the target/label column name is provided separately and is present in the CSV.
    """
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    return rows, cols

def to_xy(rows, cols, target_col):
    if target_col not in cols:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {cols}")
    feature_cols = [c for c in cols if c != target_col]
    X, y = [], []
    for r in rows:
        # convert features to float
        X.append([float(r[c]) for c in feature_cols])
        y.append(r[target_col])
    return X, y, feature_cols

def train_test_split(X, y, test_size=0.2, seed=42):
    random.Random(seed).shuffle_indices = None  # just to keep seed usage explicit
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    cut = int(len(X) * (1 - test_size))
    train_idx, test_idx = idx[:cut], idx[cut:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_test  = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test

def zscore_normalize(train_X: List[List[float]], test_X: List[List[float]]):
    # compute mean/std on train only, then apply to both train and test
    n_features = len(train_X[0])
    means = []
    stds = []
    for j in range(n_features):
        col = [row[j] for row in train_X]
        m = sum(col) / len(col)
        v = sum((a - m)**2 for a in col) / len(col)
        s = math.sqrt(v) if v > 0 else 1.0
        means.append(m)
        stds.append(s)
    def normalize(X):
        Y = []
        for row in X:
            Y.append([(row[j] - means[j]) / stds[j] for j in range(n_features)])
        return Y
    return normalize(train_X), normalize(test_X)

def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

def get_metric_fn(name: str):
    name = name.lower()
    if name in ("euclidean", "l2"):
        return euclidean
    if name in ("manhattan", "l1", "taxicab"):
        return manhattan
    raise ValueError(f"Unknown metric: {name}")

# ----------------------------
# Classifiers
# ----------------------------
class NNClassifier:
    """ 1-Nearest Neighbor (a special case of KNN with k=1). """
    def __init__(self, metric="euclidean"):
        self.metric = get_metric_fn(metric)
        self.X = None
        self.y = None

    def fit(self, X: List[List[float]], y: List[str]):
        self.X = X
        self.y = y

    def predict_one(self, x: List[float]) -> str:
        best_dist = float("inf")
        best_label = None
        for xi, yi in zip(self.X, self.y):
            d = self.metric(x, xi)
            if d < best_dist:
                best_dist = d
                best_label = yi
        return best_label

    def predict(self, X: List[List[float]]) -> List[str]:
        return [self.predict_one(x) for x in X]

class KNNClassifier:
    """ K-Nearest Neighbors (majority vote). """
    def __init__(self, k=5, metric="euclidean"):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.metric = get_metric_fn(metric)
        self.X = None
        self.y = None

    def fit(self, X: List[List[float]], y: List[str]):
        self.X = X
        self.y = y

    def predict_one(self, x: List[float]) -> str:
        # compute all distances
        dists = [(self.metric(x, xi), yi) for xi, yi in zip(self.X, self.y)]
        dists.sort(key=lambda t: t[0])
        k_nearest = dists[:self.k]
        votes = Counter(yi for _, yi in k_nearest)
        # majority vote; tie-break by nearest neighbor among tied labels
        max_vote = max(votes.values())
        candidates = [lab for lab, cnt in votes.items() if cnt == max_vote]
        if len(candidates) == 1:
            return candidates[0]
        # tie-breaker: pick label of the closest neighbor among candidates
        for dist, lab in k_nearest:
            if lab in candidates:
                return lab
        return k_nearest[0][1]

    def predict(self, X: List[List[float]]) -> List[str]:
        return [self.predict_one(x) for x in X]

# ----------------------------
# Metrics
# ----------------------------
def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true) if y_true else 0.0

def confusion_matrix(y_true: List[str], y_pred: List[str]) -> Tuple[List[List[int]], List[str]]:
    labels = sorted(set(y_true) | set(y_pred))
    index = {lab: i for i, lab in enumerate(labels)}
    m = [[0 for _ in labels] for _ in labels]
    for yt, yp in zip(y_true, y_pred):
        m[index[yt]][index[yp]] += 1
    return m, labels

def precision_recall_f1(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, float]]:
    # per-class precision, recall, f1
    m, labels = confusion_matrix(y_true, y_pred)
    res = {}
    for i, lab in enumerate(labels):
        tp = m[i][i]
        fp = sum(m[r][i] for r in range(len(labels)) if r != i)
        fn = sum(m[i][c] for c in range(len(labels)) if c != i)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        res[lab] = {"precision": prec, "recall": rec, "f1": f1}
    return res

def plot_confusion_matrix(cm, labels, title="Confusion Matrix", out_path=None):
    if plt is None:
        return
    import numpy as np
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    # add values
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center')
    fig.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()

# ----------------------------
# Cross-validation
# ----------------------------
def kfold_indices(n, k, seed=42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    folds = [[] for _ in range(k)]
    for i, val in enumerate(idx):
        folds[i % k].append(val)
    return folds

def run_cv(X, y, algo="knn", k_neighbors=5, metric="euclidean", normalize=False, folds=5, seed=42):
    folds_idx = kfold_indices(len(X), folds, seed=seed)
    accs = []
    for i in range(folds):
        test_idx = set(folds_idx[i])
        X_train = [X[j] for j in range(len(X)) if j not in test_idx]
        y_train = [y[j] for j in range(len(y)) if j not in test_idx]
        X_test  = [X[j] for j in range(len(X)) if j in test_idx]
        y_test  = [y[j] for j in range(len(y)) if j in test_idx]

        if normalize:
            X_train, X_test = zscore_normalize(X_train, X_test)

        if algo == "nn":
            clf = NNClassifier(metric=metric)
        else:
            clf = KNNClassifier(k=k_neighbors, metric=metric)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        accs.append(accuracy(y_test, preds))
    return sum(accs)/len(accs), accs

# ----------------------------
# Main script
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/iris.csv", help="Path to CSV data")
    parser.add_argument("--target", type=str, default="target", help="Target/label column name")
    parser.add_argument("--algo", type=str, choices=["nn","knn"], default="knn")
    parser.add_argument("--k", type=int, default=5, help="k for KNN (ignored for NN)")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance metric: euclidean|manhattan")
    parser.add_argument("--normalize", action="store_true", help="Z-score normalize features")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion for test split")
    parser.add_argument("--cv", type=int, default=0, help="If >0, use K-fold CV with this many folds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_cm", type=str, default="figs/confusion_matrix.png")
    args = parser.parse_args()

    rows, cols = read_csv(args.data)
    X, y, feature_cols = to_xy(rows, cols, args.target)

    if args.cv and args.cv > 1:
        mean_acc, accs = run_cv(X, y, algo=args.algo, k_neighbors=args.k, metric=args.metric,
                                normalize=args.normalize, folds=args.cv, seed=args.seed)
        print(f"[CV] algo={args.algo}, metric={args.metric}, k={args.k}, normalize={args.normalize}, folds={args.cv}")
        print("Fold accuracies:", ", ".join(f"{a:.3f}" for a in accs))
        print(f"Mean accuracy: {mean_acc:.3f}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, seed=args.seed)
        if args.normalize:
            X_train, X_test = zscore_normalize(X_train, X_test)
        if args.algo == "nn":
            clf = NNClassifier(metric=args.metric)
        else:
            clf = KNNClassifier(k=args.k, metric=args.metric)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = accuracy(y_test, preds)
        print(f"Algorithm: {args.algo.upper()}  Metric: {args.metric}  k={args.k if args.algo=='knn' else 1}  Normalize={args.normalize}")
        print(f"Test accuracy: {acc:.3f}")
        prf1 = precision_recall_f1(y_test, preds)
        print("Per-class metrics:")
        for lab, d in prf1.items():
            print(f"  {lab:15s}  precision={d['precision']:.3f}  recall={d['recall']:.3f}  f1={d['f1']:.3f}")
        cm, labels = confusion_matrix(y_test, preds)
        print("\nConfusion matrix (rows=true, cols=pred):")
        header = "        " + "  ".join(f"{lab:>12s}" for lab in labels)
        print(header)
        for lab, row in zip(labels, cm):
            print(f"{lab:>8s}  " + "  ".join(f"{v:12d}" for v in row))

        # plot/save confusion matrix if matplotlib exists
        if plt is not None:
            os.makedirs(os.path.dirname(args.out_cm), exist_ok=True)
            plot_confusion_matrix(cm, labels, out_path=args.out_cm)
            print(f"\nSaved confusion matrix to: {args.out_cm}")

if __name__ == "__main__":
    main()
