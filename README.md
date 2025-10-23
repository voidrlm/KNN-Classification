# NN & KNN Classification — From Scratch (Python)  
A minimal, VS Code–friendly project showing **Nearest Neighbor (NN)** and **K-Nearest Neighbors (KNN)** on a multi-dimensional dataset (Iris). No scikit-learn needed.

## What you get
- `src/run.py` — pure Python implementation of NN and KNN with Euclidean/Manhattan distances, normalization, train/test split, and K-fold CV
- `data/iris.csv` — classic 3-class dataset (150 samples × 4 features)
- `figs/` — confusion-matrix output location
- `matlab/knn_demo.m` — MATLAB reference showing how to run KNN on the same CSV

## Quick start
```bash
# 1) (optional) create a virtual environment
python -m venv .venv && .venv/Scripts/activate  # on Windows
# or: source .venv/bin/activate                  # on macOS/Linux

# 2) install optional plotting lib
pip install matplotlib

# 3) run NN (1-NN)
python src/run.py --data data/iris.csv --target target --algo nn --normalize

# 4) run KNN (k=5)
python src/run.py --data data/iris.csv --target target --algo knn --k 5 --metric euclidean --normalize

# 5) 5-fold cross-validation with KNN
python src/run.py --data data/iris.csv --target target --algo knn --k 7 --cv 5
```

The script prints accuracy, per-class precision/recall/F1, and saves a confusion matrix figure to `figs/confusion_matrix.png` when `matplotlib` is installed.

## Concept: Classification, NN, and KNN
**Classification** assigns a discrete label to an input vector \(x \in \mathbb{R}^d\). We learn from labeled examples (supervised learning) and then predict labels for new points.

### 1-NN (Nearest Neighbor)
- Store all training points.
- For a new sample, compute its distance to each training point and **copy the label of the closest**.
- Pros: extremely simple, no training time.
- Cons: sensitive to noise and irrelevant features, slow at prediction (must scan all points), and performance depends on how distance relates to class structure.

### K-NN (K-Nearest Neighbors)
- Consider the **k** closest points and take the **majority vote** of their labels.
- Pros: smooths out noise, often better than 1-NN.
- Cons: choice of **k** matters; too small → noisy, too large → over-smoothing. Needs feature scaling; suffers in very high dimensions (curse of dimensionality).

### Practical tips
- **Normalization** (e.g., Z-score) helps when features are on different scales.
- Try **Euclidean** and **Manhattan** distances.
- Use **cross-validation** (e.g., `--cv 5`) to pick `k` robustly.
- For large datasets, use approximate neighbors or tree/graph indices (not included here to keep it simple).

## MATLAB usage
See `matlab/knn_demo.m` for a small KNN demo loading the same CSV. Run it in MATLAB/Octave after adjusting the path if needed.

## Dataset
Iris: 150 flowers, 4 numeric features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) and 3 classes (`setosa`, `versicolor`, `virginica`).

## License
MIT
