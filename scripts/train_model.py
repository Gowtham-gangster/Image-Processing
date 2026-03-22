"""
train_model.py
==============
Training pipeline for the Mask-Aware Hybrid Person Identification System.

Steps
-----
1. Load every training image from  dataset/train/<person_id>/.
2. Extract a 1 280-D MobileNetV2 embedding for each image.
3. Train and cross-validate an SVM **or** KNN classifier.
4. Save the trained pipeline to  models/person_identifier.pkl.
5. Print a detailed classification report on the training data.

Saved artefacts
---------------
models/person_identifier.pkl
    A joblib-serialised dict:
        'pipeline'   : sklearn Pipeline  (scaler → classifier)
        'label_map'  : {int → person_id}
        'person_ids' : sorted list of person folder names
        'classifier' : 'svm' or 'knn'
        'embed_dim'  : 1280
        'trained_at' : ISO-8601 timestamp

embeddings/embeddings_store.npz   (written by EmbeddingExtractor)

Usage
-----
    python train_model.py                      # SVM (default), auto DL backend
    python train_model.py --clf knn            # KNN classifier
    python train_model.py --clf svm --c 5.0   # custom SVM regularisation
    python train_model.py --backend torch      # force PyTorch for embeddings
    python train_model.py --no-cv              # skip cross-validation search
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config import MODELS_DIR, TRAIN_DIR, LOG_LEVEL
from embeddings import EmbeddingExtractor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_OUT = os.path.join(MODELS_DIR, "person_identifier.pkl")
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_embeddings(
    extractor: EmbeddingExtractor,
    train_dir: str = TRAIN_DIR,
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """
    Walk *train_dir*, extract MobileNetV2 embeddings for every image,
    and build matching integer label arrays.

    Returns
    -------
    X         : np.ndarray  (N, 1280)  float32
    y         : np.ndarray  (N,)       int32   (numeric labels)
    label_map : dict {int → person_id}
    """
    import cv2                                               # local import – not needed at top

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: '{train_dir}'")

    person_ids = sorted(
        p for p in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, p))
    )
    if not person_ids:
        raise ValueError(
            f"No person sub-folders found in '{train_dir}'. "
            "Each sub-folder name must match a person_id in persons.csv."
        )

    label_map:   dict[int, str]   = {i: pid for i, pid in enumerate(person_ids)}
    pid_to_int:  dict[str, int]   = {pid: i for i, pid in label_map.items()}
    all_embeds:  list[np.ndarray] = []
    all_labels:  list[int]        = []

    total_images = 0
    logger.info("Extracting embeddings from %d identities …", len(person_ids))

    for pid in person_ids:
        person_dir = os.path.join(train_dir, pid)
        img_files  = sorted(
            f for f in os.listdir(person_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        )
        if not img_files:
            logger.warning("No images found for '%s' — skipping.", pid)
            continue

        label = pid_to_int[pid]
        person_count = 0

        for fname in img_files:
            img_path = os.path.join(person_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning("  Cannot read '%s' — skipping.", img_path)
                continue

            emb = extractor.extract(img)         # (1280,) L2-normalised float32
            all_embeds.append(emb)
            all_labels.append(label)
            person_count += 1

        logger.info("  [%s] %d embeddings extracted.", pid, person_count)
        total_images += person_count

    if not all_embeds:
        raise RuntimeError(
            "Zero embeddings extracted. "
            "Ensure training folders contain readable image files."
        )

    X = np.vstack(all_embeds).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)

    logger.info("Dataset ready: %d samples × %d dims | %d classes.",
                X.shape[0], X.shape[1], len(label_map))
    return X, y, label_map


# ── Classifier builders ───────────────────────────────────────────────────────

def build_svm_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    c: float = 1.0,
    kernel: str = "rbf",
    run_grid_search: bool = True,
) -> Pipeline:
    """
    Build (and optionally tune) a StandardScaler → SVC pipeline.

    Grid search explores C ∈ {0.1, 1, 10} and kernels ∈ {rbf, linear}
    with 5-fold stratified cross-validation.

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    scaler = StandardScaler()

    if run_grid_search and len(np.unique(y)) > 1 and len(y) >= 10:
        logger.info("Running SVM grid search (5-fold CV) …")
        param_grid = {
            "clf__C":      [0.1, 1.0, 5.0, 10.0],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma":  ["scale", "auto"],
        }
        base_pipe = Pipeline([
            ("scaler", scaler),
            ("clf",    SVC(probability=True, class_weight="balanced", random_state=42)),
        ])
        cv = StratifiedKFold(n_splits=min(5, _min_class_count(y)), shuffle=True,
                             random_state=42)
        gs = GridSearchCV(
            base_pipe, param_grid,
            cv=cv, scoring="accuracy",
            n_jobs=-1, verbose=1,
            refit=True,
        )
        gs.fit(X, y)
        logger.info("Best SVM params: %s  →  CV accuracy: %.3f",
                    gs.best_params_, gs.best_score_)
        return gs.best_estimator_

    # Fixed params (fast path)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(C=c, kernel=kernel, probability=True,
                       class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X, y)
    return pipeline


def build_knn_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 5,
    run_grid_search: bool = True,
) -> Pipeline:
    """
    Build (and optionally tune) a StandardScaler → KNeighborsClassifier pipeline.

    Grid search explores k ∈ {1, 3, 5, 7, 11} and metrics ∈ {euclidean, cosine}.

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    n_classes = len(np.unique(y))

    if run_grid_search and n_classes > 1 and len(y) >= 10:
        logger.info("Running KNN grid search (5-fold CV) …")
        max_k = max(1, min(11, len(y) // n_classes - 1))
        k_values = [k for k in [1, 3, 5, 7, 11] if k <= max_k] or [1]
        param_grid = {
            "clf__n_neighbors": k_values,
            "clf__metric":      ["euclidean", "cosine"],
            "clf__weights":     ["uniform", "distance"],
        }
        base_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier()),
        ])
        cv = StratifiedKFold(n_splits=min(5, _min_class_count(y)), shuffle=True,
                             random_state=42)
        gs = GridSearchCV(
            base_pipe, param_grid,
            cv=cv, scoring="accuracy",
            n_jobs=-1, verbose=1,
            refit=True,
        )
        gs.fit(X, y)
        logger.info("Best KNN params: %s  →  CV accuracy: %.3f",
                    gs.best_params_, gs.best_score_)
        return gs.best_estimator_

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="cosine",
            weights="distance",
        )),
    ])
    pipeline.fit(X, y)
    return pipeline


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(
    pipeline: Pipeline,
    label_map: dict[int, str],
    classifier: str,
    out_path: str = MODEL_OUT,
) -> None:
    """Serialise the trained pipeline + metadata to disk with joblib."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "pipeline":   pipeline,
        "label_map":  label_map,
        "person_ids": list(label_map.values()),
        "classifier": classifier,
        "embed_dim":  1280,
        "trained_at": datetime.datetime.now().isoformat(),
    }
    joblib.dump(payload, out_path, compress=3)
    logger.info("Model saved → %s", out_path)


def load_model(path: str = MODEL_OUT) -> dict:
    """Load a saved model payload from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. "
            "Run  python train_model.py  first."
        )
    payload = joblib.load(path)
    logger.info(
        "Model loaded | classifier=%s | persons=%d | trained_at=%s",
        payload["classifier"], len(payload["label_map"]), payload["trained_at"],
    )
    return payload


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    label_map: dict[int, str],
) -> None:
    """Print accuracy + sklearn classification_report on training data."""
    y_pred  = pipeline.predict(X)
    acc     = accuracy_score(y, y_pred)
    targets = [label_map[i] for i in sorted(label_map)]

    print("\n── Training Report ──────────────────────────────────────────────")
    print(f"  Training accuracy : {acc:.4f}  ({acc:.1%})")
    print("\n  Per-class breakdown (on training data):")
    print(
        classification_report(
            y, y_pred,
            target_names=targets,
            zero_division=0,
        )
    )
    print("─────────────────────────────────────────────────────────────────\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _min_class_count(y: np.ndarray) -> int:
    """Return the size of the smallest class (for setting n_splits safely)."""
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())


# ── CLI + main ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SVM/KNN person identifier from MobileNetV2 embeddings."
    )
    parser.add_argument(
        "--clf", choices=["svm", "knn"], default="svm",
        help="Classifier type (default: svm).",
    )
    parser.add_argument(
        "--backend", choices=["auto", "tf", "torch"], default="auto",
        help="Deep learning backend for embedding extraction (default: auto).",
    )
    parser.add_argument(
        "--c", type=float, default=1.0,
        help="SVM regularisation parameter C (default: 1.0, ignored with --no-cv).",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="KNN: number of neighbours (default: 5, ignored with --no-cv).",
    )
    parser.add_argument(
        "--kernel", choices=["rbf", "linear"], default="rbf",
        help="SVM kernel (default: rbf, ignored with --no-cv).",
    )
    parser.add_argument(
        "--no-cv", action="store_true",
        help="Skip grid-search cross-validation (faster, uses fixed hyperparams).",
    )
    parser.add_argument(
        "--out", default=MODEL_OUT,
        help=f"Output model path (default: {MODEL_OUT}).",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """End-to-end training pipeline."""
    print("\n══ Mask-Aware Person Identifier — Training ══════════════════════")
    logger.info("Classifier : %s", args.clf.upper())
    logger.info("DL backend : %s", args.backend)
    logger.info("Grid-search: %s", not args.no_cv)

    # ── Step 1 : Embedding extraction ────────────────────────────────────────
    logger.info("Step 1/3 — Initialising embedding extractor …")
    extractor = EmbeddingExtractor(backend=args.backend)

    logger.info("Step 2/3 — Loading images and extracting embeddings …")
    X, y, label_map = load_embeddings(extractor)

    # Also persist embeddings store for future nearest-neighbour queries
    extractor.build_store(save=True)

    # ── Step 2 : Train classifier ─────────────────────────────────────────────
    logger.info("Step 3/3 — Training %s classifier …", args.clf.upper())
    run_gs = not args.no_cv

    if args.clf == "svm":
        pipeline = build_svm_pipeline(X, y, c=args.c,
                                      kernel=args.kernel, run_grid_search=run_gs)
    else:  # knn
        pipeline = build_knn_pipeline(X, y, n_neighbors=args.k,
                                      run_grid_search=run_gs)

    # ── Step 3 : Save ─────────────────────────────────────────────────────────
    save_model(pipeline, label_map, classifier=args.clf, out_path=args.out)

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(pipeline, X, y, label_map)

    print("✅  Training complete!")
    print(f"   Model     : {args.out}")
    print(f"   Persons   : {len(label_map)}  → {list(label_map.values())}")
    print(f"   Samples   : {len(X)}")
    print(f"   Classifier: {args.clf.upper()}\n")


if __name__ == "__main__":
    args = _parse_args()
    try:
        train(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
