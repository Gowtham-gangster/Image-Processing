"""
embeddings.py
=============
Deep feature extraction for the Mask-Aware Hybrid Person Identification System.

Model
-----
**MobileNetV2** (pretrained on ImageNet, top removed) is used as the backbone.
It is lightweight (~14 MB), runs on CPU, and produces a 1 280-dimensional
L2-normalised embedding vector per face image — well suited for small datasets.

A PyTorch fallback (MobileNet_V2) is also included; it activates automatically
if TensorFlow is unavailable.

Outputs
-------
embeddings/
    person1.npz        ← embeddings + file paths for person1
    person2.npz        ← … for person2
    …
    embeddings_store.npz   ← merged store: all embeddings, labels, paths

Usage (module)
--------------
    from embeddings import EmbeddingExtractor

    ext = EmbeddingExtractor()
    vec = ext.extract(bgr_image)          # shape (1280,) float32, L2-normalised

    # Build and save the full training embedding store
    ext.build_store()

Usage (CLI)
-----------
    python embeddings.py                  # extract and save all training embeddings
    python embeddings.py --split test     # extract test embeddings using saved store
    python embeddings.py --backend torch  # force PyTorch backend
    python embeddings.py --no-save        # extract but do not write files
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import TRAIN_DIR, TEST_DIR, LOG_LEVEL

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EMBED_DIR        = os.path.join(os.path.dirname(__file__), "embeddings")
STORE_PATH       = os.path.join(EMBED_DIR, "embeddings_store.npz")
INPUT_SIZE_TF    = (224, 224)    # MobileNetV2 input
INPUT_SIZE_TORCH = (224, 224)
EMBED_DIM        = 1280          # MobileNetV2 top-pooled feature dimension
IMG_EXTS         = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

os.makedirs(EMBED_DIR, exist_ok=True)


# ── Backend loaders ───────────────────────────────────────────────────────────

def _load_tf_model():
    """Return a tf.keras MobileNetV2 feature extractor (no top)."""
    import tensorflow as tf                                   # type: ignore
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*INPUT_SIZE_TF, 3),
        include_top=False,
        pooling="avg",                 # → (batch, 1280)
        weights="imagenet",
    )
    base.trainable = False
    logger.info("TF backend: MobileNetV2 loaded (input=%s, output_dim=%d).",
                INPUT_SIZE_TF, EMBED_DIM)
    return base


def _load_torch_model():
    """Return a torchvision MobileNetV2 feature extractor (no classifier)."""
    import torch                                              # type: ignore
    import torchvision.models as tvm                         # type: ignore

    model = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()   # remove final classifier head
    model.eval()
    logger.info("PyTorch backend: MobileNetV2 loaded (output_dim=1280).")
    return model


# ── EmbeddingExtractor ────────────────────────────────────────────────────────

class EmbeddingExtractor:
    """
    Extracts L2-normalised 1 280-D feature embeddings from BGR face images
    using a pretrained MobileNetV2 backbone.

    Parameters
    ----------
    backend : str
        ``'auto'`` – try TensorFlow first, then PyTorch.
        ``'tf'``   – force TensorFlow / Keras.
        ``'torch'``– force PyTorch / torchvision.
    train_dir : str
        Root of the training split (one sub-folder per person).
    embed_dir : str
        Directory where .npz files are written.
    """

    BACKENDS = ("auto", "tf", "torch")

    def __init__(
        self,
        backend: str = "auto",
        train_dir: str = TRAIN_DIR,
        embed_dir: str = EMBED_DIR,
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}")

        self.train_dir = train_dir
        self.embed_dir = embed_dir
        self._backend  = "none"
        self._model    = None

        self._backend, self._model = self._load_model(backend)
        logger.info("EmbeddingExtractor ready | backend=%s", self._backend)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Extract a feature embedding from a single BGR image.

        Parameters
        ----------
        bgr_image : np.ndarray
            BGR image of any size (will be resized internally).

        Returns
        -------
        np.ndarray  shape (1280,)  dtype float32  L2-normalised.
        """
        if self._backend == "tf":
            return self._extract_tf(bgr_image)
        return self._extract_torch(bgr_image)

    def extract_batch(self, bgr_images: list[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for a list of BGR images.

        Returns
        -------
        np.ndarray  shape (N, 1280)  dtype float32  L2-normalised.
        """
        return np.vstack([self.extract(img) for img in bgr_images])

    def build_store(
        self,
        save: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Walk ``train_dir``, extract embeddings for every image, and
        optionally persist them.

        Per-person files
        ----------------
        ``embeddings/<person_id>.npz``
            ``embeddings`` : shape (N_person, 1280)
            ``paths``      : array of image-file paths

        Merged store
        ------------
        ``embeddings/embeddings_store.npz``
            ``embeddings`` : shape (N_total, 1280)
            ``labels``     : shape (N_total,)  int32  (numeric label)
            ``person_ids`` : shape (N_total,)  str    (person folder name)
            ``paths``      : shape (N_total,)  str    (absolute image path)
            ``label_map``  : shape (M, 2)      str    [[label_int, person_id], …]

        Returns
        -------
        dict with keys 'embeddings', 'labels', 'person_ids', 'paths', 'label_map'.
        """
        if not os.path.isdir(self.train_dir):
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")

        person_ids = sorted(
            p for p in os.listdir(self.train_dir)
            if os.path.isdir(os.path.join(self.train_dir, p))
        )

        if not person_ids:
            raise ValueError(
                f"No person sub-folders found in '{self.train_dir}'."
            )

        all_embeddings: list[np.ndarray] = []
        all_labels:     list[int]        = []
        all_person_ids: list[str]        = []
        all_paths:      list[str]        = []
        label_map:      dict[int, str]   = {}

        for label_int, pid in enumerate(person_ids):
            label_map[label_int] = pid
            person_dir = os.path.join(self.train_dir, pid)
            img_paths  = self._list_images(person_dir)

            if not img_paths:
                logger.warning("No images in '%s' — skipping.", person_dir)
                continue

            person_embeds: list[np.ndarray] = []
            person_paths:  list[str]        = []

            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning("Cannot read '%s' — skipping.", img_path)
                    continue

                emb = self.extract(img)
                person_embeds.append(emb)
                person_paths.append(img_path)

                all_embeddings.append(emb)
                all_labels.append(label_int)
                all_person_ids.append(pid)
                all_paths.append(img_path)

            if not person_embeds:
                continue

            person_arr = np.vstack(person_embeds)  # (N_person, 1280)
            logger.info(
                "  [%s] %d embeddings extracted | shape %s",
                pid, len(person_embeds), person_arr.shape,
            )

            if save:
                out_path = os.path.join(self.embed_dir, f"{pid}.npz")
                np.savez_compressed(
                    out_path,
                    embeddings=person_arr,
                    paths=np.array(person_paths),
                )
                logger.info("  Saved → %s", out_path)

        if not all_embeddings:
            logger.error("No embeddings extracted. Check training images.")
            return {}

        store = {
            "embeddings": np.vstack(all_embeddings).astype(np.float32),
            "labels":     np.array(all_labels, dtype=np.int32),
            "person_ids": np.array(all_person_ids),
            "paths":      np.array(all_paths),
            "label_map":  np.array([[str(k), v] for k, v in label_map.items()]),
        }

        if save:
            np.savez_compressed(STORE_PATH, **store)
            logger.info("Merged store saved → %s  (shape %s)",
                        STORE_PATH, store["embeddings"].shape)

        return store

    # ── Store I/O ─────────────────────────────────────────────────────────────

    @staticmethod
    def load_store(path: str = STORE_PATH) -> dict[str, np.ndarray]:
        """
        Load a saved embedding store from disk.

        Returns
        -------
        dict with keys: embeddings, labels, person_ids, paths, label_map.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Embedding store not found at '{path}'. "
                "Run  python embeddings.py  first."
            )
        data = np.load(path, allow_pickle=True)
        store = {k: data[k] for k in data.files}
        logger.info(
            "Embedding store loaded: %d samples, %d dims.",
            store["embeddings"].shape[0], store["embeddings"].shape[1],
        )
        return store

    @staticmethod
    def nearest_neighbour(
        query: np.ndarray,
        store: dict[str, np.ndarray],
        top_k: int = 1,
    ) -> list[dict]:
        """
        Find the closest embeddings in the store using cosine similarity.

        Parameters
        ----------
        query : np.ndarray
            shape (1280,)  L2-normalised query embedding.
        store : dict
            The dict returned by :meth:`load_store` or :meth:`build_store`.
        top_k : int
            Number of closest matches to return.

        Returns
        -------
        List of dicts: [{'person_id', 'label', 'similarity', 'path'}, …]
        """
        db_embeds  = store["embeddings"]          # (N, 1280)
        sims       = db_embeds @ query            # cosine sim (both L2-normed)
        top_idx    = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_idx:
            results.append({
                "person_id":  str(store["person_ids"][idx]),
                "label":      int(store["labels"][idx]),
                "similarity": float(sims[idx]),
                "path":       str(store["paths"][idx]),
            })
        return results

    @property
    def backend(self) -> str:
        return self._backend

    # ── Preprocessing ─────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess_tf(bgr: np.ndarray) -> "np.ndarray":
        """BGR → RGB → resize → MobileNetV2 preprocess → (1, 224, 224, 3)."""
        import tensorflow as tf                               # type: ignore
        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, INPUT_SIZE_TF)
        tensor  = tf.keras.applications.mobilenet_v2.preprocess_input(
            resized.astype(np.float32)
        )
        return np.expand_dims(tensor, axis=0)                # (1, 224, 224, 3)

    @staticmethod
    def _preprocess_torch(bgr: np.ndarray) -> "torch.Tensor":
        """BGR → RGB → resize → normalise → (1, 3, 224, 224) torch.Tensor."""
        import torch                                          # type: ignore
        from torchvision import transforms                   # type: ignore

        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, INPUT_SIZE_TORCH)
        tf_ops  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
        return tf_ops(resized).unsqueeze(0)                  # (1, 3, 224, 224)

    # ── Extraction back-ends ──────────────────────────────────────────────────

    def _extract_tf(self, bgr: np.ndarray) -> np.ndarray:
        tensor = self._preprocess_tf(bgr)
        emb    = self._model.predict(tensor, verbose=0)[0]   # (1280,)
        return self._l2_norm(emb.astype(np.float32))

    def _extract_torch(self, bgr: np.ndarray) -> np.ndarray:
        import torch                                          # type: ignore
        tensor = self._preprocess_torch(bgr)
        with torch.no_grad():
            out = self._model(tensor)                        # (1, 1280)
        emb = out.squeeze().cpu().numpy().astype(np.float32)
        return self._l2_norm(emb)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _l2_norm(vec: np.ndarray) -> np.ndarray:
        """L2-normalise a 1-D vector."""
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    @staticmethod
    def _list_images(directory: str) -> list[str]:
        return sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if Path(f).suffix.lower() in IMG_EXTS
        )

    def _load_model(self, backend: str):
        """Try to load the requested backend; fall back gracefully."""
        if backend in ("auto", "tf"):
            try:
                return "tf", _load_tf_model()
            except Exception as exc:
                if backend == "tf":
                    raise RuntimeError(f"TensorFlow backend failed: {exc}") from exc
                logger.info("TF unavailable (%s) — trying PyTorch.", exc)

        if backend in ("auto", "torch"):
            try:
                return "torch", _load_torch_model()
            except Exception as exc:
                raise RuntimeError(
                    f"Both TF and PyTorch backends failed. "
                    f"Install tensorflow>=2.13 or torch+torchvision. ({exc})"
                ) from exc

        raise ValueError(f"Unknown backend: {backend}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and store MobileNetV2 embeddings from the dataset."
    )
    parser.add_argument(
        "--split", choices=["train", "test"], default="train",
        help="Dataset split to process (default: train).",
    )
    parser.add_argument(
        "--backend", choices=["auto", "tf", "torch"], default="auto",
        help="Deep learning backend (default: auto).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Extract embeddings but do NOT write .npz files.",
    )
    parser.add_argument(
        "--query", default=None, metavar="IMAGE_PATH",
        help="Query a single image against the saved store.",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of nearest neighbours to return for --query (default: 3).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ── Query mode ────────────────────────────────────────────────────────────
    if args.query:
        img = cv2.imread(args.query)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {args.query}")

        ext   = EmbeddingExtractor(backend=args.backend)
        store = EmbeddingExtractor.load_store()
        emb   = ext.extract(img)
        hits  = EmbeddingExtractor.nearest_neighbour(emb, store, top_k=args.top_k)

        print(f"\nQuery: {args.query}")
        print(f"{'Rank':<5} {'Person ID':<20} {'Similarity':>10}  Path")
        print("─" * 70)
        for rank, hit in enumerate(hits, 1):
            print(
                f"{rank:<5} {hit['person_id']:<20} "
                f"{hit['similarity']:>10.4f}  {hit['path']}"
            )
        print()

    # ── Build store mode ──────────────────────────────────────────────────────
    else:
        src_dir = TRAIN_DIR if args.split == "train" else TEST_DIR
        ext     = EmbeddingExtractor(backend=args.backend, train_dir=src_dir)
        store   = ext.build_store(save=not args.no_save)

        if store:
            print(f"\n✅  Embeddings extracted  ({args.split} split)")
            print(f"   Samples    : {store['embeddings'].shape[0]}")
            print(f"   Dimensions : {store['embeddings'].shape[1]}")
            print(f"   Persons    : {len(np.unique(store['labels']))}")
            print(f"   Saved to   : {EMBED_DIR}\n")
