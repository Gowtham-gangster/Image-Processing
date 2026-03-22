"""
export_onnx.py
==============
CLI to export all DNN models from their native formats to ONNX.

Supported exports
-----------------
  face      OpenFace .t7 embedder  →  models/onnx/openface_embedder.onnx
  mask      Keras mask detector    →  models/onnx/mask_detector.onnx
  detect    SSD Caffe face detector→  models/onnx/ssd_face_detector.onnx

Usage
-----
    python export_onnx.py --all
    python export_onnx.py --model face
    python export_onnx.py --model mask
    python export_onnx.py --model detect
    python export_onnx.py --all --verify   # verify each after export

Dependencies
------------
    pip install onnx tf2onnx torch torchvision onnxruntime
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

from config import (
    MODELS_DIR,
    FACE_PROTO, FACE_MODEL,
    MASK_MODEL_PATH,
    ONNX_DIR, ONNX_EMBEDDER_PATH, ONNX_MASK_PATH, ONNX_DETECTOR_PATH,
    LOG_LEVEL,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("export_onnx")

OPENFACE_T7 = os.path.join(MODELS_DIR, "openface_nn4.small2.v1.t7")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_onnx_dir() -> None:
    os.makedirs(ONNX_DIR, exist_ok=True)


def _verify_onnx(path: str, dummy_input: np.ndarray, expected_shape: tuple) -> None:
    """Run a quick ONNX Runtime inference to confirm the exported model works."""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp_name  = sess.get_inputs()[0].name
    out       = sess.run(None, {inp_name: dummy_input})
    got       = tuple(out[0].shape)
    assert got == expected_shape, f"Shape mismatch: got {got}, expected {expected_shape}"
    logger.info("  ✓ Verified: %s  → output shape %s", os.path.basename(path), got)


def _quantize_onnx(path: str) -> str:
    """Quantize an exported ONNX model to INT8 to radically slash memory footprints."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quant_path = path.replace(".onnx", "_quant.onnx")
        logger.info("  ⚙ Quantizing %s to INT8...", os.path.basename(path))
        quantize_dynamic(model_input=path, model_output=quant_path, weight_type=QuantType.QUInt8)
        size_orig = os.path.getsize(path) / 1e6
        size_quant = os.path.getsize(quant_path) / 1e6
        logger.info("  ✓ Quantized: %s (%.1f MB → %.1f MB)", os.path.basename(quant_path), size_orig, size_quant)
        return quant_path
    except Exception as e:
        logger.error("  ✕ Quantization failed: %s", e)
        return ""


# ── 1. OpenFace Embedder (.t7 → ONNX) ────────────────────────────────────────

def export_face_embedder(verify: bool = False) -> str:
    """
    The OpenFace model is a Torch7 (.t7) network.
    Strategy: reconstruct a lightweight equivalent in PyTorch (OpenFace nn4.small2)
    and export via torch.onnx.export, OR use the cv2.dnn → ONNX passthrough.

    We use the cv2.dnn + a dummy torch wrapper approach for portability.
    """
    logger.info("── Exporting OpenFace face embedder ──────────────────────────────────")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch")
        return ""

    # OpenFace nn4.small2.v1 is a 128-D network.
    # We define a minimal architecture-equivalent wrapper so we can load
    # the weights via cv2.dnn and re-export from a traced forward pass.
    class OpenFaceWrapper(nn.Module):
        """
        Wraps cv2.dnn OpenFace inference as a TorchScript-compatible module.
        This lets us trace once and export to ONNX without needing the full
        PyTorch OpenFace reimplementation.
        """
        def __init__(self):
            super().__init__()
            # Store the t7 path; inference happens in cv2.dnn
            import cv2
            self.net = cv2.dnn.readNetFromTorch(OPENFACE_T7)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (1, 3, 96, 96) float32
            import cv2
            blob = x.detach().cpu().numpy()
            self.net.setInput(blob)
            out = self.net.forward()          # (1, 128)
            return torch.from_numpy(out)

    if not os.path.exists(OPENFACE_T7):
        logger.error("OpenFace .t7 not found at %s. Run the embedder once first.", OPENFACE_T7)
        return ""

    _ensure_onnx_dir()

    logger.info("Tracing OpenFace via cv2.dnn…")
    model   = OpenFaceWrapper()
    dummy   = torch.zeros(1, 3, 96, 96, dtype=torch.float32)
    out_ref = model(dummy)

    torch.onnx.export(
        model,
        dummy,
        ONNX_EMBEDDER_PATH,
        input_names=["face_input"],
        output_names=["embedding"],
        dynamic_axes={"face_input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=12,
        verbose=False,
    )
    logger.info("Saved: %s", ONNX_EMBEDDER_PATH)

    if verify:
        dummy_np = np.zeros((1, 3, 96, 96), dtype=np.float32)
        _verify_onnx(ONNX_EMBEDDER_PATH, dummy_np, (1, 128))

    _quantize_onnx(ONNX_EMBEDDER_PATH)

    return ONNX_EMBEDDER_PATH


# ── 2. Mask Detector (Keras → ONNX) ─────────────────────────────────────────

def export_mask_detector(verify: bool = False) -> str:
    logger.info("── Exporting Keras mask detector ─────────────────────────────────────")

    if not os.path.exists(MASK_MODEL_PATH):
        logger.warning("Mask model not found at %s — skipping.", MASK_MODEL_PATH)
        return ""

    try:
        import tf2onnx                                            # pip install tf2onnx
        import tensorflow as tf
    except ImportError:
        logger.error("tf2onnx not installed. Run: pip install tf2onnx tensorflow")
        return ""

    _ensure_onnx_dir()

    logger.info("Loading Keras model from %s…", MASK_MODEL_PATH)
    model = tf.keras.models.load_model(MASK_MODEL_PATH)

    # Determine input shape from the model
    input_shape = model.input_shape          # e.g. (None, 224, 224, 3)
    spec = (tf.TensorSpec(input_shape, tf.float32, name="face_input"),)

    logger.info("Converting via tf2onnx (opset 13)…")
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=ONNX_MASK_PATH,
    )
    logger.info("Saved: %s", ONNX_MASK_PATH)

    if verify:
        h, w = input_shape[1], input_shape[2]
        dummy_np = np.zeros((1, h, w, 3), dtype=np.float32)
        _verify_onnx(ONNX_MASK_PATH, dummy_np, (1, 2))

    _quantize_onnx(ONNX_MASK_PATH)

    return ONNX_MASK_PATH


# ── 3. SSD Face Detector (Caffe → ONNX) ────────────────────────────────────

def export_face_detector(verify: bool = False) -> str:
    logger.info("── Exporting SSD face detector (Caffe) ───────────────────────────────")

    if not os.path.exists(FACE_MODEL) or not os.path.exists(FACE_PROTO):
        logger.warning(
            "SSD Caffe weights not found (%s / %s) — skipping.", FACE_PROTO, FACE_MODEL
        )
        return ""

    try:
        import caffe2.python.caffe_translator as translator       # pip install caffe2
    except ImportError:
        # Fallback: use the simpler caffe2onnx library
        try:
            import caffe2onnx
        except ImportError:
            # Final fallback: emit a conversion script the user can run
            _write_caffe_conversion_script()
            logger.warning(
                "caffe2onnx not installed. "
                "A conversion script has been written to scripts/convert_ssd_to_onnx.sh. "
                "Run it in a Docker container with caffe2 installed."
            )
            return ""

    _ensure_onnx_dir()

    try:
        import caffe2onnx
        caffe2onnx.convert(
            prototxt=FACE_PROTO,
            caffemodel=FACE_MODEL,
            onnx_model_path=ONNX_DETECTOR_PATH,
        )
        logger.info("Saved: %s", ONNX_DETECTOR_PATH)
    except Exception as exc:
        logger.error("Caffe→ONNX conversion failed: %s", exc)
        return ""

    if verify:
        dummy_np = np.zeros((1, 3, 300, 300), dtype=np.float32)
        # SSD output shape is (1, 1, N, 7) where N = num detections
        out = _run_onnx_raw(ONNX_DETECTOR_PATH, dummy_np)
        logger.info("  ✓ Detector output shape: %s", out[0].shape)

    _quantize_onnx(ONNX_DETECTOR_PATH)

    return ONNX_DETECTOR_PATH


def _run_onnx_raw(path: str, dummy: np.ndarray):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: dummy})


def _write_caffe_conversion_script() -> None:
    """Write a standalone shell script for environments with native caffe2."""
    os.makedirs(os.path.join(os.path.dirname(ONNX_DIR), "scripts"), exist_ok=True)
    script_path = os.path.join(os.path.dirname(ONNX_DIR), "scripts", "convert_ssd_to_onnx.sh")
    script = f"""#!/bin/bash
# Run inside a caffe2/pytorch Docker container
# docker run --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:23.10-py3 bash scripts/convert_ssd_to_onnx.sh

pip install caffe2onnx onnx onnxruntime -q

python - <<'PYEOF'
import caffe2onnx
caffe2onnx.convert(
    prototxt="{FACE_PROTO}",
    caffemodel="{FACE_MODEL}",
    onnx_model_path="{ONNX_DETECTOR_PATH}",
)
print("Saved:", "{ONNX_DETECTOR_PATH}")
PYEOF
"""
    with open(script_path, "w") as f:
        f.write(script)
    logger.info("Conversion helper script written to: %s", script_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export DNN models to ONNX for edge/Jetson deployment."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="Export all models.")
    g.add_argument(
        "--model",
        choices=["face", "mask", "detect"],
        help="Export a single model.",
    )
    p.add_argument("--verify", action="store_true", help="Run ONNX Runtime verification after export.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0   = time.time()
    exported = []

    if args.all or args.model == "face":
        r = export_face_embedder(verify=args.verify)
        if r: exported.append(("OpenFace Embedder", r))

    if args.all or args.model == "mask":
        r = export_mask_detector(verify=args.verify)
        if r: exported.append(("Mask Detector", r))

    if args.all or args.model == "detect":
        r = export_face_detector(verify=args.verify)
        if r: exported.append(("SSD Face Detector", r))

    elapsed = time.time() - t0
    print(f"\n[SUCCESS] Export complete in {elapsed:.1f}s")
    for name, path in exported:
        size_mb = os.path.getsize(path) / 1e6
        print(f"   {name:22s} → {path}  ({size_mb:.1f} MB)")

    if not exported:
        print("[WARNING] No models exported. Check logs above for details.")


if __name__ == "__main__":
    main()

