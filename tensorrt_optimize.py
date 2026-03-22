"""
tensorrt_optimize.py
====================
Compile ONNX models into TensorRT .engine files for Jetson Nano deployment.

IMPORTANT — this script must be run ON THE JETSON NANO (or any CUDA host with
TensorRT installed via NVIDIA JetPack / TensorRT SDK).
It will NOT work on a standard Windows/macOS development machine.

TensorRT engines are device-specific: an engine built on the Jetson Nano
CANNOT be transferred to a different GPU or CPU.

Features
--------
- FP32 and FP16 (half-precision) precision modes.
- Dynamic batch support (batch=1 fixed for real-time streaming).
- Automatic workspace size management.
- Engine persistence: built once, loaded every subsequent run.

Usage (on Jetson Nano)
---------------------
    python tensorrt_optimize.py --all --fp16        # recommended
    python tensorrt_optimize.py --onnx models/onnx/openface_embedder.onnx --fp16
    python tensorrt_optimize.py --onnx models/onnx/mask_detector.onnx
    python tensorrt_optimize.py --list              # show available ONNX files

Dependencies (Jetson JetPack 5.x)
-----------------------------------
    pip install onnxruntime-gpu tensorrt pycuda

Docker alternative (desktop CUDA host)
---------------------------------------
    docker run --rm --gpus all -v $(pwd):/workspace \\
        nvcr.io/nvidia/tensorrt:23.10-py3 \\
        python /workspace/tensorrt_optimize.py --all --fp16
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np

from config import (
    ONNX_DIR, TRT_DIR,
    ONNX_EMBEDDER_PATH, ONNX_MASK_PATH, ONNX_DETECTOR_PATH,
    TRT_EMBEDDER_ENGINE, TRT_MASK_ENGINE, TRT_DETECTOR_ENGINE,
    LOG_LEVEL,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("tensorrt_optimize")

# Default max workspace: 256 MB (1 << 28 bytes)
DEFAULT_WORKSPACE_MB = 256

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "embedder": {
        "onnx":   ONNX_EMBEDDER_PATH,
        "engine": TRT_EMBEDDER_ENGINE,
        "input_shape": (1, 3, 96, 96),       # NCHW
        "label": "OpenFace Embedder",
    },
    "mask": {
        "onnx":   ONNX_MASK_PATH,
        "engine": TRT_MASK_ENGINE,
        "input_shape": (1, 224, 224, 3),     # NHWC (Keras-style)
        "label": "Mask Detector",
    },
    "detector": {
        "onnx":   ONNX_DETECTOR_PATH,
        "engine": TRT_DETECTOR_ENGINE,
        "input_shape": (1, 3, 300, 300),     # NCHW
        "label": "SSD Face Detector",
    },
}


# ── TensorRT builder ─────────────────────────────────────────────────────────

class TRTLogger:
    """Minimal TensorRT ILogger adapter."""
    def __init__(self):
        try:
            import tensorrt as trt
            self._trt_logger = trt.Logger(trt.Logger.WARNING)
        except ImportError:
            self._trt_logger = None

    def __getattr__(self, item):
        return getattr(self._trt_logger, item)


def build_engine(
    onnx_path: str,
    engine_path: str,
    input_shape: tuple,
    fp16: bool = True,
    workspace_mb: int = DEFAULT_WORKSPACE_MB,
) -> bool:
    """
    Build a TensorRT engine from an ONNX model.

    Parameters
    ----------
    onnx_path    : Path to source ONNX file.
    engine_path  : Destination .engine file path.
    input_shape  : Static input shape (N, C, H, W) or (N, H, W, C).
    fp16         : Enable FP16 (half-precision) — halves memory, ~2× faster.
    workspace_mb : Maximum GPU memory workspace in MB.

    Returns
    -------
    True on success, False on failure.
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error(
            "TensorRT Python bindings not found.\n"
            "Install via NVIDIA JetPack or the TensorRT Python wheel:\n"
            "  pip install tensorrt\n"
            "Or run in the official TensorRT Docker image."
        )
        return False

    if not os.path.exists(onnx_path):
        logger.error("ONNX model not found: %s", onnx_path)
        logger.error("Run  python export_onnx.py --all  first.")
        return False

    os.makedirs(TRT_DIR, exist_ok=True)
    trt_logger = trt.Logger(trt.Logger.WARNING)

    logger.info("Building TRT engine from %s …", os.path.basename(onnx_path))
    t0 = time.time()

    with trt.Builder(trt_logger) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, trt_logger) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 precision enabled.")
        elif fp16:
            logger.warning("FP16 requested but not supported on this device — using FP32.")

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("Parser error %d: %s", i, parser.get_error(i))
                return False

        # Optim profile for dynamic dims
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        profile.set_shape(input_name, input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)

        # Build and serialize
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            logger.error("Engine build failed.")
            return False

        with open(engine_path, "wb") as f:
            f.write(serialized)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(engine_path) / 1e6
    logger.info("Engine saved: %s  (%.1f MB, built in %.1fs)", engine_path, size_mb, elapsed)
    return True


def load_engine(engine_path: str):
    """
    Load a previously built TensorRT engine for inference.

    Returns a trt.IExecutionContext or None if unavailable.
    """
    try:
        import tensorrt as trt
        import pycuda.autoinit                                    # noqa: F401
    except ImportError:
        logger.error("TensorRT / PyCUDA not installed.")
        return None

    if not os.path.exists(engine_path):
        logger.error("Engine not found: %s. Run tensorrt_optimize.py first.", engine_path)
        return None

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime    = trt.Runtime(trt_logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx    = engine.create_execution_context()
    logger.info("TRT engine loaded: %s", os.path.basename(engine_path))
    return ctx


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compile ONNX models to TensorRT .engine files."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all",  action="store_true", help="Compile all three models.")
    g.add_argument("--onnx", metavar="PATH",      help="Compile a specific ONNX file.")
    g.add_argument("--list", action="store_true", help="List available ONNX models.")
    p.add_argument("--fp16",  action="store_true", default=True, help="Enable FP16 (default: on).")
    p.add_argument("--fp32",  action="store_true",               help="Force FP32 precision.")
    p.add_argument("--workspace", type=int, default=DEFAULT_WORKSPACE_MB,
                   help=f"GPU workspace in MB (default: {DEFAULT_WORKSPACE_MB}).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fp16 = args.fp16 and not args.fp32

    if args.list:
        print("\nAvailable ONNX models:")
        for key, m in MODELS.items():
            exists = "✓" if os.path.exists(m["onnx"]) else "✗ (not exported yet)"
            print(f"  [{key:10s}] {m['label']:25s} {m['onnx']}  {exists}")
        return

    results: list[tuple[str, bool]] = []

    if args.all:
        for key, m in MODELS.items():
            ok = build_engine(
                m["onnx"], m["engine"], m["input_shape"],
                fp16=fp16, workspace_mb=args.workspace,
            )
            results.append((m["label"], ok))

    elif args.onnx:
        # Try to match against known registry for input shape
        matched = next(
            (m for m in MODELS.values() if os.path.abspath(m["onnx"]) == os.path.abspath(args.onnx)),
            None,
        )
        engine_path = args.onnx.replace(".onnx", ".engine").replace(
            os.path.basename(ONNX_DIR), os.path.basename(TRT_DIR)
        )
        if matched:
            shape = matched["input_shape"]
        else:
            # Accept 4-D input shape from user
            logger.warning("Model not in registry — defaulting input shape to (1,3,224,224).")
            shape = (1, 3, 224, 224)

        ok = build_engine(args.onnx, engine_path, shape, fp16=fp16, workspace_mb=args.workspace)
        results.append((os.path.basename(args.onnx), ok))

    print("\n──────────────────────────────────────────────")
    print(f"  Precision: {'FP16' if fp16 else 'FP32'}")
    for label, ok in results:
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {label}")
    print("──────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
