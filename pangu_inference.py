"""
pangu_inference.py — Iterative 7-day Pangu-Weather inference chain.

Step schedule:
  Steps 0-7  : pangu_weather_6.onnx  (6-hour model)  → hours 6,12,18,24,30,36,42,48
  Steps 8-12 : pangu_weather_24.onnx (24-hour model)  → hours 72,96,120,144,168

Each step output is cached as output_data/weekly_cache/step_NN.npz.
A manifest.json records the MD5 hash of the input files so the cache is
invalidated automatically when the input data changes.

Public API
----------
    run_weekly_inference(force_refresh=False) -> list[dict]
        Returns a list of 13 step dicts, each containing:
            hour (int), output_upper (ndarray), output_surface (ndarray)

    count_cached_steps() -> int
        Returns the number of cached steps valid for the current input data.
"""

import hashlib
import json
import os
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the app works from any cwd
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
_PARENT     = os.path.join(_HERE, "..")

MODEL_6H    = os.path.join(_PARENT, "pangu_weather_6.onnx")
MODEL_24H   = os.path.join(_PARENT, "pangu_weather_24.onnx")
INPUT_DIR   = os.path.join(_PARENT, "input_data")
CACHE_DIR   = os.path.join(_PARENT, "output_data", "weekly_cache")
MANIFEST    = os.path.join(CACHE_DIR, "manifest.json")

SURFACE_NPY = os.path.join(INPUT_DIR, "input_surface.npy")
UPPER_NPY   = os.path.join(INPUT_DIR, "input_upper.npy")

# ---------------------------------------------------------------------------
# Step schedule: (step_index, model_key, lead_hours)
# ---------------------------------------------------------------------------
_STEPS = (
    [(i, "6h",  (i + 1) * 6)        for i in range(8)]     # 6h … 48h
    + [(i, "24h", 48 + (i - 7) * 24) for i in range(8, 13)] # 72h … 168h
)


# ---------------------------------------------------------------------------
# Lazy ONNX session loader — loads each model once per process
# ---------------------------------------------------------------------------
_sessions: dict = {}


def _get_session(model_key: str):
    if model_key in _sessions:
        return _sessions[model_key]

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime-gpu") from exc

    model_path = MODEL_6H if model_key == "6h" else MODEL_24H
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pangu-Weather model not found: {model_path}")

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    print(f"[pangu_inference] Loading {model_key} model with {providers[0]}")
    sess = ort.InferenceSession(model_path, providers=providers)
    _sessions[model_key] = sess
    return sess


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _step_cache_path(step_idx: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"step_{step_idx:02d}.npz")


def _input_hash() -> str:
    """MD5 of input_surface.npy + input_upper.npy concatenated."""
    h = hashlib.md5()
    for path in (SURFACE_NPY, UPPER_NPY):
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def _load_manifest() -> dict:
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            return json.load(f)
    return {}


def _save_manifest(data: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(data, f, indent=2)


def _cache_valid(step_idx: int, current_hash: str, manifest: dict) -> bool:
    path = _step_cache_path(step_idx)
    return (
        os.path.exists(path)
        and manifest.get("input_hash") == current_hash
    )


# ---------------------------------------------------------------------------
# Single-step inference
# ---------------------------------------------------------------------------

def _run_step(
    model_key: str,
    input_upper: np.ndarray,
    input_surface: np.ndarray,
) -> tuple:
    """Run one Pangu-Weather step; returns (output_upper, output_surface)."""
    sess = _get_session(model_key)
    outputs = sess.run(
        ["output", "output_surface"],
        {"input": input_upper, "input_surface": input_surface},
    )
    return outputs[0], outputs[1]   # (5,13,721,1440), (4,721,1440)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_cached_steps() -> int:
    """Return the number of valid cached weekly steps for the current inputs."""
    if not os.path.exists(SURFACE_NPY) or not os.path.exists(UPPER_NPY):
        return 0
    current_hash = _input_hash()
    manifest = _load_manifest()
    return sum(
        1 for i, _, _ in _STEPS
        if _cache_valid(i, current_hash, manifest)
    )


def run_weekly_inference(force_refresh: bool = False) -> list:
    """
    Run the full 13-step iterative Pangu-Weather chain.

    Returns a list of 13 dicts:
        {"step": int, "model": str, "hour": int,
         "output_upper": ndarray, "output_surface": ndarray}

    Steps are cached individually; re-running only recomputes what is missing
    (or everything if force_refresh=True or input files changed).
    """
    # Verify input files exist
    missing = [p for p in (SURFACE_NPY, UPPER_NPY) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Input files not found for weekly inference: {missing}\n"
            "Run fetch_era5.py or ensure input_data/ is populated."
        )

    current_hash = _input_hash()
    manifest     = _load_manifest()
    hash_changed = manifest.get("input_hash") != current_hash

    if force_refresh or hash_changed:
        print(f"[pangu_inference] {'force_refresh=True' if force_refresh else 'Input data changed'} — clearing weekly cache.")
        # Remove all cached steps
        for i, _, _ in _STEPS:
            p = _step_cache_path(i)
            if os.path.exists(p):
                os.remove(p)
        manifest = {}

    # Load initial inputs
    cur_surface = np.load(SURFACE_NPY).astype(np.float32)
    cur_upper   = np.load(UPPER_NPY).astype(np.float32)

    results = []
    step_loaded_from_cache = False

    for step_idx, model_key, lead_hour in _STEPS:
        cache_path = _step_cache_path(step_idx)

        if _cache_valid(step_idx, current_hash, manifest):
            # Load from cache — also need to update cur_surface/cur_upper
            # for subsequent uncached steps to chain correctly
            data = np.load(cache_path)
            out_upper   = data["output_upper"]
            out_surface = data["output_surface"]
            step_loaded_from_cache = True
            print(f"[pangu_inference] Step {step_idx:02d} ({lead_hour:3d}h): loaded from cache")
        else:
            # If we were loading from cache and hit an uncached step, we need
            # cur_surface/cur_upper from the previous step's output.
            # At this point cur_surface/cur_upper already hold the correct
            # state (either initial or last computed step's output), so we
            # can run inference directly.
            print(f"[pangu_inference] Step {step_idx:02d} ({lead_hour:3d}h): running {model_key} model…")
            out_upper, out_surface = _run_step(model_key, cur_upper, cur_surface)

            # Cache this step
            np.savez_compressed(
                cache_path,
                output_upper=out_upper,
                output_surface=out_surface,
            )
            print(f"[pangu_inference] Step {step_idx:02d}: cached → {cache_path}")

        # Advance the rolling state for the next step
        cur_upper   = out_upper
        cur_surface = out_surface

        results.append({
            "step":           step_idx,
            "model":          model_key,
            "hour":           lead_hour,
            "output_upper":   out_upper,
            "output_surface": out_surface,
        })

    # Save manifest with current hash
    _save_manifest({"input_hash": current_hash, "steps": len(_STEPS)})
    print(f"[pangu_inference] Weekly inference complete — {len(results)} steps.")
    return results
