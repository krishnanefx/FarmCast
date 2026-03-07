"""
fetch_era5.py — Download real ERA5 reanalysis data from the Copernicus CDS
and convert it into the exact numpy format required by Pangu-Weather's
1-hour ONNX model.

Prerequisites
-------------
1. Register at https://cds.climate.copernicus.eu and accept the terms for:
     - reanalysis-era5-single-levels
     - reanalysis-era5-pressure-levels
2. Get your Personal Access Token from https://cds.climate.copernicus.eu/profile
3. Create ~/.cdsapirc containing:
     url: https://cds.climate.copernicus.eu/api
     key: <PERSONAL-ACCESS-TOKEN>

Usage
-----
    python fetch_era5.py

Output files (saved one directory above the repo root, matching data_loader.py defaults):
    ../input_data/input_surface.npy   (4, 721, 1440)  float32
    ../input_data/input_upper.npy     (5, 13, 721, 1440) float32
    ../input_data/era5_meta.json
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# Target date — ERA5 has a ~5 day reanalysis lag; use 7-day buffer to be safe
# ---------------------------------------------------------------------------
target_dt = datetime.utcnow() - timedelta(days=7)
date_str  = target_dt.strftime("%Y-%m-%d")
time_str  = "12:00"

print(f"[ERA5] Fetching ERA5 data for {date_str} at {time_str} UTC")

# ---------------------------------------------------------------------------
# Output paths — one directory above the repo root (matches data_loader.py)
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_HERE, "..", "input_data")
SURFACE_NPY = os.path.join(OUTPUT_DIR, "input_surface.npy")
UPPER_NPY   = os.path.join(OUTPUT_DIR, "input_upper.npy")
META_JSON   = os.path.join(OUTPUT_DIR, "era5_meta.json")
SURFACE_GRIB = "surface_raw.grib"
UPPER_GRIB   = "upper_raw.grib"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 — Initialise CDS client
# ---------------------------------------------------------------------------
try:
    import cdsapi
except ImportError:
    print("[ERA5] ERROR: cdsapi is not installed. Run: pip install cdsapi")
    sys.exit(1)

cdsrc = os.path.expanduser("~/.cdsapirc")
if not os.path.exists(cdsrc):
    print(
        "[ERA5] ERROR: ~/.cdsapirc not found.\n"
        "  1. Register at https://cds.climate.copernicus.eu\n"
        "  2. Get your token from https://cds.climate.copernicus.eu/profile\n"
        "  3. Create ~/.cdsapirc containing:\n"
        "       url: https://cds.climate.copernicus.eu/api\n"
        "       key: <PERSONAL-ACCESS-TOKEN>"
    )
    sys.exit(1)

c = cdsapi.Client()


def _retrieve_with_retry(dataset: str, request: dict, target: str) -> None:
    """Retrieve from CDS, retrying once after 60 s on timeout."""
    for attempt in range(2):
        try:
            c.retrieve(dataset, request, target)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == 0 and "timeout" in str(exc).lower():
                print(
                    "[ERA5] CDS request queued — this can take a few minutes, please wait. "
                    "Retrying in 60 s…"
                )
                time.sleep(60)
            else:
                print(f"[ERA5] ERROR during CDS download: {exc}")
                sys.exit(1)


# ---------------------------------------------------------------------------
# Step 2 — Fetch surface variables → input_surface.npy
# Order: [0] MSLP, [1] U10, [2] V10, [3] T2M
# ---------------------------------------------------------------------------
print("[ERA5] Requesting surface variables…")
_retrieve_with_retry(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "mean_sea_level_pressure",   # MSLP → index 0
            "10m_u_component_of_wind",   # U10  → index 1
            "10m_v_component_of_wind",   # V10  → index 2
            "2m_temperature",            # T2M  → index 3
        ],
        "date": date_str,
        "time": time_str,
        "format": "grib",
        "grid": [0.25, 0.25],
    },
    SURFACE_GRIB,
)
print(f"[ERA5] Surface grib downloaded → {SURFACE_GRIB}")

# ---------------------------------------------------------------------------
# Step 3 — Fetch upper-air variables → input_upper.npy
# Order: [0] Z, [1] Q, [2] T, [3] U, [4] V  ×  13 pressure levels
# ---------------------------------------------------------------------------
print("[ERA5] Requesting upper-air variables (13 levels)…")
_retrieve_with_retry(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "geopotential",          # Z → index 0  (m²/s², no conversion needed)
            "specific_humidity",     # Q → index 1
            "temperature",           # T → index 2
            "u_component_of_wind",   # U → index 3
            "v_component_of_wind",   # V → index 4
        ],
        "pressure_level": [
            "1000", "925", "850", "700", "600",
            "500",  "400", "300", "250", "200",
            "150",  "100", "50",          # exactly 13 levels, top-down order
        ],
        "date": date_str,
        "time": time_str,
        "format": "grib",
        "grid": [0.25, 0.25],
    },
    UPPER_GRIB,
)
print(f"[ERA5] Upper-air grib downloaded → {UPPER_GRIB}")

# ---------------------------------------------------------------------------
# Step 4 — Convert grib to numpy arrays
# ERA5 latitude: 90 → -90 N  (top-to-bottom, matches Pangu grid exactly)
# ERA5 longitude: 0 → 359.75 E (matches Pangu grid exactly)
# ---------------------------------------------------------------------------
try:
    import xarray as xr
except ImportError:
    print("[ERA5] ERROR: xarray is not installed. Run: pip install xarray cfgrib")
    sys.exit(1)

print("[ERA5] Converting surface grib to numpy…")
ds_surface = xr.open_dataset(
    SURFACE_GRIB, engine="cfgrib",
    backend_kwargs={"indexpath": ""},
)

mslp = ds_surface["msl"].values.astype(np.float32)   # (721, 1440)
u10  = ds_surface["u10"].values.astype(np.float32)
v10  = ds_surface["v10"].values.astype(np.float32)
t2m  = ds_surface["t2m"].values.astype(np.float32)
ds_surface.close()

input_surface = np.stack([mslp, u10, v10, t2m], axis=0)   # (4, 721, 1440)
if input_surface.shape != (4, 721, 1440):
    print(f"[ERA5] ERROR: unexpected surface shape {input_surface.shape}, expected (4, 721, 1440)")
    sys.exit(1)
print(f"[ERA5] input_surface shape: {input_surface.shape}  dtype: {input_surface.dtype}")

print("[ERA5] Converting upper-air grib to numpy (13 pressure levels)…")
ds_upper = xr.open_dataset(
    UPPER_GRIB, engine="cfgrib",
    backend_kwargs={
        "indexpath": "",
        "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
    },
)

PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

z_list, q_list, t_list, u_list, v_list = [], [], [], [], []
for pl in PRESSURE_LEVELS:
    z_list.append(ds_upper["z"].sel(isobaricInhPa=pl).values.astype(np.float32))
    q_list.append(ds_upper["q"].sel(isobaricInhPa=pl).values.astype(np.float32))
    t_list.append(ds_upper["t"].sel(isobaricInhPa=pl).values.astype(np.float32))
    u_list.append(ds_upper["u"].sel(isobaricInhPa=pl).values.astype(np.float32))
    v_list.append(ds_upper["v"].sel(isobaricInhPa=pl).values.astype(np.float32))
ds_upper.close()

# Stack: (5 variables, 13 levels, 721 lat, 1440 lon)
input_upper = np.stack([
    np.stack(z_list, axis=0),
    np.stack(q_list, axis=0),
    np.stack(t_list, axis=0),
    np.stack(u_list, axis=0),
    np.stack(v_list, axis=0),
], axis=0)

if input_upper.shape != (5, 13, 721, 1440):
    print(f"[ERA5] ERROR: unexpected upper-air shape {input_upper.shape}, expected (5, 13, 721, 1440)")
    sys.exit(1)
print(f"[ERA5] input_upper  shape: {input_upper.shape}  dtype: {input_upper.dtype}")

# ---------------------------------------------------------------------------
# Step 5 — Save to input_data/ and clean up grib files
# ---------------------------------------------------------------------------
np.save(SURFACE_NPY, input_surface)
np.save(UPPER_NPY,   input_upper)
print(f"[ERA5] Saved {SURFACE_NPY}")
print(f"[ERA5] Saved {UPPER_NPY}")

for grib in (SURFACE_GRIB, UPPER_GRIB):
    if os.path.exists(grib):
        os.remove(grib)

# ---------------------------------------------------------------------------
# Step 6 — Save metadata JSON (read by data_loader / Streamlit sidebar)
# ---------------------------------------------------------------------------
meta = {
    "date":       date_str,
    "time":       time_str,
    "fetched_at": datetime.utcnow().isoformat(),
    "source":     "ERA5 reanalysis via CDS API",
}
with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)
print(f"[ERA5] Metadata written to {META_JSON}")

# Also delete old Pangu cache so the next app load runs fresh inference
cache_path = os.path.join(_HERE, "..", "output_data", "pangu_output.npz")
if os.path.exists(cache_path):
    os.remove(cache_path)
    print(f"[ERA5] Deleted stale cache {cache_path} — next app load will re-run inference.")

print(f"\n✅ ERA5 data ready for {date_str} {time_str} UTC")
print("✅ Run the FarmCast app — it will now use real atmospheric data.")
print("   streamlit run app.py --server.port 8501 --server.address 0.0.0.0")
