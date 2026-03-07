"""
era5_fetcher.py

Download ERA5 reanalysis data from CDS and convert it to the NumPy layout
expected by Pangu-Weather:
  input_surface.npy -> (4, 721, 1440)
  input_upper.npy   -> (5, 13, 721, 1440)
"""

import os
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np

PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

_SURFACE_VARS = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
]

_UPPER_VARS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

_CACHE_MAX_AGE_SECONDS = 6 * 3600


def _latest_era5_time() -> datetime:
    """
    Pick a likely available ERA5 analysis cycle.
    ERA5T latency is typically multiple days; use a 6-day safety offset.
    """
    now = datetime.now(timezone.utc)
    target = now - timedelta(days=6)
    hour = 12 if target.hour >= 12 else 0
    return target.replace(hour=hour, minute=0, second=0, microsecond=0)


def _cache_is_fresh(surface_npy: str, upper_npy: str) -> bool:
    if not os.path.exists(surface_npy) or not os.path.exists(upper_npy):
        return False
    age = time.time() - os.path.getmtime(surface_npy)
    return age < _CACHE_MAX_AGE_SECONDS


def _download_era5_surface(target_time: datetime, output_file: str) -> None:
    import cdsapi

    client = cdsapi.Client()
    request = {
        "product_type": ["reanalysis"],
        "variable": _SURFACE_VARS,
        "year": [str(target_time.year)],
        "month": [f"{target_time.month:02d}"],
        "day": [f"{target_time.day:02d}"],
        "time": [f"{target_time.hour:02d}:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    client.retrieve("reanalysis-era5-single-levels", request, output_file)


def _download_era5_pressure(target_time: datetime, output_file: str) -> None:
    import cdsapi

    client = cdsapi.Client()
    request = {
        "product_type": ["reanalysis"],
        "variable": _UPPER_VARS,
        "pressure_level": [str(p) for p in PRESSURE_LEVELS],
        "year": [str(target_time.year)],
        "month": [f"{target_time.month:02d}"],
        "day": [f"{target_time.day:02d}"],
        "time": [f"{target_time.hour:02d}:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    client.retrieve("reanalysis-era5-pressure-levels", request, output_file)


def _resolve_var_by_long_name(ds, available_vars: list[str], short_name: str) -> str:
    if short_name in available_vars:
        return short_name
    for var_name in available_vars:
        long_name = ds[var_name].attrs.get("long_name", "").lower()
        if short_name == "msl" and "sea level pressure" in long_name:
            return var_name
        if short_name == "u10" and "10" in long_name and "u" in long_name:
            return var_name
        if short_name == "v10" and "10" in long_name and "v" in long_name:
            return var_name
        if short_name == "t2m" and "2" in long_name and "temperature" in long_name:
            return var_name
        if short_name == "z" and "geopotential" in long_name:
            return var_name
        if short_name == "q" and "specific humidity" in long_name:
            return var_name
        if short_name == "t" and "temperature" in long_name and "dew" not in long_name:
            return var_name
        if short_name == "u" and (("u component" in long_name) or ("u-component" in long_name)):
            return var_name
        if short_name == "v" and (("v component" in long_name) or ("v-component" in long_name)):
            return var_name
    raise KeyError(f"Cannot find ERA5 variable '{short_name}' in {available_vars}")


def _netcdf_to_surface_npy(nc_path: str, npy_path: str) -> None:
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    available = list(ds.data_vars)
    var_order = [
        _resolve_var_by_long_name(ds, available, "msl"),
        _resolve_var_by_long_name(ds, available, "u10"),
        _resolve_var_by_long_name(ds, available, "v10"),
        _resolve_var_by_long_name(ds, available, "t2m"),
    ]

    surface = np.zeros((4, 721, 1440), dtype=np.float32)
    for idx, var_name in enumerate(var_order):
        data = ds[var_name].values
        while data.ndim > 2:
            data = data[0]
        if data.shape != (721, 1440):
            raise ValueError(f"Unexpected shape for {var_name}: {data.shape}; expected (721, 1440)")
        surface[idx] = data.astype(np.float32)

    ds.close()
    np.save(npy_path, surface)


def _netcdf_to_upper_npy(nc_path: str, npy_path: str) -> None:
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    available = list(ds.data_vars)
    var_order = [
        _resolve_var_by_long_name(ds, available, "z"),
        _resolve_var_by_long_name(ds, available, "q"),
        _resolve_var_by_long_name(ds, available, "t"),
        _resolve_var_by_long_name(ds, available, "u"),
        _resolve_var_by_long_name(ds, available, "v"),
    ]

    pressure_dim = None
    for dim_name in ["level", "pressure_level", "isobaricInhPa", "plev"]:
        if dim_name in ds.dims or dim_name in ds.coords:
            pressure_dim = dim_name
            break
    if pressure_dim is None:
        raise KeyError(f"Cannot find pressure dimension in dataset dims: {list(ds.dims)}")

    levels = np.asarray(ds[pressure_dim].values, dtype=float)
    upper = np.zeros((5, 13, 721, 1440), dtype=np.float32)

    target_indices = []
    for p in PRESSURE_LEVELS:
        matches = np.where(np.isclose(levels, float(p)))[0]
        if len(matches) == 0:
            raise ValueError(f"Pressure level {p} hPa not found in file levels {levels.tolist()}")
        target_indices.append(int(matches[0]))

    for idx, var_name in enumerate(var_order):
        data = ds[var_name].values
        while data.ndim > 3:
            data = data[0]
        if data.shape[0] != len(levels):
            raise ValueError(f"Unexpected level count for {var_name}: {data.shape}")
        data = data[target_indices]
        if data.shape != (13, 721, 1440):
            raise ValueError(f"Unexpected shape for {var_name}: {data.shape}; expected (13, 721, 1440)")
        upper[idx] = data.astype(np.float32)

    ds.close()
    np.save(npy_path, upper)


def ensure_era5_inputs(input_dir: str, force_refresh: bool = False) -> tuple[str, str]:
    """
    Ensure `input_surface.npy` and `input_upper.npy` exist and are reasonably fresh.
    Returns `(surface_npy_path, upper_npy_path)`.
    """
    os.makedirs(input_dir, exist_ok=True)

    surface_npy = os.path.join(input_dir, "input_surface.npy")
    upper_npy = os.path.join(input_dir, "input_upper.npy")
    meta_json = os.path.join(input_dir, "era5_meta.json")
    surface_nc = os.path.join(input_dir, "era5_surface.nc")
    upper_nc = os.path.join(input_dir, "era5_pressure.nc")

    if not force_refresh and _cache_is_fresh(surface_npy, upper_npy):
        return surface_npy, upper_npy

    target_time = _latest_era5_time()

    try:
        _download_era5_surface(target_time, surface_nc)
        _download_era5_pressure(target_time, upper_nc)

        _netcdf_to_surface_npy(surface_nc, surface_npy)
        _netcdf_to_upper_npy(upper_nc, upper_npy)

        with open(meta_json, "w") as f:
            f.write("{\n")
            f.write(f'  "date": "{target_time.strftime("%Y-%m-%d")}",\n')
            f.write(f'  "time": "{target_time.strftime("%H:%M")}",\n')
            f.write(f'  "fetched_at": "{datetime.now(timezone.utc).isoformat()}",\n')
            f.write('  "source": "ERA5 reanalysis via CDS API"\n')
            f.write("}\n")

        return surface_npy, upper_npy
    except Exception as exc:
        warnings.warn(f"ERA5 download/conversion failed: {exc}", stacklevel=2)
        if os.path.exists(surface_npy) and os.path.exists(upper_npy):
            return surface_npy, upper_npy
        raise RuntimeError(f"No usable ERA5 inputs available: {exc}") from exc
    finally:
        for path in [surface_nc, upper_nc]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
