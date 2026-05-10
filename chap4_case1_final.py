import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Cartopy (optional)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


# =========================
# CONFIG – 3-DAY WINDOW ONLY
# =========================
START_DATE = np.datetime64("2021-07-22")
END_DATE   = np.datetime64("2021-07-24")   # 3 days: 22, 23, 24 July 2021

# Inputs
PATH_ROMS   = "NEATL-2021-2025-TEMP-REGRID-FINAL.nc"            # NEATL SST (no depth)
PATH_TRUTH  = "noaa_icesmi_combinefile_FINAL_1res1982_2024.nc"  
PATH_CNN_H3 = "CNN_LSTM_test/cnn2dlstm_pred_2021_Jul22_24_seq15.nc"

# Variable names
VAR_ROMS  = "temp"
VAR_TRUTH = "sst"
VAR_CNN   = "sst"

# Output
OUT_DIR = Path("CNN_LSTM_test/hindcast_compare_July22_2021_H3_only_TEST2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sites (Name, lat, lon in 0–360°)
SITE_LIST = [
    ("Galway Bay", 53.20, 350.70),
    ("Bantry Bay", 51.68, 350.50),
]


# =========================
# HELPERS
# =========================
def harmonize(ds: xr.Dataset) -> xr.Dataset:
    """Standardise coordinate names and floor time to daily."""
    ren = {}
    if "latitude"  in ds.coords: ren["latitude"]  = "lat"
    if "longitude" in ds.coords: ren["longitude"] = "lon"
    if ren:
        ds = ds.rename(ren)
    if "time" in ds.coords:
        ds = ds.assign_coords(time=ds["time"].dt.floor("D"))
    return ds

def to0360(da: xr.DataArray) -> xr.DataArray:
    """Convert lon to 0–360° if needed."""
    if (da.lon < 0).any():
        da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
    return da

def lon360_to180(da: xr.DataArray) -> xr.DataArray:
    """Convert lon 0–360° to –180–180° for Cartopy."""
    lon = ((da.lon + 180) % 360) - 180
    return da.assign_coords(lon=lon).sortby("lon")

def cut_window(da: xr.DataArray, start=START_DATE, end=END_DATE) -> xr.DataArray:
    """Subset to evaluation window."""
    return da.sel(time=slice(start, end))

def intersect_times(*das: xr.DataArray) -> np.ndarray:
    """Common dates between all data arrays."""
    t = das[0].time.values
    for da in das[1:]:
        t = np.intersect1d(t, da.time.values)
    return np.sort(t)

def shift_to_valid_time(da: xr.DataArray, lead_days: int) -> xr.DataArray:
    """Shift forecast times by a given number of days."""
    return da if lead_days == 0 else da.assign_coords(time=da.time + np.timedelta64(lead_days, "D"))

def auto_align_to_truth(fore: xr.DataArray, truth: xr.DataArray, candidates=(0, 1, 2, 3)) -> xr.DataArray:
    best, best_cnt = None, -1
    for lead in candidates:
        test = shift_to_valid_time(fore, lead)
        cnt  = len(np.intersect1d(test.time.values, truth.time.values))
        if cnt > best_cnt:
            best, best_cnt = test, cnt
    return best

def common_mask_dynamic(f: xr.DataArray, o: xr.DataArray):
    valid = np.isfinite(f) & np.isfinite(o)
    return f.where(valid), o.where(valid)

def r2_score_spatiotemporal(fore: xr.DataArray, obs: xr.DataArray) -> float:
    """
    R² over all valid samples across (time, lat, lon).
    Note: R² can be negative if the model is worse than predicting the mean.
    """
    f, o = common_mask_dynamic(fore, obs)
    fv = f.values.ravel()
    ov = o.values.ravel()

    m = np.isfinite(fv) & np.isfinite(ov)
    fv = fv[m]
    ov = ov[m]

    if ov.size < 2:
        return np.nan

    ss_res = np.sum((fv - ov) ** 2)
    ss_tot = np.sum((ov - np.mean(ov)) ** 2)

    if ss_tot == 0:
        return np.nan

    return float(1.0 - ss_res / ss_tot)

def metrics_regional(fore: xr.DataArray, obs: xr.DataArray):
    """Domain-wide bias, MAE, RMSE, R² over the full 3-day window."""
    f, o = common_mask_dynamic(fore, obs)
    d = f - o
    bias = float(d.mean(("time", "lat", "lon")).values)
    mae  = float(np.abs(d).mean(("time", "lat", "lon")).values)
    rmse = float(np.sqrt((d**2).mean(("time", "lat", "lon"))).values)
    r2   = r2_score_spatiotemporal(fore, obs)
    return dict(bias=bias, mae=mae, rmse=rmse, r2=r2)

def metrics_regional_by_day(fore: xr.DataArray, obs: xr.DataArray) -> pd.DataFrame:
    """
    Regional metrics for each day (computed over spatial grid only for that day).
    Returns columns: time, bias, mae, rmse, r2.
    """
    days = np.array(fore.time.values)
    rows = []
    for t in days:
        f_t = fore.sel(time=t)
        o_t = obs.sel(time=t)

        f_m, o_m = common_mask_dynamic(f_t, o_t)
        d = f_m - o_m

        bias = float(d.mean(("lat", "lon")).values)
        mae  = float(np.abs(d).mean(("lat", "lon")).values)
        rmse = float(np.sqrt((d**2).mean(("lat", "lon"))).values)

        # spatial R² for that day
        fv = f_m.values.ravel()
        ov = o_m.values.ravel()
        m = np.isfinite(fv) & np.isfinite(ov)
        fv, ov = fv[m], ov[m]

        if ov.size < 2:
            r2 = np.nan
        else:
            ss_res = np.sum((fv - ov) ** 2)
            ss_tot = np.sum((ov - np.mean(ov)) ** 2)
            r2 = np.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

        rows.append(dict(time=pd.to_datetime(str(t)), bias=bias, mae=mae, rmse=rmse, r2=r2))

    return pd.DataFrame(rows).sort_values("time")

def daily_rmse_series(fore: xr.DataArray, obs: xr.DataArray) -> pd.Series:
    """Daily spatial RMSE (per day, averaged over all grid points)."""
    f, o = common_mask_dynamic(fore, obs)
    return np.sqrt(((f - o) ** 2).mean(("lat", "lon"))).to_series()

def bias_rmse_maps(fore: xr.DataArray, obs: xr.DataArray):
    """2D maps of time-mean bias and RMSE across the 3-day window."""
    f, o = common_mask_dynamic(fore, obs)
    d = f - o
    bias = d.mean("time").rename("BIAS")
    rmse = np.sqrt((d**2).mean("time")).rename("RMSE")
    return bias, rmse

def nearest_finite_series_pair(
    fore: xr.DataArray, obs: xr.DataArray, lat: float, lon: float, max_radius: int = 6
):
    """
    Find a nearby grid point where both forecast and obs have finite data.
    Expands search in a small square if the nearest point is NaN.
    """
    i_lat = int(np.argmin(np.abs(fore.lat.values - lat)))
    i_lon = int(np.argmin(np.abs(fore.lon.values - lon)))
    for r in range(0, max_radius + 1):
        sl_lat = slice(max(0, i_lat - r), min(fore.lat.size, i_lat + r + 1))
        sl_lon = slice(max(0, i_lon - r), min(fore.lon.size, i_lon + r + 1))
        F = fore.isel(lat=sl_lat, lon=sl_lon)
        O = obs.isel(lat=sl_lat,  lon=sl_lon)
        mask = np.isfinite(F).any("time") & np.isfinite(O).any("time")
        if mask.any():
            jj, ii = np.argwhere(mask.values)[0]
            return F.isel(lat=jj, lon=ii), O.isel(lat=jj, lon=ii)
    # fallback: nearest neighbour
    return fore.sel(lat=lat, lon=lon, method="nearest"), obs.sel(lat=lat, lon=lon, method="nearest")


# =========================
# LOAD DATA
# =========================
ds_roms_full  = harmonize(xr.open_dataset(PATH_ROMS))
ds_truth_full = harmonize(xr.open_dataset(PATH_TRUTH))
ds_cnn3_full  = harmonize(xr.open_dataset(PATH_CNN_H3))

roms_full  = to0360(ds_roms_full[VAR_ROMS])
truth_full = to0360(ds_truth_full[VAR_TRUTH])
cnn3_full  = to0360(ds_cnn3_full[VAR_CNN])

# Align ROMS/CNN forecast times to truth's valid dates
cnn3_full = auto_align_to_truth(cnn3_full, truth_full, candidates=(0, 1, 2, 3))
roms_full = auto_align_to_truth(roms_full,  truth_full, candidates=(0, 1, 2, 3))

# Window + intersect dates for the 3-day event
roms_w  = cut_window(roms_full)
truth_w = cut_window(truth_full)
cnn3_w  = cut_window(cnn3_full)

h3_dates = intersect_times(truth_w, roms_w, cnn3_w)
roms_h3, truth_h3, cnn_h3 = (
    roms_w.sel(time=h3_dates),
    truth_w.sel(time=h3_dates),
    cnn3_w.sel(time=h3_dates),
)

print(f"H3 common days in window: {len(h3_dates)} -> {h3_dates}")


# =========================
# REGIONAL METRICS + CI (3 days)  +  PER-DAY METRICS (NEW)
# =========================
def block_bootstrap_ci(series: np.ndarray, n_boot=2000, block=3, alpha=0.05, seed=42):
    """
    Block bootstrap CI for the mean of daily RMSE.
    For a 3-day window, the whole record as one block
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    n = x.size
    if n < block:
        bs = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    else:
        blocks = np.lib.stride_tricks.sliding_window_view(x, block)
        k = max(1, int(np.ceil(n / block)))
        bs = np.array([
            np.nanmean(blocks[rng.integers(0, blocks.shape[0], size=k), :].reshape(-1)[:n])
            for _ in range(n_boot)
        ])
    return float(np.nanmean(x)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

def regional_with_ci(fore, obs, label):
    """3-day metrics + RMSE CI for a given model."""
    m = metrics_regional(fore, obs)
    rmse_series = daily_rmse_series(fore, obs)
    mu, lo, hi = block_bootstrap_ci(rmse_series.values, n_boot=2000, block=3, alpha=0.05, seed=42)
    m.update(dict(model=label, rmse_mean=mu, rmse_lo=lo, rmse_hi=hi, n_days=int(rmse_series.notna().sum())))
    return m

rows = []
if len(h3_dates) > 0:
    rows += [
        regional_with_ci(cnn_h3,  truth_h3, "CNN_H3"),
        regional_with_ci(roms_h3, truth_h3, "NEATL_H3"),
    ]

# --- Aggregate (3-day) metrics CSV ---
metrics_df = pd.DataFrame(rows).sort_values(["model"])
metrics_df.to_csv(OUT_DIR / "regional_metrics_h3_July_2021_3days.csv", index=False)
print("\nRegional metrics for H=3 days (3-day window):\n", metrics_df)

# --- Per-day metrics CSV ---
if len(h3_dates) > 0:
    daily_rows = []
    for label, fore in [("CNN_H3", cnn_h3), ("NEATL_H3", roms_h3)]:
        df_day = metrics_regional_by_day(fore, truth_h3)
        df_day["model"] = label
        daily_rows.append(df_day)

    daily_metrics_df = pd.concat(daily_rows, ignore_index=True)[["model", "time", "bias", "mae", "rmse", "r2"]]
    daily_metrics_df.to_csv(OUT_DIR / "regional_metrics_h3_July_2021_daily.csv", index=False)
    print("\nPer-day regional metrics saved to: regional_metrics_h3_July_2021_daily.csv")
    print(daily_metrics_df)


# =========================
# COMPACT CARTOPY MAPS (3-day mean)
# =========================
def plot_map_compact(da: xr.DataArray, title: str, path: Path):
    if not HAS_CARTOPY:
        fig = plt.figure(figsize=(8.0, 4.8))
        ax = plt.gca()
        da.plot(ax=ax)
        plt.title(title)
        plt.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
        plt.close()
        return

    dab = lon360_to180(da)  # [-180,180]
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(9.0, 5.4))
    ax = plt.axes(projection=proj)

    lon0, lon1 = float(dab.lon.min()) - 0.25, float(dab.lon.max()) + 0.25
    lat0, lat1 = float(dab.lat.min()) - 0.20, float(dab.lat.max()) + 0.20
    ax.set_extent([lon0, lon1, lat0, lat1], crs=proj)

    pm = ax.pcolormesh(
        dab.lon, dab.lat, dab.values,
        transform=proj, shading="auto", rasterized=True
    )

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.coastlines(resolution="10m", linewidth=0.8, zorder=3)

    gl = ax.gridlines(
        draw_labels=True, x_inline=False, y_inline=False,
        linewidth=0.35, color="gray", alpha=0.55, linestyle="--"
    )
    gl.right_labels = False
    gl.top_labels   = False
    gl.left_labels  = True
    gl.bottom_labels= True
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}

    cb = plt.colorbar(pm, ax=ax, shrink=0.82, pad=0.006)
    cb.set_label(dab.name or "")

    ax.set_title(title)
    plt.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close()

def save_maps_for(tag: str, fore: xr.DataArray, obs: xr.DataArray):
    b, r = bias_rmse_maps(fore, obs)  # mean over the 3 days
    b.to_netcdf(OUT_DIR / f"bias_map_{tag}.nc")
    r.to_netcdf(OUT_DIR / f"rmse_map_{tag}.nc")
    plot_map_compact(b, f"BIAS - {tag}", OUT_DIR / f"bias_map_{tag}.png")
    plot_map_compact(r, f"RMSE - {tag}", OUT_DIR / f"rmse_map_{tag}.png")

if len(h3_dates) > 0:
    save_maps_for("CNN_H3",   cnn_h3,  truth_h3)
    save_maps_for("NEATL_H3", roms_h3, truth_h3)


# =========================
# SITE TIME-SERIES
# =========================
def plot_site_horizon(name, lat, lon, fore, roms, obs, tag):
    t = intersect_times(fore, roms, obs)
    if len(t) == 0:
        print(f"[SITE] No overlap for {name} ({tag}); skipping.")
        return

    f_loc, o_loc = nearest_finite_series_pair(fore.sel(time=t), obs.sel(time=t), lat, lon)
    r_loc, _     = nearest_finite_series_pair(roms.sel(time=t), obs.sel(time=t), lat, lon)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(o_loc.time.values, o_loc.values, label="Obs", linewidth=1.4)
    ax.plot(f_loc.time.values, f_loc.values, label=f"CNN {tag}", linewidth=1.1)
    ax.plot(r_loc.time.values, r_loc.values, label=f"NEATL {tag}", linewidth=1.1)

    ticks = pd.date_range(str(START_DATE), str(END_DATE), freq="D")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %Y"))

    ax.set_title(f"{name} - {tag}")
    ax.set_ylabel("SST (°C)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()

    fname = f"site_timeseries_{name.replace(' ','_')}_{tag.replace('=','').replace(' ','_')}.png"
    plt.savefig(OUT_DIR / fname, dpi=160)
    plt.close()

if len(h3_dates) > 0:
    for nm, la, lo in SITE_LIST:
        plot_site_horizon(nm, la, lo, cnn_h3, roms_h3, truth_h3, "H=3 days")

print(f"\nAll outputs in: {OUT_DIR.resolve()}")
print(" - regional_metrics_h3_July_2021_3days.csv")
print(" - regional_metrics_h3_July_2021_daily.csv")
print(" - bias_map_CNN_H3/NEATL_H3 (nc/png)")
print(" - rmse_map_CNN_H3/NEATL_H3 (nc/png)")
print(" - site_timeseries_*_H-3_days.png")
