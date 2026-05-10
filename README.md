# Chapter 4: Case-Study Model Comparison for Marine Heatwave Events

This repository contains the Python scripts used in Chapter 4 to compare short-lead sea surface temperature (SST) forecasts from 
a deep learning model (DL) against an established regional ocean model during selected marine heatwave case-study events around Ireland.

The comparison is designed around two 3-day extreme-event windows:

| Case study | Event window | Script | Main comparison |
|---|---:|---|---|
| Case Study 1 | 22–24 July 2021 | `chap4_case1_final.py` | CNN2D-LSTM forecast vs NEATL-ROMS vs observed/merged SST |
| Case Study 2 | 20–22 June 2023 | `chap4_case2_final.py` | CNN2D-LSTM forecast vs NEATL-ROMS vs observed/merged SST |

The scripts calculate regional, daily, spatial, and site-specific verification outputs. They are intended to support reproducible comparison of model behaviour 
during short marine heatwave events.

---

## 1. Scientific aim

The purpose of these scripts is to evaluate whether a data-driven CNN2D-LSTM forecast can reproduce short-term SST evolution during marine heatwave conditions, 
and to compare its behaviour with the NEATL-ROMS regional ocean model.

The analysis focuses on:

- **Domain-wide skill** over the Irish study region.
- **Daily model performance** during each 3-day event window.
- **Spatial error structure**, using bias and RMSE maps.
- **Site-specific behaviour** at selected aquaculture-relevant locations.
- **Model complementarity**, especially where deep learning and numerical models show different strengths or weaknesses.

---


## 2. Required input files

Both scripts require three main NetCDF inputs:

### 2.1 NEATL-ROMS model output

```python
PATH_ROMS = "NEATL-2021-2025-TEMP-REGRID-FINAL.nc"
VAR_ROMS = "temp"
```

This file contains the regional ocean model SST field used for comparison. In these scripts, the expected variable name is `temp`.

### 2.2 Observed / merged SST reference dataset

```python
PATH_TRUTH = "noaa_icesmi_combinefile_FINAL_1res1982_2024.nc"
VAR_TRUTH = "sst"
```

This is treated as the reference or observed SST dataset. It is the dataset against which both model forecasts are verified.

### 2.3 CNN2D-LSTM forecast output

For Case Study 1:

```python
PATH_CNN_H3 = "CNN_LSTM_test/cnn2dlstm_pred_2021_Jul22_24_seq15.nc"
VAR_CNN = "sst"
```

For Case Study 2:

```python
PATH_CNN_H3 = "CNN_LSTM_test/cnn2dlstm_pred_2023_Jun20_22_seq15.nc"
VAR_CNN = "sst"
```

The CNN2D-LSTM files contain the 3-day forecast SST fields generated from the deep learning model.

---

## 3. Python environment

The scripts were written for Python-based scientific analysis using Xarray, NumPy, Pandas, Matplotlib, and optionally Cartopy.

### 3.1 Core dependencies

```text
python >= 3.9
xarray
numpy
pandas
matplotlib
netCDF4
h5netcdf
```

### 3.2 Optional dependency for map plotting

```text
cartopy
```

Cartopy is optional. If Cartopy is not available, the scripts automatically fall back to basic Xarray/Matplotlib plotting.

### 3.3 Example installation using conda

```bash
conda create -n chap4_model_compare python=3.10
conda activate chap4_model_compare
conda install -c conda-forge xarray numpy pandas matplotlib netcdf4 h5netcdf cartopy
```

### 3.4 Example installation using pip

```bash
python -m venv chap4_model_compare
source chap4_model_compare/bin/activate  # Linux/macOS
# On Windows: chap4_model_compare\Scripts\activate

pip install xarray numpy pandas matplotlib netCDF4 h5netcdf
pip install cartopy  # optional; conda is often easier for Cartopy
```

---

## 4. How to run the scripts

Place the required NetCDF files in the locations expected by the script, or edit the path variables at the top of each script.

Run Case Study 1:

```bash
python chap4_case1_final.py
```

Run Case Study 2:

```bash
python chap4_case2_final.py
```

Each script creates its own output folder automatically.

---

## 5. What each script does

Each script follows the same workflow:

1. **Load datasets** using Xarray.
2. **Standardise coordinates** by renaming `latitude`/`longitude` to `lat`/`lon` where needed.
3. **Floor timestamps to daily resolution** so all datasets can be compared on daily dates.
4. **Convert longitudes to 0–360° format** if needed.
5. **Align forecast valid times** to the observed/reference SST dates using possible 0–3 day shifts.
6. **Subset the 3-day event window**.
7. **Find common dates** across the CNN2D-LSTM forecast, NEATL-ROMS output, and observed/reference SST.
8. **Calculate regional metrics** over the full 3-day window.
9. **Calculate daily metrics** for each event day.
10. **Generate spatial bias and RMSE maps**.
11. **Generate site-specific SST time-series plots** for Galway Bay and Bantry Bay.
12. **Save outputs** as CSV, NetCDF, and PNG files.

---

## 6. Case-study configuration

The event window is controlled by:

```python
START_DATE = np.datetime64("YYYY-MM-DD")
END_DATE   = np.datetime64("YYYY-MM-DD")
```

For the 2021 case:

```python
START_DATE = np.datetime64("2021-07-22")
END_DATE   = np.datetime64("2021-07-24")
```

For the 2023 case:

```python
START_DATE = np.datetime64("2023-06-20")
END_DATE   = np.datetime64("2023-06-22")
```

The scripts currently evaluate a 3-day forecast horizon, referred to in the output labels as `H3` or `H=3 days`.

---

## 7. Site-specific locations

The scripts currently extract site time series for:

| Site | Latitude | Longitude format used in script |
|---|---:|---:|
| Galway Bay | 53.20 | 350.70 |
| Bantry Bay | 51.68 | 350.50 |

Longitudes are given in 0–360° format. For example, 350.70° corresponds approximately to -9.30° in -180–180° format.

To add another site, edit:

```python
SITE_LIST = [
    ("Galway Bay", 53.20, 350.70),
    ("Bantry Bay", 51.68, 350.50),
]
```

For example:

```python
SITE_LIST.append(("New Site", 52.50, 349.80))
```

---

## 8. Output files

### 8.1 Case Study 1 output folder

```text
CNN_LSTM_test/hindcast_compare_July22_2021_H3_only_TEST2/
```

Main outputs:

```text
regional_metrics_h3_July_2021_3days.csv
regional_metrics_h3_July_2021_daily.csv
bias_map_CNN_H3.nc
bias_map_CNN_H3.png
bias_map_NEATL_H3.nc
bias_map_NEATL_H3.png
rmse_map_CNN_H3.nc
rmse_map_CNN_H3.png
rmse_map_NEATL_H3.nc
rmse_map_NEATL_H3.png
site_timeseries_Galway_Bay_H-3_days.png
site_timeseries_Bantry_Bay_H-3_days.png
```

### 8.2 Case Study 2 output folder

```text
CNN_LSTM_test/hindcast_compare_June20_2023_H3_only_TEST2/
```

Main outputs:

```text
regional_metrics_h3_June_2023_3days.csv
regional_metrics_h3_June_2023_daily.csv
bias_map_CNN_H3.nc
bias_map_CNN_H3.png
bias_map_NEATL_H3.nc
bias_map_NEATL_H3.png
rmse_map_CNN_H3.nc
rmse_map_CNN_H3.png
rmse_map_NEATL_H3.nc
rmse_map_NEATL_H3.png
site_timeseries_Galway_Bay_H-3_days.png
site_timeseries_Bantry_Bay_H-3_days.png
```

---

## 9. Metrics calculated

The scripts calculate the following metrics using only grid cells where both model and reference SST values are finite.

### 9.1 Bias

Bias measures whether the model is systematically warmer or cooler than the reference SST.

```text
Bias = mean(model SST - reference SST)
```

A positive bias means the model is warmer than the reference dataset. A negative bias means the model is cooler.

### 9.2 MAE

Mean Absolute Error measures the average absolute difference between the model and reference SST.

```text
MAE = mean(abs(model SST - reference SST))
```

### 9.3 RMSE

Root Mean Square Error gives stronger weight to larger errors.

```text
RMSE = sqrt(mean((model SST - reference SST)^2))
```

### 9.4 R²

R² is calculated across all valid spatiotemporal samples for the 3-day regional score and across all valid spatial grid cells for each daily score.

```text
R² = 1 - SS_res / SS_tot
```

R² can be negative if the model performs worse than simply predicting the mean of the reference SST.

### 9.5 Bootstrap confidence interval for RMSE

The scripts also calculate a simple block-bootstrap confidence interval for the mean daily RMSE. Because each case-study window has only 3 days, 
the confidence interval should be interpreted cautiously and mainly used as a consistency indicator rather than a robust long-term uncertainty estimate.

---

## 10. Notes on time alignment

The scripts include an `auto_align_to_truth()` function. This tests 0, 1, 2, and 3-day shifts and selects the shift that gives the largest date overlap with the reference SST.

This is useful because different forecast files may store time as either forecast initialisation time or valid forecast time. The function helps ensure that the model output is compared against the correct observation dates.

Users should still check the printed common dates before interpreting results.

---

## 11. Notes on spatial masking

All metrics use a dynamic common finite-value mask:

```python
valid = np.isfinite(forecast) & np.isfinite(observation)
```

This means that only locations where both the forecast and the reference dataset contain valid SST are included in the metric calculation.

This avoids unfairly penalising a model for missing values where no valid reference value is available, or vice versa.

---

## 12. Reusing the scripts for another event

To reuse these scripts for another case study:

1. Copy either `chap4_case1_final.py` or `chap4_case2_final.py`.
2. Rename the copied script, for example:

```text
chap4_case3_final.py
```

3. Update the event dates:

```python
START_DATE = np.datetime64("YYYY-MM-DD")
END_DATE   = np.datetime64("YYYY-MM-DD")
```

4. Update the CNN2D-LSTM forecast file path:

```python
PATH_CNN_H3 = "path/to/new/cnn_forecast_file.nc"
```

5. Update the output directory:

```python
OUT_DIR = Path("CNN_LSTM_test/new_case_output_folder")
```

6. Update site locations if required:

```python
SITE_LIST = [
    ("Site name", latitude, longitude_0_360),
]
```

7. Run the script and check:

```text
H3 common days in window: ...
```

The number of common days should match the expected event window. For a 3-day event, it should usually print 3 common days.

---

## 13. Important assumptions and limitations

- The observed/merged SST dataset is treated as the reference dataset.
- The comparison is limited to short 3-day event windows.
- The confidence intervals are based on very few daily samples and should not be over-interpreted.
- Site-specific values are extracted from the nearest finite grid cell, not from an exact in-situ station point.
- Longitudes are internally handled in 0–360° format, while Cartopy maps are converted to -180–180° format for plotting.
- The scripts compare SST fields only; they do not include atmospheric forcing, currents, mixed-layer dynamics, or other explanatory variables.
- Model comparison results are case-study-specific and should not be generalised without further evaluation across more events and seasons.

---

