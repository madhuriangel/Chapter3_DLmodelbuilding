"""
map_four_locations_cartopy.py

Create a map of four Irish aquaculture locations using Cartopy.

- Input: hard-coded latitude/longitude and site names
- Processing:
    * Convert longitudes from 0–360° to –180–180° (for Cartopy plotting)
    * Draw land/sea background, coastlines, borders, and optional counties shapefile
    * Focus the map on Ireland
    * Plot site markers and labels
- Output:
    * Saves high-resolution PNG: 'ireland_four_points_cartopy.png'
    * Also displays the figure interactively

"""

import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader

# ----------------------------------------------------------------------
# Data: four aquaculture locations (lat, lon in degrees, 0–360 convention)
# ----------------------------------------------------------------------
points = [
    {"lat": 51.375, "lon": 350.375, "name": "Bantry South"},
    {"lat": 51.625, "lon": 349.875, "name": "Tahilla"},
    {"lat": 52.375, "lon": 349.875, "name": "Castlemaine Harbour"},
    {"lat": 53.625, "lon": 350.125, "name": "Killary Outer"},
]


def to_minus180_180(lon: float) -> float:
    """
    Convert longitude from [0, 360) to the [-180, 180) range.

    Cartopy typically expects longitudes in the –180° to 180° convention when
    using PlateCarree. This helper ensures longitudes like 350° become –10°.

    Parameters
    ----------
    lon : float
        Longitude in degrees in the 0–360° convention.

    Returns
    -------
    float
        Longitude in the –180–180° convention.
    """
    return ((lon + 180.0) % 360.0) - 180.0


# Add converted longitudes for plotting
for p in points:
    p["lon_plot"] = to_minus180_180(p["lon"])

# ----------------------------------------------------------------------
# Map setup
# ----------------------------------------------------------------------
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(5.7, 3.5))
ax = plt.axes(projection=proj)

# ----------------------------------------------------------------------
# Base map features (Natural Earth, via cartopy.feature)
# ----------------------------------------------------------------------
try:
    # Ocean and land background
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(
        cfeature.LAND,
        edgecolor="none",
        facecolor="#b8e3b8",   # light green land colour
        zorder=1
    )

    # Coastlines and borders
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=2)

    # Lakes
    ax.add_feature(
        cfeature.LAKES,
        edgecolor="black",
        facecolor="#cfe8ff",   # light blue lakes
        linewidth=0.3,
        zorder=1
    )
except Exception:
    # Fallback: basic coastlines if Natural Earth features fail
    ax.coastlines(resolution="10m", linewidth=0.6)

# ----------------------------------------------------------------------
# Optional: overlay local counties shapefile (if available)
# ----------------------------------------------------------------------
shp_path = "sst_analysis_code/counties/counties.shp"  # update path if needed
if os.path.exists(shp_path):
    try:
        reader = shapereader.Reader(shp_path)
        geoms = list(reader.geometries())
        ax.add_geometries(
            geoms,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
    except Exception:
        # If shapefile cannot be read, proceed without it
        pass

# ----------------------------------------------------------------------
# Map extent: focus on Ireland region
# [min_lon, max_lon, min_lat, max_lat] in degrees
# ----------------------------------------------------------------------
ax.set_extent([-12.5, -4.0, 51.0, 55.5], crs=proj)
gl = ax.gridlines(
    draw_labels=True,
    dms=False,
    x_inline=False,
    y_inline=False,
    linewidth=0.4,
    alpha=0.6,
)
gl.top_labels = False
gl.right_labels = False

# ----------------------------------------------------------------------
# Plot sites (triangular markers + text labels)
# ----------------------------------------------------------------------
for p in points:
    # Marker for the location
    ax.plot(
        p["lon_plot"],
        p["lat"],
        marker="^",
        markersize=7,
        markeredgecolor="black",
        markerfacecolor="red",
        transform=proj,
        zorder=5,
    )
    # Slight offset for the text label
    ax.text(
        p["lon_plot"] + 0.1,
        p["lat"] + 0.05,
        p["name"],
        fontsize=9,
        transform=proj,
        zorder=6,
    )

ax.set_title("Ireland – Four Aquaculture Locations", fontsize=12, pad=8)

# ----------------------------------------------------------------------
# Save figure (high resolution, tight bounding box) and show
# ----------------------------------------------------------------------
out_path = "ireland_four_points_cartopy.png"
plt.tight_layout()
plt.savefig(out_path, dpi=400, bbox_inches="tight")
print(f"Saved map to: {out_path}")

plt.show()
