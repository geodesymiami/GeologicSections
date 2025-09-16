#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse, base64, re, warnings
from pathlib import Path

#plotting

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from typing import Union, Literal
from matplotlib import image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MultipleLocator, NullFormatter, FuncFormatter

#mapping
import folium
from branca.colormap import LinearColormap
from folium import FeatureGroup, LayerControl

#--------------------- helpers ---------------------

def normalize_label(label_text: str) -> str:
    """Normalize labels for matching: uppercase, trim, '&'→'AND', 'LIMEROCK'→'LIMESTONE', collapse whitespace."""
    label_text = str(label_text).upper().strip()
    label_text = label_text.replace("&", "AND").replace("LIMEROCK", "LIMESTONE")
    label_text = re.sub(r"\s+", " ", label_text)
    return label_text

def slugify(text: str) -> str:
    """Make an ALL_CAPS slug: keep A–Z/0–9, convert others to underscores, trim leading/trailing underscores."""
    return re.sub(r"[^A-Z0-9]+", "_", str(text).upper()).strip("_")

def to_numeric(value):
    """Convert to numeric; non-parsable values become NaN (pandas.to_numeric with errors='coerce')."""
    return pd.to_numeric(value, errors="coerce")

def find_spt_column(df: pd.DataFrame) -> str | None:
    """Return the (original-cased) column name that stores SPT-N values, or None."""
    if df is None or df.empty:
        return None
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for key in ["spt_n", "n_value", "n_bpf", "n", "sptn", "spt_n_bpf", "spt_blows"]:
        if key in lower_map:
            return lower_map[key]
    return None

# --------------------- pattern PNG discovery ---------------------

def name_candidates(soil_label: str):
    """Generate plausible normalized key variants for a soil/lithology label."""
    base_key = normalize_label(soil_label).replace(" ", "_")
    candidates = {base_key}
    candidates.add(base_key.replace("_(", "_").replace(")", ""))  # LIMESTONE_(FILL) -> LIMESTONE_FILL
    candidates.add(base_key.replace("_FILL", "_(FILL)"))          # LIMESTONE_FILL -> LIMESTONE_(FILL)
    candidates.add(base_key.replace("SHELLS", "SHELL"))
    candidates.add(base_key.replace("SHELL", "SHELLS"))
    candidates.add(base_key.replace("COQUINA_AND_SAND", "SAND_AND_COQUINA"))
    return list(dict.fromkeys(candidates))

def load_textures_index(pattern_dir: Path):
    """Build an index of texture PNGs keyed by UPPERCASE filename stem.

    Example: 'sand.png' -> key 'SAND'
    Raises SystemExit if the directory contains no PNG files.
    """
    files = {path.stem.upper(): path for path in Path(pattern_dir).glob("*.png")}
    if not files:
        raise SystemExit(f"No PNGs found in pattern folder: {pattern_dir}")
    return files

def ensure_rgba_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure an image array is RGBA uint8.
    - If dtype != uint8: scale (assumes 0–1 floats) and cast to uint8.
    - If 2D: replicate to RGB and add opaque alpha.
    - If 3 channels: append opaque alpha.
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 2:  # grayscale -> RGBA
        image = np.stack([image, image, image, np.full_like(image, 255)], axis=-1)
    if image.shape[-1] == 3:
        alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
        image = np.concatenate([image, alpha], axis=-1)
    return image

def resolve_texture(texture_cache: dict, texture_index: dict, soil_label: str, trim_border_px: int = 12):
    """Load a PNG for the given soil label (trying name variants), cache as RGBA uint8, and return it."""
    if soil_label in texture_cache:
        return texture_cache[soil_label]
    for candidate in name_candidates(soil_label):
        path = texture_index.get(candidate)
        if path:
            img = mpimg.imread(path)
            # trim 1-px frame
            if trim_border_px > 0 and img.ndim >= 2 and img.shape[0] > 2*trim_border_px and img.shape[1] > 2*trim_border_px:
                img = img[trim_border_px:-trim_border_px, trim_border_px:-trim_border_px, ...]
            texture_cache[soil_label] = ensure_rgba_uint8(img)
            return texture_cache[soil_label]
    return None

# --------------------- drawing primitives ---------------------

def draw_pattern_block(axes, x_left, y_bottom, width, height, texture_rgba, feet_per_tile: float = 8.0) -> None:
    """Draw a rectangle filled by a vertically tiled PNG texture, clipped to the block bounds."""
    if (texture_rgba is None
        or not np.isfinite(width) or width <= 0
        or not np.isfinite(height) or height <= 0):
        return
    H, W = texture_rgba.shape[:2]

    # how many tiles vertically so each tile ≈ feet_per_tile tall in data space
    n_v = max(1, int(np.ceil(height / float(feet_per_tile))))

    # choose horizontal tiles so the pixel aspect roughly matches the block aspect
    block_aspect   = width / height
    texture_aspect = W / H
    # target: (n_h*W)/(n_v*H) ≈ block_aspect -> n_h ≈ block_aspect * (n_v*H)/W
    n_h = max(1, int(np.ceil(block_aspect * (n_v * H) / float(W))))

    # build tiled image
    tile_texture = np.tile(texture_rgba, (n_v, n_h, 1))

    image = axes.imshow(
        tile_texture,
        extent=(x_left, x_left + width, y_bottom, y_bottom + height),
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        zorder=1,
    )
    clip_vertices = [
        (x_left, y_bottom),
        (x_left + width, y_bottom),
        (x_left + width, y_bottom + height),
        (x_left, y_bottom + height),
        (x_left, y_bottom)
    ]
    image.set_clip_path(MplPath(clip_vertices), axes.transData)

def merge_same_lithology(intervals_df: pd.DataFrame, gap_tolerance_ft: float = 0.05) -> list[dict[str, Union[float, str]]]:
    """Merge consecutive intervals with identical 'lith_key' when (next.top - current.bot) ≤ gap_tolerance_ft (feet)."""
    if intervals_df.empty:
        return []
    sorted_intervals = intervals_df.sort_values("top_ft").reset_index(drop=True)
    merged_intervals = []
    current_top_ft   = float(sorted_intervals.at[0,"top_ft"])
    current_bot_ft   = float(sorted_intervals.at[0,"bot_ft"])
    current_lith     = str(sorted_intervals.at[0,"lith_key"])

    for i in range(1, len(sorted_intervals)):
        next_top_ft  = float(sorted_intervals.at[i,"top_ft"])
        next_bot_ft  = float(sorted_intervals.at[i,"bot_ft"])
        next_lith    = str(sorted_intervals.at[i,"lith_key"])

        if next_lith == current_lith and abs(next_top_ft - current_bot_ft) <= gap_tolerance_ft:
            current_bot_ft = max(current_bot_ft, next_bot_ft)
        else:
            merged_intervals.append({"top_ft":current_top_ft, "bot_ft":current_bot_ft, "lith_key":current_lith})
            current_top_ft, current_bot_ft, current_lith = next_top_ft, next_bot_ft, next_lith

    merged_intervals.append({"top_ft":current_top_ft, "bot_ft":current_bot_ft, "lith_key":current_lith})
    return merged_intervals

# --------------------- sand % ---------------------

def sandy_weight(label_text: str, scoring_mode: str = "weighted") -> float:
    """Return a sand-content score in [0, 1] for a lithology label.
    Modes:
      - 'lenient': any label containing 'SAND' (but not 'CONCRETE') → 1.0
      - 'strict' : only labels starting with 'SAND' → 1.0
      - 'weighted' (default): heuristics for common mixes (e.g., 'SANDSTONE' → 0.6)
    """
    if not isinstance(label_text, str):
        return 0.0
    normalized = normalize_label(label_text)

    if scoring_mode == "lenient":
        return 1.0 if "SAND" in normalized and "CONCRETE" not in normalized else 0.0
    if scoring_mode == "strict":
        return 1.0 if normalized.startswith("SAND") else 0.0
    
    # weighted
    if normalized.startswith("SAND"):                return 1.0
    elif "SANDSTONE" in normalized:                  return 0.6
    elif "CEMENTED SAND" in normalized:              return 0.7
    elif "SAND AND SILT" in normalized:              return 0.7
    elif ("COQUINA AND SAND" in normalized or           
          "SAND AND COQUINA" in normalized):         return 0.7
    elif ("SAND AND SHELL" in normalized or
          "SAND AND SHELLS" in normalized):          return 0.7
    elif "LIMESTONE AND SAND" in normalized:         return 0.3
    elif "CONCRETE" in normalized:                   return 0.0
    return 0.0

def percent_sand_by_boring(intervals_df: pd.DataFrame, scoring_mode: str = "weighted") -> pd.DataFrame:
    """Compute % sand per (project, boring_id) using interval thickness-weighted scores."""
    required = {"project", "boring_id", "top_ft", "bot_ft", "soil_major"}
    missing = required - set(intervals_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    results = []
    for (project, boring_id), group in intervals_df.groupby(["project","boring_id"]):
        segment_thickness_ft = (group["bot_ft"] - group["top_ft"]).clip(lower=0)
        total_thickness_ft = float(segment_thickness_ft.sum())
        weights = group["soil_major"].map(lambda label: sandy_weight(label, scoring_mode))
        sandy_thickness_ft = float((segment_thickness_ft * weights).sum())
        pct_sand = (sandy_thickness_ft / total_thickness_ft * 100.0) if total_thickness_ft > 0 else np.nan
        results.append({"project": project, "boring_id": boring_id, "pct_sand": pct_sand})
    return pd.DataFrame(results)

#--------------------- boring log ---------------------

def plot_panel(intervals_df: pd.DataFrame, ground_elev_ft: float | None, output_png: str | Path,
               texture_cache: dict, texture_index: dict, title: str,
               axis_mode: Literal["depth", "elevation"] = "depth", pattern_scale_ft: float = 14.0) -> str:
    """
    Render a single boring panel with:
      - Depth/elevation axis
      - Lithology track filled with repeating PNG textures
      - Placeholders for water level and well (layout only)

    Parameters
    ----------
    intervals_df : pd.DataFrame
        Rows with columns 'depth_top_ft', 'depth_bot_ft', 'soil_major' for a single boring.
    ground_elev_ft : float or NaN
        Ground surface elevation (used for elevation mode).
    output_png : str or Path
        Output image path.
    texture_cache : dict
        Cache mapping normalized lith keys -> RGBA uint8 arrays.
    texture_index : dict
        Mapping texture name (e.g., 'SAND') -> Path('sand.png').
    title : str
        Panel title (usually boring ID).
    axis_mode : {'depth', 'elevation'}
        Depth mode plots increasing depth downward; elevation mode uses NGVD elevations.
    """

    intervals = intervals_df.copy()
    intervals["top_ft"] = to_numeric(intervals["depth_top_ft"])
    intervals["bot_ft"] = to_numeric(intervals["depth_bot_ft"])
    intervals["lith_key"] = intervals["soil_major"].map(normalize_label)

    if axis_mode == "depth":
        y_top, y_bot = intervals["top_ft"], intervals["bot_ft"]
        ylabel = "Depth (ft)"
        ymin, ymax = 0.0, float(y_bot.max())
        ymin_plot, ymax_plot = ymax, ymin  # inverted axis for depth
    elif axis_mode == "elevation":
        if pd.isna(ground_elev_ft):
            ground_elev_ft = 0.0
        y_top, y_bot = ground_elev_ft - intervals["top_ft"], ground_elev_ft - intervals["bot_ft"]
        ylabel = "Elevation (ft NGVD)"
        ymin, ymax = float(y_bot.min()), float(y_top.max())
        ymin_plot, ymax_plot = ymin, ymax 
    else:
        raise ValueError("axis_mode must be 'depth' or 'elevation'")
    
    merged = merge_same_lithology(pd.DataFrame({
        "top_ft":y_top, 
        "bot_ft":y_bot, 
        "lith_key":intervals["lith_key"],
    }))
    
    data_height = ymax - ymin
    fig_width = 2.3
    fig_height = max(6.5, data_height / 12)
    fig, axs = plt.subplots(1, 4, figsize=(fig_width, fig_height), dpi=260,
                            gridspec_kw={"width_ratios":[0.28, 0.52, 0.08, 0.12], "wspace":0.08})
    ax_depth, ax_lith, ax_water, ax_well = axs
    for ax in axs:
        ax.set_facecolor("white")

    # Depth/elevation axis
    ax_depth.set_xlim(0, 1)
    ax_depth.set_ylim(ymin_plot, ymax_plot)
    ax_depth.set_xticks([])
    ax_depth.yaxis.set_minor_locator(MultipleLocator(1))
    ax_depth.yaxis.set_major_locator(MultipleLocator(5))
    ax_depth.yaxis.set_minor_formatter(NullFormatter())
    ax_depth.yaxis.set_major_formatter(FuncFormatter(lambda v,pos: f"{int(v):d}"))
    ax_depth.tick_params(axis="y", which="major", length=4, width=0.9, labelsize=8)
    ax_depth.tick_params(axis="y", which="minor", length=2, width=0.6)
    ax_depth.set_ylabel(ylabel, fontsize=9, fontweight="bold", labelpad=6)
    for side in ("top","right","left","bottom"):
        ax_depth.spines[side].set_visible(False)
    ax_depth.plot([0.0, 1.0], [ax_depth.get_ylim()[0]] * 2, color="#111", lw=1.0)
    ax_depth.plot([0.0, 1.0], [ax_depth.get_ylim()[1]] * 2, color="#111", lw=1.0)

    # Lithology track
    ax_lith.set_xlim(0, 1)
    ax_lith.set_ylim(ymin_plot, ymax_plot)
    track_x_left, track_width = 0.05, 0.90

    for interval_row in merged:
        y_start, y_end = interval_row["top_ft"], interval_row["bot_ft"]
        if y_end < y_start:
            y_start, y_end = y_end, y_start
        lith_key = interval_row["lith_key"]
        texture_rgba = resolve_texture(texture_cache, texture_index, lith_key)
        if texture_rgba is None:
            texture_rgba = resolve_texture(texture_cache, texture_index, "SAND")

        draw_pattern_block(
            ax_lith,
            track_x_left,
            y_start,
            track_width,
            y_end - y_start,
            texture_rgba,
            feet_per_tile=pattern_scale_ft,
        )

    ax_lith.add_patch(Rectangle(
        (track_x_left, min(ymin_plot, ymax_plot)),
        track_width, abs(ymax_plot - ymin_plot),
        fill=False, edgecolor="#111", linewidth=1.2, zorder=3,
    ))
    
    ax_lith.set_xticks([])
    ax_lith.set_yticks([])
    for side in ("top","right","left","bottom"):
        ax_lith.spines[side].set_visible(False)
    ax_lith.set_title(title, fontsize=11, pad=6, color="#111")

    # to match layout
    for ax in (ax_water, ax_well):
        ax.set_xlim(0,1)
        ax.set_ylim(ymin_plot, ymax_plot)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top","right","left","bottom"):
            ax.spines[side].set_visible(False)

    fig.tight_layout(pad=0.3)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return str(output_png)

# --------------------- read inputs ---------------------

def read_intervals_from_folder(folder_path: Path) -> pd.DataFrame:
    """Read and concatenate all interval CSVs in a folder.

    Requires columns: project, boring_id, depth_top_ft, depth_bot_ft, soil_major.
    """
    if not Path(folder_path).is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder_path}")
    
    csv_paths = sorted([path for path in Path(folder_path).glob("*.csv") if path.is_file()])
    if not csv_paths:
        raise ValueError(f"No interval CSVs found in: {folder_path}")

    required_columns = {"project", "boring_id", "depth_top_ft", "depth_bot_ft", "soil_major"}
    dataframes = []
    for path in csv_paths:
        csv_df = pd.read_csv(path)
        # Normalize headers to avoid issues like ' depth_top_ft'
        csv_df.columns = csv_df.columns.str.strip()

        missing_columns = [c for c in required_columns if c not in csv_df.columns]
        if missing_columns:
            raise ValueError(f"{path.name} missing columns: {missing_columns}")
        
        dataframes.append(csv_df)

    intervals_df = pd.concat(dataframes, ignore_index=True)
    return intervals_df

def read_locations_csv(locations_csv_path: Path) -> pd.DataFrame:
    """Read a locations CSV and return normalized fields.

    Requires columns: building, boring_id, lat, lon, and an elevation column
    (accepted names: 'elevation (ft.)', 'elevation (ft)', 'elevation_ft', 'elev_ft', 'elevation').
    """
    locations_df = pd.read_csv(locations_csv_path)
    locations_df.columns = locations_df.columns.str.strip()

    required_columns = {"building", "boring_id", "lat", "lon"}
    missing_columns = [c for c in required_columns if c not in locations_df.columns]
    if missing_columns:
        raise ValueError(f"Locations file missing columns: {missing_columns}")

    elevation_candidates = {"elevation (ft.)", "elevation (ft)", "elevation_ft", "elev_ft", "elevation"}
    elevation_column = None
    for column in locations_df.columns:
        if str(column).strip().lower() in elevation_candidates:
            elevation_column = column
            break
    if elevation_column is None:
        raise ValueError("Locations file must have an elevation column (e.g., 'elevation (ft.)').")
    
    #normalize values
    locations_df = locations_df.copy()
    locations_df["building"] = locations_df["building"].astype(str).str.strip()
    locations_df["building_key"] = locations_df["building"].map(normalize_label)
    locations_df["boring_id"] = locations_df["boring_id"].astype(str).str.strip()
    locations_df["lat"] = to_numeric(locations_df["lat"]); locations_df["lon"] = to_numeric(locations_df["lon"])
    locations_df["elevation_ft"] = to_numeric(locations_df[elevation_column])

    # company/year
    company_syns = {"company", "company_name", "consultant", "firm", "geotech_company", "geotechnical_company"}
    year_syns    = {"year", "report_year", "date_year"}
    company_col = next((c for c in locations_df.columns if str(c).strip().lower() in company_syns), None)
    year_col    = next((c for c in locations_df.columns if str(c).strip().lower() in year_syns), None)

    if company_col is not None:
        locations_df["company"] = locations_df[company_col].astype(str).str.strip()
    else:
        locations_df["company"] = np.nan

    if year_col is not None:
        # keep text if not strictly numeric; otherwise coerce to int-like string
        y = locations_df[year_col]
        yn = pd.to_numeric(y, errors="coerce")
        locations_df["year"] = np.where(yn.notna(), yn.astype("Int64").astype(str), y.astype(str).str.strip())
    else:
        locations_df["year"] = np.nan

    return locations_df[["building", "building_key", "boring_id", "lat", "lon", "elevation_ft", "company", "year"]]

# --------------------- cross-section ---------------------

def latlon_to_local_xy(lat_deg, lon_deg) -> tuple[np.ndarray, np.ndarray]:
    """Convert latitude/longitude (degrees) to local planar x/y in meters.

    Uses the equirectangular approximation centered at the dataset mean.
    Positive x = east, positive y = north.
    """
    lat_deg = np.asarray(lat_deg, dtype=float)
    lon_deg = np.asarray(lon_deg, dtype=float)

    lat0_deg = np.nanmean(lat_deg)
    lon0_deg = np.nanmean(lon_deg)
    earth_radius_m = 6371000.0

    x_m = np.deg2rad(lon_deg - lon0_deg) * earth_radius_m * np.cos(np.deg2rad(lat0_deg))
    y_m = np.deg2rad(lat_deg - lat0_deg) * earth_radius_m
    return x_m, y_m

def plot_cross_section(project_name: str,
                       project_intervals: pd.DataFrame,
                       texture_cache: dict[str, np.ndarray],
                       texture_index: dict[str, Path],
                       output_png: str | Path,
                       min_gap_ft: float = 18.0,
                       spt_mode: Literal["overlay", "track", "only", "none"] = "overlay",
                       spt_cap: float = 60.0,
                       pattern_scale_ft: float = 14.0) -> str | None:
    """
    Create a lithology cross-section for a project.

    - X axis: borings ordered west→east (by longitude), equal spacing.
    - Y axis: depth (ft)
    - Each boring: a narrow column filled with texture patterns per merged lithology intervals
    - Ground line: connects top-of-hole (TOH) elevations

    Parameters
    ----------
    project_name : str
        Name/label of the project for the figure title.
    project_intervals : pd.DataFrame
        Rows must include: 'boring_id', 'lat', 'lon', 'elevation_ft',
        and lithology intervals per boring: 'depth_top_ft', 'depth_bot_ft', 'soil_major'.
    texture_cache : dict
        Cache mapping normalized lith keys → RGBA uint8 arrays.
    texture_index : dict
        Mapping texture key (e.g., 'SAND') → Path('sand.png').
    output_png : str or Path
        Output image path for the cross-section.
    min_gap_ft : float
        Minimum enforced horizontal gap between adjacent stations (ft).
    
    SPT (Standard Penetration Test) options
    ----------
    spt_mode : {'overlay','track','only','none'}
        How to plot SPT-N. 
        'overlay' draws sized dots on the lithology;
        'track' adds a narrow SPT track next to lithology; 'only' draws SPT track only;
        'none' disables SPT-N plotting.
    spt_cap : float
        Max N used for scaling ('track' bar length and 'overlay' dot size).
    """
    required = {"boring_id","lat","lon","elevation_ft","depth_top_ft","depth_bot_ft","soil_major"}
    missing = required - set(project_intervals.columns)
    if missing:
        raise ValueError(f"Missing columns in project_intervals: {sorted(missing)}")

    # Order west→east and place columns consecutively with no gaps
    COL_WIDTH_FT = max(12.0, float(min_gap_ft))
    boring_summary = (
        project_intervals
        .groupby("boring_id", as_index=False)
        .agg(lat=("lat","first"), lon=("lon","first"), elevation_ft=("elevation_ft","first"))
        .dropna(subset=["lat","lon"])
        .sort_values("lon")
        .reset_index(drop=True)
    )
    if boring_summary.empty:
        warnings.warn(f"[{project_name}] No coordinates; skipping cross section.")
        return None
    boring_summary["station"] = np.arange(len(boring_summary)) * COL_WIDTH_FT

    # build merged blocks (in depth space) per boring
    # intervals per boring (depth space) -> merge adjacent identical lith
    intervals_by_boring = {bid: df.copy() for bid, df in project_intervals.groupby("boring_id")}
    merged_blocks: dict[str, list[dict]] = {}
    for _, row in boring_summary.iterrows():
        bid = row["boring_id"]
        bdf = intervals_by_boring.get(bid, pd.DataFrame()).copy()
        bdf["top_depth_ft"] = to_numeric(bdf["depth_top_ft"])
        bdf["bot_depth_ft"] = to_numeric(bdf["depth_bot_ft"])
        bdf["lith_key"] = bdf["soil_major"].map(normalize_label)
        blocks_df = pd.DataFrame({
            "top_ft":  bdf["top_depth_ft"].astype(float),
            "bot_ft":  bdf["bot_depth_ft"].astype(float),
            "lith_key": bdf["lith_key"].astype(str),
        })
        blocks_df = blocks_df[np.isfinite(blocks_df["top_ft"]) & np.isfinite(blocks_df["bot_ft"])]
        merged_blocks[bid] = merge_same_lithology(blocks_df)

    # SPT: collect points per boring (mid-depth, N)
    spt_points_by_boring: dict[str, pd.DataFrame] = {}
    global_spt_max = 0.0

    for _, row in boring_summary.iterrows():
        bid = row["boring_id"]
        bdf = intervals_by_boring.get(bid, pd.DataFrame()).copy()
        spt_col = find_spt_column(bdf)

        if spt_mode != "none" and spt_col is not None:
            top = pd.to_numeric(bdf["depth_top_ft"], errors="coerce")
            bot = pd.to_numeric(bdf["depth_bot_ft"], errors="coerce")
            nval = pd.to_numeric(bdf[spt_col], errors="coerce")

            mid = (top + bot) / 2.0
            mask = np.isfinite(mid) & np.isfinite(nval)
            pts = pd.DataFrame({"depth_ft": mid[mask].astype(float),
                                "n": nval[mask].astype(float)})
            spt_points_by_boring[bid] = pts.sort_values("depth_ft").reset_index(drop=True)
            if not pts.empty:
                global_spt_max = max(global_spt_max, float(pts["n"].max()))
        else:
            spt_points_by_boring[bid] = pd.DataFrame(columns=["depth_ft","n"])

    # choose scaling for SPT track
    if not np.isfinite(spt_cap) or spt_cap <= 0:
        spt_cap = 60.0
    if global_spt_max > 0:
        spt_cap = max(spt_cap, np.ceil(global_spt_max/5.0)*5.0)

    # depth offset so each boring’s 0-depth sits at its own elevation
    elev = pd.to_numeric(boring_summary.set_index("boring_id")["elevation_ft"], errors="coerce")
    max_ground_elev = float(np.nanmax(elev.values)) if np.isfinite(np.nanmax(elev.values)) else 0.0
    surface_offset_by_boring = ((max_ground_elev - elev).where(np.isfinite(elev), 0.0)).to_dict()

    # compute panel depth needed (max of offset + deepest interval)
    panel_depth = 0.0
    for bid, blocks in merged_blocks.items():
        off = surface_offset_by_boring.get(bid, 0.0)
        if blocks:
            dmax = max(max(b["top_ft"], b["bot_ft"]) for b in blocks)
            panel_depth = max(panel_depth, off + dmax)

    ymin, ymax = 0.0, max(200.0, float(np.ceil(panel_depth / 10.0) * 10.0))

    # figure layout - left: section, right: legend panel
    n_borings = len(boring_summary)
    fig_width = max(8, min(16, 1.6 + 0.9 * n_borings)) # width of the **section** area
    fig_height = max(6.5, (ymax - ymin) / 15.0)
    legend_frac = 0.65  # right panel is 65% of the section width
    total_width = fig_width * (1.0 + legend_frac)
    #fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    #ax = fig.add_subplot(1, 1, 1)
    fig = plt.figure(figsize=(total_width, fig_height), dpi=300)
    gs  = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1.0, legend_frac],   # right column width
        height_ratios=[1.0, 1.0],          # top half (legend), bottom half (empty)
        wspace=0.12, hspace=0.06
    )
    ax         = fig.add_subplot(gs[:, 0])   # section spans both rows on the left
    legend_ax  = fig.add_subplot(gs[0, 1])   # top-right legend panel
    spacer_ax  = fig.add_subplot(gs[1, 1])   # bottom-right empty panel
    spacer_ax.axis("off")

    # Left axis: Depth (ft) increasing downward
    ax.set_ylim(ymax, ymin)
    
    ax.set_xlabel("")
    ax.set_xticks([])                  # no ticks
    ax.tick_params(
        axis="x", which="both",
        bottom=False, top=False,
        labelbottom=False, labeltop=False
    )

    ax.set_ylabel("Depth (ft)", fontsize=10, fontweight="bold",)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_facecolor("white")
    
    for s in ("top","right","left","bottom"):
        ax.spines[s].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    
    ax.spines["bottom"].set_visible(False)

    # right axis: Depth (m)
    FT_TO_M = 0.3048
    ax_m = ax.secondary_yaxis('right',
        functions=(lambda ft: ft * FT_TO_M, lambda m: m / FT_TO_M))
    ax_m.set_ylabel("Depth (m)", fontsize=10, fontweight="bold",)
    ax_m.yaxis.set_major_locator(MultipleLocator(5))
    ax_m.yaxis.set_minor_locator(MultipleLocator(1))

    # ground line (0-depth per boring, offset by elevation)
    ax.plot(
        boring_summary["station"].to_numpy(dtype=float),
        [surface_offset_by_boring[bid] for bid in boring_summary["boring_id"]],
        color="#333", lw=1.2, zorder=3
    )

    # draw each boring
    col_width = COL_WIDTH_FT
    for r in boring_summary.itertuples(index=False):
        bid = r.boring_id
        station = float(r.station)
        x_left = station - col_width / 2.0
        surface_offset = float(surface_offset_by_boring.get(bid, 0.0))
        blocks = merged_blocks.get(bid, [])
        spt_pts = spt_points_by_boring.get(bid)

        # sub-track widths
        gap = 0.04 * col_width
        if spt_mode == "track":
            lith_x_left = x_left
            lith_width  = col_width * 0.68
            spt_x_left  = lith_x_left + lith_width + gap
            spt_width   = col_width - (lith_width + gap)
        elif spt_mode == "only":
            lith_width  = 0.0
            lith_x_left = x_left
            spt_x_left  = x_left + 0.08 * col_width
            spt_width   = col_width * 0.84
        else:  # 'overlay' or 'none' -> full-width lithology
            lith_x_left = x_left
            lith_width  = col_width
            spt_x_left  = x_left + 0.70 * col_width
            spt_width   = col_width * 0.26  # used only for 'overlay' label ticks if desired

        # lithology (skip if 'only')
        if spt_mode != "only":
            for blk in blocks:
                y0 = surface_offset + min(blk["top_ft"], blk["bot_ft"])
                y1 = surface_offset + max(blk["top_ft"], blk["bot_ft"])
                h  = y1 - y0
                if not np.isfinite(h) or h <= 0:
                    continue
                tex = resolve_texture(texture_cache, texture_index, blk["lith_key"])
                if tex is None:
                    tex = resolve_texture(texture_cache, texture_index, "SAND")
                draw_pattern_block(ax, lith_x_left, y0, lith_width, h, tex, feet_per_tile=pattern_scale_ft)

        # SPT plotting
        if spt_mode in ("overlay", "track", "only") and spt_pts is not None and not spt_pts.empty:
            depths = surface_offset + spt_pts["depth_ft"].to_numpy(dtype=float)
            Ns     = np.clip(spt_pts["n"].to_numpy(dtype=float), 0.0, np.inf)

            if spt_mode == "overlay":
                # CURVE over the lithology column: N -> horizontal position
                # Use full lithology width with the same cap-based scaling as 'track'
                # x = lith_x_left + (N / spt_cap) * lith_width
                #pad = 0.03 * lith_width # if want to avoid touching edges
                #x_curve = lith_x_left + pad + (np.minimum(Ns, spt_cap)/float(spt_cap)) * max(lith_width - 2*np.pad, 1e-3)
                x_curve = lith_x_left + (np.minimum(Ns, spt_cap) / float(spt_cap)) * max(lith_width, 1e-3)
                # polyline (sorted by depth)
                ax.plot(x_curve, depths, color="#111", lw=1.6, zorder=7)
            else:
                # CURVE in a dedicated SPT track: N -> horizontal position
                x_right = spt_x_left + (np.minimum(Ns, spt_cap) / float(spt_cap)) * spt_width
                # polyline (sorted by depth)
                ax.plot(x_right, depths, color="#111", lw=1.6, zorder=7)
                # small markers to emphasize measured points
                ax.scatter(x_right, depths, s=12, c="#111", zorder=8)

                # track frame
                ax.add_patch(Rectangle((spt_x_left, surface_offset),
                                       spt_width, ymax - surface_offset,
                                       fill=False, edgecolor="#111", linewidth=0.9, zorder=5))

        # column frame around the whole boring
        ax.add_patch(Rectangle((x_left, surface_offset), col_width, ymax - surface_offset,
                               fill=False, edgecolor="#111", linewidth=1.0, zorder=4))

    # boring labels at a single, consistent level (just above 0-depth)
    label_y = -2.0  # 2 above the top edge
    for r in boring_summary.itertuples(index=False):
        ax.text(float(r.station), label_y, str(r.boring_id),
                ha="center", va="bottom", fontsize=14, color="#111",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
                clip_on=False, transform=ax.transData)

        # elevation label
        el = elev.get(r.boring_id, np.nan)
        if np.isfinite(el):
            ax.text(float(r.station), label_y - 4.0, f"ELEV {el:.1f}'",
                    ha="center", va="bottom", fontsize=10, color="#666",
                    clip_on=False, transform=ax.transData)

    # X limits
    ax.set_xlim(-0.5 * col_width, (n_borings - 0.5) * col_width)

    # legend box (placeholder; separate from section plotting)
    # figure-relative placement: [left, bottom, width, height]
    # tweak if need more/less room needed for legend contents
    legend_ax.set_facecolor("white")
    for side in ("top", "right", "left", "bottom"):
        legend_ax.spines[side].set_visible(True)
        legend_ax.spines[side].set_linewidth(1.0)
        legend_ax.spines[side].set_edgecolor("#111")
    legend_ax.set_xticks([]); legend_ax.set_yticks([])
    legend_ax.text(0.5, 0.5, "Legend", ha="center", va="center",
                   fontsize=10, color="#666", transform=legend_ax.transAxes)
    fig.suptitle(str(project_name), fontsize=24, y=0.995)

    # company/year footer under legend (top of bottom-right spacer)
    comp = None
    yr   = None
    if "company" in project_intervals.columns:
        vals = project_intervals["company"].dropna().astype(str)
        if not vals.empty:
            # choose the most frequent non-null value
            comp = vals.mode().iloc[0]
    if "year" in project_intervals.columns:
        vals = project_intervals["year"].dropna().astype(str)
        if not vals.empty:
            yr = vals.mode().iloc[0]

    if comp or yr:
        label = "COMPANY/YEAR: "
        if comp and yr:    label += f"{comp} - {yr}"
        elif comp:         label += f"{comp}"
        elif yr:           label += f"{yr}"
        # write at the very top of the spacer panel, centered
        spacer_ax.text(0.5, 0.98, label,
                       ha="center", va="top", fontsize=11, color="#111", fontweight="bold",
                       transform=spacer_ax.transAxes)

    fig.tight_layout(pad=0.6)

    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return str(output_png)
# --------------------- map + logs + sections ---------------------

def run(intervals_folder_path, locations_csv_path, pattern_dir_path, output_dir_path,
        sand_mode: str = "weighted", spt_mode: str = "overlay", spt_cap: float = 60.0,
        axis_mode: str = "depth", min_gap_ft: float = 18.0, pattern_scale_ft: float = 14.0,
        ymin_ft: float = 0.0, ymax_ft: float = 200.0, legend_frac: float = 0.55):
    """
    Build an interactive map with boring locations, popups showing each boring log,
    and per-project cross-section images.

    Parameters
    ----------
    intervals_folder_path : str | Path
        Folder containing interval CSVs (requires columns: project, boring_id, depth_top_ft, depth_bot_ft, soil_major).
    locations_csv_path : str | Path
        CSV with boring locations (building, boring_id, lat, lon, elevation, company, year columns).
    pattern_dir_path : str | Path
        Folder with *.png texture patterns (e.g., sand.png).
    output_dir_path : str | Path
        Output folder; creates 'borings/' (logs), 'sections/' (cross-sections), and borings_map.html.
    sand_mode : {'weighted','lenient','strict'}
        Heuristic used by percent_sand_by_boring.
    spt_mode : {'overlay','track','only','none'}
        How to draw SPT N-values.
    spt_cap : float
        Cap for N scaling in curves.
    axis_mode : {'depth','elevation'}
        How to render individual boring logs.
    min_gap_ft : float
        Minimum horizontal spacing enforced between adjacent borings in cross-sections.
    pattern_scale_ft : float
        Vertical feet represented by one pattern tile.
    ymin_ft, ymax_ft : float
        Common depth range (ft) used across plots.
    legend_frac : float
        Legend column width as a fraction of section width (top half only).
    """
    output_dir_path = Path(output_dir_path); output_dir_path.mkdir(parents=True, exist_ok=True)
    borings_dir_path = output_dir_path / "borings"; borings_dir_path.mkdir(parents=True, exist_ok=True)
    sections_dir_path = output_dir_path / "sections"; sections_dir_path.mkdir(parents=True, exist_ok=True)

    # read and normalize data
    intervals_df = read_intervals_from_folder(Path(intervals_folder_path))
    intervals_df["project_key"] = intervals_df["project"].astype(str).str.strip()
    intervals_df["proj_norm"] = intervals_df["project"].map(normalize_label)
    intervals_df["boring_id"] = intervals_df["boring_id"].astype(str).str.strip()
    intervals_df["soil_major"] = intervals_df["soil_major"].astype(str).str.strip()
    intervals_df["top_ft"] = to_numeric(intervals_df["depth_top_ft"])
    intervals_df["bot_ft"] = to_numeric(intervals_df["depth_bot_ft"])

    locations_df = read_locations_csv(Path(locations_csv_path))

    # join on normalized project/building + boring_id
    merged_df = intervals_df.merge(
        locations_df,
        left_on=["proj_norm","boring_id"],
        right_on=["building_key","boring_id"],
        how="left"
    )
    merged_df["building"] = merged_df["building"].fillna(merged_df["project"])

    # textures
    texture_index: dict[str, Path] = load_textures_index(Path(pattern_dir_path))
    texture_cache: dict[str, np.ndarray] = {}

    # sand %
    sand_pct_df = percent_sand_by_boring(merged_df, scoring_mode=sand_mode)
    sand_pct_by_boring = {(r.project, r.boring_id): r.pct_sand for r in sand_pct_df.itertuples(index=False)}

    # map setup
    boring_coords = (
        merged_df.groupby(["project","boring_id"])
        .agg(lat=("lat","first"), lon=("lon","first"))
        .dropna()
        .reset_index()
    )
    if boring_coords.empty:
        raise ValueError("No coordinates to plot (check 'lat/lon' in locations).")

    boring_map = folium.Map(
        location=[float(boring_coords["lat"].mean()), float(boring_coords["lon"].mean())],
        tiles="CartoDB positron",
        zoom_start=14,
        control_scale=True
    )
    colormap = LinearColormap(["#FFFDE7","#FFE082","#FBC02D","#F57F17"], vmin=0, vmax=100)
    colormap.caption = "Sand percentage (%)"
    colormap.add_to(boring_map)

    # per project layers (sorted north -> south, and numbered)
    # order projects by mean latitude (NaNs go last)
    proj_order = (
        merged_df.groupby("project", as_index=False)
                 .agg(mean_lat=("lat", "mean"))
                 .sort_values(["mean_lat","project"], ascending=[False, True], na_position="last")
                 ["project"]
                 .tolist()
    )

    n_projects = len(proj_order)
    # zero-pad so lexicographic sorts stay numeric
    pad = max(2, len(str(n_projects)))

    for idx, project in enumerate(proj_order, start=1):
        project_df = merged_df[merged_df["project"] == project]
        layer_group = FeatureGroup(name=str(project), show=True)

        # numbered section filename
        section_png_path = sections_dir_path / f"{idx:0{pad}d}_section_{slugify(project)}.png"

        # cross-section
        section_png = plot_cross_section(
            project, project_df,
            texture_cache, texture_index,
            section_png_path,
            min_gap_ft=min_gap_ft,
            spt_mode=spt_mode,
            spt_cap=spt_cap,
            pattern_scale_ft=pattern_scale_ft,
        )

        # markers and individual logs
        for boring_id, boring_df in project_df.groupby("boring_id"):
            if not boring_df["lat"].notna().any() or not boring_df["lon"].notna().any():
                continue

            lat = float(boring_df["lat"].dropna().iloc[0])
            lon = float(boring_df["lon"].dropna().iloc[0])
            elev_ft = (
                boring_df["elevation_ft"].dropna().iloc[0]
                if boring_df["elevation_ft"].notna().any()
                else np.nan
            )

            log_png_path = borings_dir_path / f"log_{slugify(project)}_{slugify(boring_id)}.png"
            plot_panel(
                boring_df[["depth_top_ft", "depth_bot_ft", "soil_major"]],
                elev_ft, log_png_path, texture_cache, texture_index,
                title=str(boring_id), axis_mode=axis_mode, pattern_scale_ft=pattern_scale_ft
            )

            pct = float(sand_pct_by_boring.get((project, boring_id), np.nan))
            color = colormap(pct if np.isfinite(pct) else 0.0)
            pct_txt = "n/a" if not np.isfinite(pct) else f"{pct:.1f}%"

            # embed images in popup
            with open(log_png_path, "rb") as f:
                b64_log = base64.b64encode(f.read()).decode("ascii")

            section_embed_html = ""
            if section_png and Path(section_png).exists():
                with open(section_png, "rb") as f:
                    b64_sec = base64.b64encode(f.read()).decode("ascii")
                section_embed_html = (
                    f'<div style="margin-top:6px">'
                    f'<img src="data:image/png;base64,{b64_sec}" '
                    f'style="width:360px; height:auto; border:1px solid #888;"/></div>'
                )

            popup_html = f"""
            <div style="font-family:system-ui,Arial,sans-serif; font-size:12px;">
              <b>{project}</b><br>
              Boring: <b>{boring_id}</b><br>
              Sand: <b>{pct_txt}</b><br>
              Elev (ft NGVD): <b>{('%.2f' % elev_ft) if np.isfinite(elev_ft) else 'N/A'}</b><br><br>
              <img src="data:image/png;base64,{b64_log}" style="width:240px; height:auto; border:1px solid #888;"/>
              {section_embed_html}
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=6, weight=1, color="#333",
                fill=True, fill_color=color, fill_opacity=0.95,
                tooltip=f"{project} / {boring_id} — {pct_txt}",
                popup=folium.Popup(folium.IFrame(html=popup_html, width=520, height=640), max_width=560),
            ).add_to(layer_group)

        layer_group.add_to(boring_map)

    LayerControl(collapsed=False).add_to(boring_map)
    boring_map.fit_bounds(
        [[boring_coords["lat"].min(), boring_coords["lon"].min()],
         [boring_coords["lat"].max(), boring_coords["lon"].max()]]
    )
    
    out_html_path = Path(output_dir_path) / "borings_map.html"
    boring_map.save(str(out_html_path))
    print("[INFO] Map:", out_html_path)
    print("[INFO] Logs folder:", borings_dir_path)
    print("[INFO] Sections folder:", sections_dir_path)

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description="Map + logs + cross-sections")
    ap.add_argument("--intervals-folder", required=True, help="Folder contains CSVs of spt intervals")
    ap.add_argument("--locations", required=True, help="CSV file of boring locations")
    ap.add_argument("--pattern-dir", required=True, help="Folder with pattern tiles - PNG")
    ap.add_argument("--outdir", default="geosec_out", help="Output directory")
    ap.add_argument("--sand_mode", dest="sand_mode",
                    choices=["weighted","strict","lenient"], default="weighted",
                    help="Heuristic used by percent_sand_by_boring")
    ap.add_argument("--axis_mode", choices=["depth","elevation"], default="depth")
    ap.add_argument("--spt-mode", choices=["overlay","track","only","none"], default="overlay")
    ap.add_argument("--spt-cap", type=float, default=60.0)
    ap.add_argument("--min-gap-ft", type=float, default=18.0,
                    help="Minimum horizontal gap between adjacent borings in section view -ft")
    ap.add_argument("--pattern-scale-ft", type=float, default=8.0,
                    help="Vertical feet represented by one pattern tile; larger -> bigger dots/lines")
    ap.add_argument("--ymin-ft", type=float, default=0.0, help="Minimum depth (ft)")
    ap.add_argument("--ymax-ft", type=float, default=200.0, help="Maximum depth (ft)")
    ap.add_argument("--legend-frac", type=float, default=0.55,
                    help="Legend column width as a fraction of section width (0.55 means 55%)")
    args = ap.parse_args()

    run(args.intervals_folder, args.locations, args.pattern_dir, args.outdir,
        sand_mode=args.sand_mode, axis_mode=args.axis_mode, min_gap_ft=args.min_gap_ft,
        spt_mode=args.spt_mode, spt_cap=args.spt_cap, pattern_scale_ft=args.pattern_scale_ft,
        ymin_ft=args.ymin_ft, ymax_ft=args.ymax_ft, legend_frac=args.legend_frac)


if __name__ == "__main__":
    main()

