#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse, base64, re, warnings
from pathlib import Path

#plotting
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MultipleLocator, NullFormatter, FuncFormatter

#mapping
import folium
from branca.colormap import LinearColormap
from folium import FeatureGroup, LayerControl, IFrame

#--------------------- helpers ---------------------

def nkey(s: str) -> str:
    """Normalize labels for matching."""
    s = str(s).upper().strip()
    s = s.replace("&", "AND").replace("LIMEROCK", "LIMESTONE")
    s = re.sub(r"\s+", " ", s)
    return s

def slug(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

# --------------------- pattern PNG discovery ---------------------

def name_candidates(soil_key: str):
    k = nkey(soil_key).replace(" ", "_")
    c = {k}
    c.add(k.replace("_(", "_").replace(")", ""))  # LIMESTONE_(FILL) -> LIMESTONE_FILL
    c.add(k.replace("_FILL", "_(FILL)"))          # LIMESTONE_FILL -> LIMESTONE_(FILL)
    c.add(k.replace("SHELLS", "SHELL"))
    c.add(k.replace("SHELL", "SHELLS"))
    c.add(k.replace("COQUINA_AND_SAND", "SAND_AND_COQUINA"))
    return list(dict.fromkeys(c))

def load_textures_index(pattern_dir: Path):
    files = {p.stem.upper(): p for p in Path(pattern_dir).glob("*.png")}
    if not files:
        raise SystemExit(f"No PNGs found in pattern folder: {pattern_dir}")
    return files

def ensure_rgba_uint8(arr):
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:  # grayscale -> RGBA
        arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
    if arr.shape[-1] == 3:
        alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=-1)
    return arr

def resolve_texture(cache: dict, files_index: dict, soil_key: str):
    """load  PNG for soil_key using candidates (cache RGBA array)s."""
    if soil_key in cache:
        return cache[soil_key]
    for cand in name_candidates(soil_key):
        p = files_index.get(cand)
        if p:
            img = mpimg.imread(p)
            cache[soil_key] = ensure_rgba_uint8(img)
            return cache[soil_key]
    return None

# --------------------- drawing primitives ---------------------

def draw_pattern_block(ax, x, y0, w, h, rgba_tex, ft_per_tile=4.0):
    """Tile a PNG vertically and clip to block rectangle."""
    if h <= 0 or rgba_tex is None:
        return
    reps = max(1, int(np.ceil(h / ft_per_tile)))
    tile = np.vstack([rgba_tex] * reps)
    im = ax.imshow(tile, extent=(x, x+w, y0, y0+h),
                   origin="lower", interpolation="nearest",
                   aspect="auto", zorder=1)
    verts = [(x,y0),(x+w,y0),(x+w,y0+h),(x,y0+h),(x,y0)]
    im.set_clip_path(MplPath(verts), ax.transData)

def merge_same_lithology(depth_df: pd.DataFrame, tol: float = 0.05):
    """Merge consecutive intervals with same 'lith_key' if next.top - prev.bot <= tol (ft)."""
    if depth_df.empty: return []
    s = depth_df.sort_values("top_ft").reset_index(drop=True)
    out = []
    cur_t, cur_b, cur_l = float(s.at[0,"top_ft"]), float(s.at[0,"bot_ft"]), str(s.at[0,"lith_key"])
    for i in range(1, len(s)):
        t, b, l = float(s.at[i,"top_ft"]), float(s.at[i,"bot_ft"]), str(s.at[i,"lith_key"])
        if l == cur_l and abs(t - cur_b) <= tol:
            cur_b = max(cur_b, b)
        else:
            out.append({"top_ft":cur_t, "bot_ft":cur_b, "lith_key":cur_l})
            cur_t, cur_b, cur_l = t, b, l
    out.append({"top_ft":cur_t, "bot_ft":cur_b, "lith_key":cur_l})
    return out

# --------------------- sand % ---------------------

def sandy_weight(label: str, mode: str = "weighted") -> float:
    if not isinstance(label, str): return 0.0
    u = nkey(label)
    if mode == "lenient":
        return 1.0 if "SAND" in u and "CONCRETE" not in u else 0.0
    if mode == "strict":
        return 1.0 if u.startswith("SAND") else 0.0
    # weighted
    if u.startswith("SAND"):              return 1.0
    if "SANDSTONE" in u:                  return 0.6
    if "CEMENTED SAND" in u:              return 0.7
    if "SAND AND SILT" in u:              return 0.7
    if "COQUINA AND SAND" in u:           return 0.7
    if "SAND AND SHELL" in u:             return 0.7
    if "LIMESTONE AND SAND" in u:         return 0.3
    if "CONCRETE" in u:                   return 0.0
    return 0.0

def percent_sand_by_boring(df: pd.DataFrame, mode: str = "weighted") -> pd.DataFrame:
    rows = []
    for (proj, bid), g in df.groupby(["project","boring_id"]):
        seg = (g["bot_ft"] - g["top_ft"]).clip(lower=0)
        total = float(seg.sum())
        w = g["soil_major"].map(lambda s: sandy_weight(s, mode))
        sandy = float((seg * w).sum())
        pct = (sandy/total*100.0) if total>0 else np.nan
        rows.append({"project":proj, "boring_id":bid, "pct_sand":pct})
    return pd.DataFrame(rows)

#--------------------- boring log ---------------------

def plot_panel(g: pd.DataFrame, elev_ft, out_png, textures_cache: dict,
                     textures_index: dict, title, axis_mode="depth"):
    """
    Depth - Lithology - water (will be added) - well (will be added)
    """
    g2 = g.copy()
    g2["top_ft"] = to_num(g2["depth_top_ft"])
    g2["bot_ft"] = to_num(g2["depth_bot_ft"])
    g2["lith_key"] = g2["soil_major"].map(nkey)

    if axis_mode == "depth":
        y_top, y_bot = g2["top_ft"], g2["bot_ft"]
        ylabel = "Depth (ft)"
        ymin, ymax = 0.0, float(y_bot.max())
        ylo, yhi = ymax, ymin  # for set_ylim
    else:
        if pd.isna(elev_ft): elev_ft = 0.0
        y_top, y_bot = elev_ft - g2["top_ft"], elev_ft - g2["bot_ft"]
        ylabel = "Elevation (ft NGVD)"
        ymin, ymax = float(y_bot.min()), float(y_top.max())
        ylo, yhi = ymin, ymax

    merged = merge_same_lithology(pd.DataFrame({"top_ft":y_top, "bot_ft":y_bot, "lith_key":g2["lith_key"]}))
    data_height = ymax - ymin

    fig_w = 2.3
    fig_h = max(6.5, data_height / 12)
    fig, axs = plt.subplots(1, 4, figsize=(fig_w, fig_h), dpi=260,
                            gridspec_kw={"width_ratios":[0.28, 0.52, 0.08, 0.12], "wspace":0.08})
    ax_depth, ax_lith, ax_wl, ax_well = axs
    for a in axs: a.set_facecolor("white")

    # Depth axis
    ax_depth.set_xlim(0,1); ax_depth.set_ylim(ylo, yhi)
    ax_depth.set_xticks([])
    ax_depth.yaxis.set_minor_locator(MultipleLocator(1))
    ax_depth.yaxis.set_major_locator(MultipleLocator(5))
    ax_depth.yaxis.set_minor_formatter(NullFormatter())
    ax_depth.yaxis.set_major_formatter(FuncFormatter(lambda v,pos: f"{int(v):d}"))
    ax_depth.tick_params(axis="y", which="major", length=4, width=0.9, labelsize=8)
    ax_depth.tick_params(axis="y", which="minor", length=2, width=0.6)
    ax_depth.set_ylabel(ylabel, fontsize=9, labelpad=6)
    for s in ("top","right","left","bottom"): ax_depth.spines[s].set_visible(False)
    ax_depth.plot([0.0,1.0],[ax_depth.get_ylim()[0]]*2, color="#111", lw=1.0)
    ax_depth.plot([0.0,1.0],[ax_depth.get_ylim()[1]]*2, color="#111", lw=1.0)

    # Lithology track
    ax_lith.set_xlim(0,1); ax_lith.set_ylim(ylo, yhi)
    x0, w = 0.05, 0.90
    for r in merged:
        yb0, yb1 = r["top_ft"], r["bot_ft"]
        if yb1 < yb0: yb0, yb1 = yb1, yb0
        key = r["lith_key"]
        tex = resolve_texture(textures_cache, textures_index, key)
        if tex is None:
            tex = resolve_texture(textures_cache, textures_index, "SAND")
        draw_pattern_block(ax_lith, x0, yb0, w, yb1-yb0, tex, ft_per_tile=4.0)
    ax_lith.add_patch(Rectangle((x0, min(ylo,yhi)), w, abs(yhi-ylo),
                                fill=False, edgecolor="#111", linewidth=1.2, zorder=3))
    ax_lith.set_xticks([]); ax_lith.set_yticks([])
    for s in ("top","right","left","bottom"): ax_lith.spines[s].set_visible(False)
    ax_lith.set_title(title, fontsize=11, pad=6, color="#111")

    # Placeholders to match layout
    for ax in (ax_wl, ax_well):
        ax.set_xlim(0,1); ax.set_ylim(ylo, yhi); ax.set_xticks([]); ax.set_yticks([])
        for s in ("top","right","left","bottom"): ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)

# --------------------- read inputs ---------------------

def read_intervals_from_folder(folder: Path) -> pd.DataFrame:
    files = sorted([p for p in Path(folder).glob("*.csv") if p.is_file()])
    if not files:
        raise SystemExit(f"No interval CSVs found in: {folder}")
    frames = []
    for p in files:
        df = pd.read_csv(p)
        need = {"project","boring_id","depth_top_ft","depth_bot_ft","soil_major"}
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise SystemExit(f"{p.name} missing columns: {miss}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out

def read_locations_csv(loc_csv: Path) -> pd.DataFrame:
    loc = pd.read_csv(loc_csv)
    need = {"building","boring_id","lat","lon"}
    miss = [c for c in need if c not in loc.columns]
    if miss:
        raise SystemExit(f"Locations file missing columns: {miss}")
    elev_col = None
    for c in loc.columns:
        if str(c).strip().lower() in {"elevation (ft.)","elevation (ft)","elevation_ft","elev_ft","elevation"}:
            elev_col = c; break
    if elev_col is None:
        raise SystemExit("Locations file must have an elevation column (e.g., 'elevation (ft.)').")
    loc = loc.copy()
    loc["building"] = loc["building"].astype(str).str.strip()
    loc["build_norm"] = loc["building"].map(nkey)
    loc["boring_id"] = loc["boring_id"].astype(str).str.strip()
    loc["lat"] = to_num(loc["lat"]); loc["lon"] = to_num(loc["lon"])
    loc["elevation_ft"] = to_num(loc[elev_col])
    return loc[["building","build_norm","boring_id","lat","lon","elevation_ft"]]

# --------------------- cross-section ---------------------

def latlon_to_local_xy(lat, lon):
    """Convert lat/lon arrays to local meters using equirectangular approx."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = np.nanmean(lat)
    R = 6371000.0
    x = np.deg2rad(lon - np.nanmean(lon)) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R
    return x, y

def stations_along_pca(lat, lon):
    """Project points onto first principal component to get station (meters)."""
    x, y = latlon_to_local_xy(lat, lon)
    pts = np.column_stack([x, y])
    pts = pts - np.nanmean(pts, axis=0)
    # SVD for principal axis
    U, S, Vt = np.linalg.svd(np.nan_to_num(pts), full_matrices=False)
    v = Vt[0]  #direction
    s = pts @ v
    #shift to start at 0
    s = s - np.nanmin(s)
    return s  #meters

def plot_cross_section(proj_name, gproj, textures_cache, textures_index, out_png,
                       min_gap_ft=18.0):
    """
    Create a cross-section for a project:
    - x-axis: station (ft) along principal line through borings
    - y-axis: elevation (ft-NGVD)
    - each boring drawn as a narrow column filled with lithology patterns
    - ground line connecting TOH elevations
    """
    # one row per boring for coords/elev
    heads = (gproj.groupby("boring_id")
                  .agg(lat=("lat","first"), lon=("lon","first"),
                       elevation_ft=("elevation_ft","first"))
                  .dropna(subset=["lat","lon"]))
    if heads.empty:
        warnings.warn(f"[{proj_name}] No coordinates; skipping cross section.")
        return None

    # stations in feet
    st_m = stations_along_pca(heads["lat"].values, heads["lon"].values)
    st_ft = st_m * 3.28084
    heads = heads.assign(station_ft=st_ft)
    heads = heads.sort_values("station_ft").reset_index()

    # minimum gap between adjacent stations (in feet) ----
    # tweak for wider spacing
    MIN_GAP_FT = float(min_gap_ft)

    x = heads["station_ft"].to_numpy(dtype=float)
    if len(x) >= 2:
        x_adj = x.copy()
        for i in range(1, len(x_adj)):
            if x_adj[i] - x_adj[i-1] < MIN_GAP_FT:
                x_adj[i] = x_adj[i-1] + MIN_GAP_FT
        # keep the mean position unchanged (recentre)
        x_adj -= (x_adj.mean() - x.mean())
    else:
        x_adj = x
    heads["station_adj"] = x_adj

    # figure Y-limits (elevation)
    # bottoms per boring
    merged_blocks = {}  # boring_id -> list of merged blocks in elevation
    for _, row in heads.iterrows():
        bid  = row["boring_id"]
        elev = float(row["elevation_ft"]) if pd.notna(row["elevation_ft"]) else 0.0
        gb = gproj[gproj["boring_id"] == bid].copy()
        gb["top_elev"] = elev - to_num(gb["depth_top_ft"])
        gb["bot_elev"] = elev - to_num(gb["depth_bot_ft"])
        gb["lith_key"] = gb["soil_major"].map(nkey)

        # frame (no duplicate column names)
        blocks_df = pd.DataFrame({
            "top_ft":  gb["top_elev"].astype(float),
            "bot_ft":  gb["bot_elev"].astype(float),
            "lith_key": gb["lith_key"].astype(str),
        })
        merged_blocks[bid] = merge_same_lithology(blocks_df)

    # Robust Y limits (handles NaNs, empty blocks) ----
    toh_vals = heads["elevation_ft"].to_numpy(dtype=float)

    # bottoms from merged blocks
    bot_vals_list = [min(b["top_ft"], b["bot_ft"])
                 for blocks in merged_blocks.values() for b in blocks]
    bot_vals = np.array(bot_vals_list, dtype=float) if bot_vals_list else np.array([], float)

    # if all TOH are NaN, fall back to block tops (or 0.0)
    if not np.isfinite(toh_vals).any():
        tops_from_blocks = [max(b["top_ft"], b["bot_ft"])
                            for blocks in merged_blocks.values() for b in blocks]
        toh_vals = np.array(tops_from_blocks if tops_from_blocks else [0.0], dtype=float)

    max_toh = float(np.nanmax(toh_vals)) if toh_vals.size else 0.0
    min_bot = float(np.nanmin(bot_vals)) if bot_vals.size else (max_toh - 10.0)

    pad = 2.0
    ymin, ymax = min_bot - pad, max_toh + pad + 2.0
    if not (np.isfinite(ymin) and np.isfinite(ymax)) or ymin == ymax:
        ymin, ymax = -10.0, 10.0

    # layout
    n = len(heads)
    fig_w = max(8, min(16, 1.6 + 0.9 * n))
    fig_h = max(6.5, (ymax - ymin) / 15.0)

    # layout (single-panel -no legend)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=180)
    ax = fig.add_subplot(1,1,1)
    ax.set_facecolor("white")
    for s in ("top","right","left","bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.1)

    # draw ground line
    ax.plot(heads["station_adj"], heads["elevation_ft"], color="#333", lw=1.2, zorder=3)

    # draw each boring
    span = float(heads["station_adj"].max() - heads["station_adj"].min()) if len(heads) else 0.0
    col_w = max(6.0, min(18.0, 0.55 * span / max(1, len(heads))))
    col_w = min(col_w, 0.8 * MIN_GAP_FT)  #never wider than the enforced gap

    for _, row in heads.iterrows():
        bid  = row["boring_id"]
        st   = float(row["station_adj"])   #use adjusted station
        elev = float(row["elevation_ft"])
        blocks = merged_blocks.get(bid, [])
        x0 = st - col_w/2.0
        for b in blocks:
            y0 = min(b["top_ft"], b["bot_ft"])
            y1 = max(b["top_ft"], b["bot_ft"])
            tex = resolve_texture(textures_cache, textures_index, b["lith_key"])
            if tex is None:
                tex = resolve_texture(textures_cache, textures_index, "SAND")
            draw_pattern_block(ax, x0, y0, col_w, y1 - y0, tex, ft_per_tile=4.0)

        # frame
        ax.add_patch(Rectangle((x0, ymin), col_w, ymax - ymin,
                               fill=False, edgecolor="#111", linewidth=1.0, zorder=4))
        # label
        ax.text(
            st, ymax + 1.0, str(bid), ha="center", va="bottom", fontsize=9, color="#111",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
            clip_on=False, transform=ax.transData
        )

    # axes (robust for single-boring / NaNs)
    x_min = float(heads["station_adj"].min()) if len(heads) else 0.0
    x_max = float(heads["station_adj"].max()) if len(heads) else 1.0
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        x_min, x_max = 0.0, 1.0

    if x_min == x_max:
        pad_x = 15.0  # ft padding for single boring
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
    else:
        ax.set_xlim(x_min - col_w, x_max + col_w)

    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Station (ft)", fontsize=10)
    ax.set_ylabel("Elevation (ft NGVD)", fontsize=10)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # keep a full frame around the cross-section
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.suptitle(str(proj_name), fontsize=14, y=0.995)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)
# --------------------- map + logs + sections orchestrator ---------------------

def run(intervals_folder, locations_csv, pattern_dir, out_dir,
        sand_mode="weighted", axis_mode="depth", min_gap_ft=18.0):

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "borings"; png_dir.mkdir(parents=True, exist_ok=True)
    sec_dir = out_dir / "sections"; sec_dir.mkdir(parents=True, exist_ok=True)

    intervals = read_intervals_from_folder(Path(intervals_folder))
    intervals = intervals.copy()
    intervals["project"] = intervals["project"].astype(str).str.strip()
    intervals["proj_norm"] = intervals["project"].map(nkey)
    intervals["boring_id"] = intervals["boring_id"].astype(str).str.strip()
    intervals["soil_major"] = intervals["soil_major"].astype(str).str.strip()
    intervals["top_ft"] = to_num(intervals["depth_top_ft"])
    intervals["bot_ft"] = to_num(intervals["depth_bot_ft"])

    loc = read_locations_csv(Path(locations_csv))

    # normalized project/building + boring_id
    data = intervals.merge(
        loc,
        left_on=["proj_norm","boring_id"],
        right_on=["build_norm","boring_id"],
        how="left"
    )
    data["building"] = data["building"].fillna(data["project"])

    # textures
    files_index = load_textures_index(Path(pattern_dir))
    textures_cache = {}

    # sand %
    pctdf = percent_sand_by_boring(data, mode=sand_mode)
    pct_map = {(r.project, r.boring_id): r.pct_sand for r in pctdf.itertuples(index=False)}

    # map setup
    coords = (
        data.groupby(["project","boring_id"]).agg(lat=("lat","first"), lon=("lon","first")).dropna().reset_index()
    )
    if coords.empty:
        raise SystemExit("No coordinates to plot (check 'lat/lon' in locations).")

    m = folium.Map(
        location=[float(coords["lat"].mean()), float(coords["lon"].mean())],
        tiles="CartoDB positron", zoom_start=14, control_scale=True
    )
    colormap = LinearColormap(["#FFFDE7","#FFE082","#FBC02D","#F57F17"], vmin=0, vmax=100)
    colormap.caption = "Sand percentage (%)"
    colormap.add_to(m)

    # loop per project
    for proj, gproj in data.groupby("project"):
        fg = FeatureGroup(name=str(proj), show=True)

        # cross-section for that project
        sec_png = plot_cross_section(
            proj, gproj, textures_cache, files_index,
            sec_dir / f"section_{slug(proj)}.png",
            min_gap_ft=min_gap_ft
        )


        # markers and individual logs
        for bid, gbor in gproj.groupby("boring_id"):
            if not gbor["lat"].notna().any() or not gbor["lon"].notna().any():
                continue
            la = float(gbor["lat"].dropna().iloc[0])
            lo = float(gbor["lon"].dropna().iloc[0])
            elev_ft = gbor["elevation_ft"].dropna().iloc[0] if gbor["elevation_ft"].notna().any() else np.nan

            png_path = png_dir / f"log_{slug(proj)}_{slug(bid)}.png"
            plot_panel(
                gbor[["depth_top_ft","depth_bot_ft","soil_major"]],
                elev_ft, png_path, textures_cache, files_index,
                title=str(bid), axis_mode=axis_mode
            )

            pct = float(pct_map.get((proj, bid), np.nan))
            color = colormap(pct if np.isfinite(pct) else 0.0)
            pct_txt = "n/a" if not np.isfinite(pct) else f"{pct:.1f}%"

            with open(png_path, "rb") as f:
                b64_log = base64.b64encode(f.read()).decode("ascii")

            sec_embed = ""
            if sec_png and Path(sec_png).exists():
                with open(sec_png, "rb") as f:
                    b64_sec = base64.b64encode(f.read()).decode("ascii")
                sec_embed = f'<div style="margin-top:6px"><img src="data:image/png;base64,{b64_sec}" style="width:360px; height:auto; border:1px solid #888;"/></div>'

            html = f"""
            <div style="font-family:system-ui,Arial,sans-serif; font-size:12px;">
              <b>{proj}</b><br>
              Boring: <b>{bid}</b><br>
              Sand: <b>{pct_txt}</b><br>
              Elev (ft NGVD): <b>{('%.2f' % elev_ft) if np.isfinite(elev_ft) else 'N/A'}</b><br><br>
              <img src="data:image/png;base64,{b64_log}" style="width:240px; height:auto; border:1px solid #888;"/>
              {sec_embed}
            </div>
            """

            folium.CircleMarker(
                location=[la, lo],
                radius=6, weight=1, color="#333",
                fill=True, fill_color=color, fill_opacity=0.95,
                tooltip=f"{proj} / {bid} — {pct_txt}",
                popup=folium.Popup(IFrame(html=html, width=520, height=640), max_width=560),
            ).add_to(fg)

        fg.add_to(m)

    LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[coords["lat"].min(), coords["lon"].min()],
                  [coords["lat"].max(), coords["lon"].max()]])
    out_html = Path(out_dir) / "borings_map.html"
    m.save(str(out_html))
    print("[INFO] Map:", out_html)
    print("[INFO] Logs folder:", png_dir)
    print("[INFO] Sections folder:", sec_dir)

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description="Map + logs + cross-sections")
    ap.add_argument("--intervals-folder", required=True, help="Folder contains CSVs of spt intervals")
    ap.add_argument("--locations", required=True, help="CSV file of boring locations")
    ap.add_argument("--pattern-dir", required=True, help="Folder with pattern tiles - PNG")
    ap.add_argument("--outdir", default="Output directory")
    ap.add_argument("--sand_mode", choices=["weighted","strict","lenient"], default="weighted")
    ap.add_argument("--axis_mode", choices=["depth","elevation"], default="depth")
    ap.add_argument("--min-gap-ft", type=float, default=18.0,
                    help="Minimum horizontal gap between adjacent borings in section view -ft")

    args = ap.parse_args()

    run(args.intervals_folder, args.locations, args.pattern_dir, args.outdir,
        sand_mode=args.sand_mode, axis_mode=args.axis_mode, min_gap_ft=args.min_gap_ft)


if __name__ == "__main__":
    main()

