

import pandas as pd
import numpy as np
import folium
from folium import FeatureGroup, LayerControl
from branca.colormap import LinearColormap
from pathlib import Path
import argparse, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def slug(s): 
    return re.sub(r"[^A-Z0-9]+","_",str(s).upper()).strip("_")

def haversine_km(lat1, lon1, lat2, lon2):
    # vectorized haversine (kilometers)
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arctan2(np.sqrt(np.clip(a,0,1)), np.sqrt(np.clip(1-a,0,1)))
    return R*c

def is_sandy(label: str) -> bool:
    if not isinstance(label, str): return False
    u = label.upper()
    if "CONCRETE" in u: return False
    return "SAND" in u  #includes combos like SAND AND SILT, CEMENTED SAND, etc.

def parse_n_value(x, n_scale=60.0):
    if x is None: return np.nan
    if isinstance(x, (int,float)) and not np.isnan(x): return float(x)
    s = str(x).strip().upper()
    if s in {"", "NAN", "NA", "N/A", "--", "-"}: return np.nan
    if "WOR" in s: return 0.0
    if "WOH" in s: return np.nan
    if "REFUS" in s: return n_scale
    m = re.match(r'^\s*(\d+)\s*/\s*(\d+)"?\s*$', s)
    if m: return float(m.group(1))
    m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$', s)
    if m: return 0.5*(float(m.group(1))+float(m.group(2)))
    try: return float(s)
    except: return np.nan

#-------------------- plot style maps --------------------
COLOR_MAP = {
    "SAND": "#FFD54F",
    "SILTY SAND": "#FBC02D",
    "SILT": "#BDBDBD",
    "PEAT": "#6D4C41",
    "LIMESTONE": "#80CBC4",
    "SANDSTONE": "#E0B070",
    "CEMENTED SAND": "#CDAA7D",
    "SAND AND CEMENTED SAND": "#D7B98B",
    "SAND AND SILT": "#E6D27A",
    "SAND AND SHELLS": "#F4E1A9",
    "SAND AND SANDSTONE": "#E8C07D",
    "LIMESTONE AND SAND": "#A5D6C4",
    "LIMESTONE AND SAND (FILL)": "#CFEBE6",
    "CONCRETE (FILL)": "#DCDCDC",
    "SAND AND LIMEROCK (FILL)": "#CFD8DC",
}
HATCH_MAP = {
    "SAND": ".",
    "SILTY SAND": "x",
    "SILT": "-",
    "PEAT": "///",
    "LIMESTONE": "||",
    "SANDSTONE": "++",
    "CEMENTED SAND": "xx",
    "SAND AND CEMENTED SAND": "x.",
    "SAND AND SILT": "-.",
    "SAND AND SHELLS": "o",
    "SAND AND SANDSTONE": "x+",
    "LIMESTONE AND SAND": "|.",
    "LIMESTONE AND SAND (FILL)": "|..",
    "CONCRETE (FILL)": "////",
    "SAND AND LIMEROCK (FILL)": "o-",
}
DEFAULT_COLOR = "#CCCCCC"
DEFAULT_HATCH = "..."

def merge_same_lithology(sub_df, tol=1e-2):
    rows = []
    if sub_df.empty: return rows
    s = sub_df.sort_values("top_ft")
    cur_top = float(s.iloc[0]["top_ft"])
    cur_bot = float(s.iloc[0]["bot_ft"])
    cur_lith = str(s.iloc[0].get("soil_major",""))
    for _, r in s.iloc[1:].iterrows():
        t = float(r["top_ft"]); b = float(r["bot_ft"]); lith = str(r.get("soil_major",""))
        if lith == cur_lith and abs(t - cur_bot) <= tol:
            cur_bot = max(cur_bot, b)
        else:
            rows.append({"top_ft": cur_top, "bot_ft": cur_bot, "soil_major": cur_lith})
            cur_top, cur_bot, cur_lith = t, b, lith
    rows.append({"top_ft": cur_top, "bot_ft": cur_bot, "soil_major": cur_lith})
    return rows

#-------------------- core functions --------------------
def assign_building_to_borings(intervals_df, loc_df, max_km=0.3):
    """
    For each boring logs in intervals_df, assign a building by nearest match in loc_df,
    preferring rows with same boring_id; falls back to global nearest.
    max_km: maximum allowed distance to accept a match (km).
    """
    dfb = (intervals_df.groupby("boring_id")
             .agg(lat=("lat","first"), lon=("lon","first"))
             .reset_index())
    dfb["lat"] = pd.to_numeric(dfb["lat"], errors="coerce")
    dfb["lon"] = pd.to_numeric(dfb["lon"], errors="coerce")

    out = []
    for _, r in dfb.iterrows():
        bid, la, lo = r["boring_id"], r["lat"], r["lon"]
        cand = loc_df.copy()
        same = cand[cand["boring_id"]==bid]
        if not np.isnan(la) and not np.isnan(lo):
            target_lat = np.full(len(cand), la); target_lon = np.full(len(cand), lo)
            dkm = haversine_km(target_lat, target_lon, cand["lat"].values, cand["lon"].values)
            cand = cand.assign(_dkm=dkm)
            if not same.empty:
                target_lat = np.full(len(same), la); target_lon = np.full(len(same), lo)
                dkm_same = haversine_km(target_lat, target_lon, same["lat"].values, same["lon"].values)
                same = same.assign(_dkm=dkm_same).sort_values("_dkm")
                row = same.iloc[0]
            else:
                row = cand.sort_values("_dkm").iloc[0]
            out.append({"boring_id": bid, "building": row["building"] if row["_dkm"] <= max_km else None})
        else:
            if not same.empty and same["building"].nunique()==1:
                out.append({"boring_id": bid, "building": same["building"].iloc[0]})
            else:
                out.append({"boring_id": bid, "building": None})
    return pd.DataFrame(out)

def compute_sand_percent(intervals_df):
    rows = []
    for (bld, bid), g in intervals_df.groupby(["building","boring_id"]):
        lengths = (g["bot_ft"] - g["top_ft"]).clip(lower=0)
        total = float(lengths.sum())
        sandy = float((g.loc[g["is_sandy"], "bot_ft"] - g.loc[g["is_sandy"], "top_ft"]).clip(lower=0).sum())
        pct = (sandy / total * 100.0) if total > 0 else np.nan
        la = g["lat"].dropna().iloc[0] if g["lat"].notna().any() else np.nan
        lo = g["lon"].dropna().iloc[0] if g["lon"].notna().any() else np.nan
        rows.append({"building": bld, "boring_id": bid, "pct_sand": pct, "lat": la, "lon": lo})
    return pd.DataFrame(rows)

def order_by_location(boring_ids, loc_df, reverse=False):
    if loc_df is None or loc_df.empty: return list(boring_ids)
    sub = loc_df[loc_df["boring_id"].isin(boring_ids)].dropna(subset=["lat","lon"]).copy()
    if sub.empty: return list(boring_ids)
    lat0 = sub["lat"].mean()
    x = sub["lon"] * np.cos(np.deg2rad(lat0))
    y = sub["lat"]
    A = np.c_[x.values - x.mean(), y.values - y.mean()]
    if A.size == 0: return list(boring_ids)
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    v1 = vh[0]
    t = A @ v1
    sub = sub.assign(_t=t).sort_values("_t", ascending=not reverse)
    ordered = sub["boring_id"].tolist()
    missing = [b for b in boring_ids if b not in set(ordered)]
    return ordered + missing

def plot_building_logs(building, sub_df, out_png, n_scale=60.0):
    borings = list(dict.fromkeys(sub_df["boring_id"].tolist()))  # CSV order
    loc_df = (sub_df.groupby("boring_id").agg(lat=("lat","first"), lon=("lon","first")).reset_index())
    borings = order_by_location(borings, loc_df)

    col_w, gap, left_margin = 1.2, 0.8, 1.0
    fig_w = max(8, 0.95 * len(borings) + 3)
    fig_h = 9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")

    max_depth = float(sub_df["bot_ft"].max())
    max_depth = np.ceil(max_depth/5)*5

    for i, bid in enumerate(borings):
        bx0 = left_margin + i * (col_w + gap)
        cx = bx0 + col_w/2
        g = sub_df[sub_df["boring_id"]==bid].sort_values("top_ft").copy()

        merged = merge_same_lithology(g[["top_ft","bot_ft","soil_major"]])
        for r in merged:
            y0, y1 = r["top_ft"], r["bot_ft"]
            lith = str(r["soil_major"])
            color = COLOR_MAP.get(lith, DEFAULT_COLOR)
            hatch = HATCH_MAP.get(lith, DEFAULT_HATCH)
            ax.add_patch(Rectangle((bx0, y0), col_w, y1 - y0, facecolor=color, edgecolor="black", linewidth=1.0, hatch=hatch))

        #SPT-N curve
        n_vals = g[["bot_ft","n_numeric"]].dropna()
        if not n_vals.empty:
            x_line = cx + (n_vals["n_numeric"].clip(0, n_scale) / n_scale - 0.5) * (col_w * 0.9)
            y_line = n_vals["bot_ft"]
            ax.plot(x_line, y_line, linewidth=2.0, color="black")

        ax.add_patch(Rectangle((bx0, 0), col_w, max_depth, fill=False, edgecolor="black", linewidth=1.6))
        ax.text(cx, -2, bid, ha="center", va="top", fontsize=12, fontweight="bold")

    ax.set_ylim(max_depth, 0)
    ax.set_xlim(0, left_margin + len(borings) * (col_w + gap))
    ax.grid(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Depth (ft)")
    ft_to_m = 0.3048
    sec = ax.secondary_yaxis("right", functions=(lambda y: y*ft_to_m, lambda m: m/ft_to_m))
    sec.set_ylabel("Depth (m)"); sec.tick_params(length=0)

    present = list(dict.fromkeys(sub_df["soil_major"].dropna().map(str).tolist()))
    legend_elems = [Rectangle((0,0),1,1,facecolor=COLOR_MAP.get(l, DEFAULT_COLOR),edgecolor="black", hatch=HATCH_MAP.get(l, DEFAULT_HATCH)) for l in present]
    if legend_elems:
        ax.legend(legend_elems, present, title="Soil major", loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png

#-------------------- pipeline --------------------
def run(intervals_csv, loc_csv, out_dir, n_scale=60.0):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(intervals_csv)
    df["boring_id"] = df["boring_id"].astype(str).str.strip()
    df["soil_major"] = df["soil_major"].astype(str).str.strip()
    df["top_ft"] = pd.to_numeric(df["depth_top_ft"], errors="coerce")
    df["bot_ft"] = pd.to_numeric(df["depth_bot_ft"], errors="coerce")
    for c in ["lat","lon"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["n_numeric"] = df.get("n_value", np.nan).map(parse_n_value)
    df["is_sandy"] = df["soil_major"].map(is_sandy)

    loc = pd.read_csv(loc_csv)
    for c in ["boring_id","building"]:
        if c in loc.columns: loc[c] = loc[c].astype(str).str.strip()
    loc = loc.dropna(subset=["lat","lon"])

    #if merged file lacks a building column, infer it by nearest locations
    if "building" not in df.columns or df["building"].isna().all():
        assign = assign_building_to_borings(df, loc, max_km=0.3)  # 300 m radius
        df = df.merge(assign, on="boring_id", how="left")
    else:
        df["building"] = df["building"].astype(str).str.strip()

    #sand rate per boring
    summary = compute_sand_percent(df)
    summary.to_csv(out_dir / "sandy_percentages_by_boring.csv", index=False)

    #folium map (carto basemap -clean-)
    d = summary.dropna(subset=["lat","lon"]).copy()
    if d.empty:
        raise SystemExit("No coordinates to map.")
    base_lat = float(d["lat"].mean()); base_lon = float(d["lon"].mean())
    m = folium.Map(location=[base_lat, base_lon], tiles="CartoDB positron", zoom_start=14, control_scale=True)

    colormap = LinearColormap(["#FFFDE7","#FFE082","#FBC02D","#F57F17"], vmin=0, vmax=max(100.0,float(d["pct_sand"].max())))
    colormap.caption = "Sand percentage (%)"; colormap.add_to(m)

    #per-building logs + marker layers
    for bld, sub in df.groupby("building"):
        out_png = out_dir / f"logs_{slug(bld)}.png"
        plot_building_logs(bld, sub.copy(), str(out_png), n_scale=n_scale)

        fg = FeatureGroup(name=str(bld), show=True)
        dsub = summary[summary["building"]==bld].dropna(subset=["lat","lon"])
        for _, r in dsub.iterrows():
            pct = float(r["pct_sand"]) if pd.notna(r["pct_sand"]) else 0.0
            color = colormap(pct)
            html = (
                f"<b>{bld}</b><br>"
                f"Boring: <b>{r['boring_id']}</b><br>"
                f"Sand: <b>{pct:.1f}%</b><br>"
                f"<a href='logs_{slug(bld)}.png' target='_blank'>Open building log</a>"
            )
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=6, weight=1, color="#333",
                fill=True, fill_color=color, fill_opacity=0.95,
                tooltip=f"{r['boring_id']} — {pct:.1f}% sand",
                popup=folium.Popup(html, max_width=260)
            ).add_to(fg)
        fg.add_to(m)

    LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[d["lat"].min(), d["lon"].min()],[d["lat"].max(), d["lon"].max()]])
    map_path = out_dir / "sunnyisles_borings_map.html"
    m.save(str(map_path))
    return {"map_html": str(map_path), "summary_csv": str(out_dir / "sandy_percentages_by_boring.csv"), "out_dir": str(out_dir)}

def main():
    ap = argparse.ArgumentParser(description="Map % sand and per-building soil logs")
    ap.add_argument("--intervals", required=True, help="Merged intervals CSV (boring_id, depth_top_ft, depth_bot_ft, soil_major, lat, lon, [building])")
    ap.add_argument("--locations", required=True, help="boring_locations.csv (boring_id, lat, lon, building)")
    ap.add_argument("--outdir", default="si_out", help="Output directory")
    ap.add_argument("--nscale", type=float, default=60.0, help="N-value scale for plot width")
    args = ap.parse_args()
    res = run(args.intervals, args.locations, args.outdir, n_scale=args.nscale)
    print("[INFO] Map:", res["map_html"])
    print("[INFO] Summary:", res["summary_csv"])
    print("[INFO] Output dir:", res["out_dir"])

if __name__ == "__main__":
    main()

