from __future__ import annotations

import argparse
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from folium.plugins import DualMap


# ============================================================
# CONFIG / DEFAULTS
# ============================================================
RESULTS_PATH = "experiments/latest_run/results_df.csv"
TEST_PATH = "data/raw/test_data.csv"
OUTPUT_DIR = "map_views"
DEFAULT_IDX = 0
MAX_PREVIOUS_POINTS = 200

TIME_COL = "TIME"
LAT_COL = "LAT"
LON_COL = "LON"
MMSI_COL = "MMSI"
VOYAGE_COL = "voyage_id"


# ============================================================
# HELPERS
# ============================================================
def load_results(results_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)

    for col in ["anchor_time", "pred_time", "TIME"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if MMSI_COL in df.columns:
        df[MMSI_COL] = df[MMSI_COL].astype(str)
    if VOYAGE_COL in df.columns:
        df[VOYAGE_COL] = df[VOYAGE_COL].astype(str)

    return df



def load_test_data(test_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(test_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df[MMSI_COL] = df[MMSI_COL].astype(str)
    df[VOYAGE_COL] = df[VOYAGE_COL].astype(str)
    return df.sort_values([VOYAGE_COL, TIME_COL]).reset_index(drop=True)



def resolve_row(results_df: pd.DataFrame, idx: int | None = None, row_id: str | None = None) -> pd.Series:
    if row_id is not None:
        hits = results_df[results_df["row_id"].astype(str) == str(row_id)]
        if hits.empty:
            raise ValueError(f"row_id={row_id} not found in results_df")
        return hits.iloc[0]

    if idx is None:
        idx = 0

    hits = results_df[results_df["idx"] == idx]
    if not hits.empty:
        return hits.iloc[0]

    if idx < 0 or idx >= len(results_df):
        raise IndexError(f"idx={idx} is out of bounds for results_df of length {len(results_df)}")

    return results_df.iloc[idx]



def reduce_points(track_df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(track_df) <= max_points:
        return track_df

    sampled_idx = np.linspace(0, len(track_df) - 1, max_points, dtype=int)
    return track_df.iloc[sampled_idx].copy()



def get_prior_track(
    test_df: pd.DataFrame,
    row: pd.Series,
    max_previous_points: int,
) -> pd.DataFrame:
    mmsi = str(row[MMSI_COL])
    voyage_id = str(row[VOYAGE_COL])

    anchor_time = row.get("anchor_time")
    if pd.isna(anchor_time):
        anchor_time = row.get("TIME")

    voyage_df = test_df[
        (test_df[MMSI_COL] == mmsi) &
        (test_df[VOYAGE_COL] == voyage_id)
    ].copy()

    if voyage_df.empty:
        raise ValueError(f"No matching voyage found in test data for MMSI={mmsi}, voyage_id={voyage_id}")

    voyage_df = voyage_df.sort_values(TIME_COL).reset_index(drop=True)

    if pd.notna(anchor_time):
        prior_df = voyage_df[voyage_df[TIME_COL] <= anchor_time].copy()
    else:
        prior_df = voyage_df.copy()

    if prior_df.empty:
        prior_df = voyage_df.iloc[[0]].copy()

    if max_previous_points is not None and max_previous_points > 0 and len(prior_df) > max_previous_points:
        prior_df = prior_df.iloc[-max_previous_points:].copy()

    return prior_df.reset_index(drop=True)



def format_popup(lines: list[str]) -> str:
    return "<br>".join(lines)



def add_prior_track(target_map, track_df: pd.DataFrame):
    pts = track_df[[LAT_COL, LON_COL]].dropna().values.tolist()

    if len(pts) >= 2:
        folium.PolyLine(
            pts,
            color="gray",
            weight=3,
            opacity=0.8,
            tooltip="Previous trajectory",
        ).add_to(target_map)

    for i, (_, r) in enumerate(track_df.iterrows()):
        folium.CircleMarker(
            location=[r[LAT_COL], r[LON_COL]],
            radius=2,
            color="gray",
            fill=True,
            fill_color="gray",
            fill_opacity=0.75,
            opacity=0.75,
            popup=format_popup([
                f"track_idx: {i}",
                f"TIME: {r[TIME_COL]}",
                f"LAT: {r[LAT_COL]:.6f}",
                f"LON: {r[LON_COL]:.6f}",
            ]),
        ).add_to(target_map)



def add_anchor_and_truth(target_map, row: pd.Series):
    last_lat = float(row["last_lat"])
    last_lon = float(row["last_lon"])
    true_lat = float(row["true_lat"])
    true_lon = float(row["true_lon"])

    folium.Marker(
        [last_lat, last_lon],
        tooltip="Anchor / current point",
        popup=format_popup([
            "Anchor / current point",
            f"anchor_time: {row.get('anchor_time', 'NA')}",
            f"LAT: {last_lat:.6f}",
            f"LON: {last_lon:.6f}",
            f"SOG: {row.get('last_speed', 'NA')}",
            f"COG: {row.get('last_cog', 'NA')}",
        ]),
        icon=folium.Icon(color="black", icon="play", prefix="fa"),
    ).add_to(target_map)

    folium.Marker(
        [true_lat, true_lon],
        tooltip="True next point",
        popup=format_popup([
            "True next point",
            f"pred_time: {row.get('pred_time', 'NA')}",
            f"LAT: {true_lat:.6f}",
            f"LON: {true_lon:.6f}",
        ]),
        icon=folium.Icon(color="green", icon="ok", prefix="fa"),
    ).add_to(target_map)

    folium.PolyLine(
        [[last_lat, last_lon], [true_lat, true_lon]],
        color="green",
        weight=3,
        opacity=0.9,
        dash_array="8,6",
        tooltip="Actual movement",
    ).add_to(target_map)



def add_tcn_prediction(target_map, row: pd.Series):
    last_lat = float(row["last_lat"])
    last_lon = float(row["last_lon"])
    pred_lat = float(row["tcn_pred_lat"])
    pred_lon = float(row["tcn_pred_lon"])
    err = float(row["tcn_error_yds"])

    folium.Marker(
        [pred_lat, pred_lon],
        tooltip="TCN prediction",
        popup=format_popup([
            "TCN prediction",
            f"LAT: {pred_lat:.6f}",
            f"LON: {pred_lon:.6f}",
            f"TCN error: {err:.2f} yds",
        ]),
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(target_map)

    folium.PolyLine(
        [[last_lat, last_lon], [pred_lat, pred_lon]],
        color="blue",
        weight=3,
        opacity=0.9,
        dash_array="4,6",
        tooltip=f"TCN predicted movement | error={err:.2f} yds",
    ).add_to(target_map)



def add_dr_prediction(target_map, row: pd.Series):
    last_lat = float(row["last_lat"])
    last_lon = float(row["last_lon"])
    pred_lat = float(row["dr_pred_lat"])
    pred_lon = float(row["dr_pred_lon"])
    err = float(row["dr_error_yds"])

    folium.Marker(
        [pred_lat, pred_lon],
        tooltip="Dead reckoning prediction",
        popup=format_popup([
            "Dead reckoning prediction",
            f"LAT: {pred_lat:.6f}",
            f"LON: {pred_lon:.6f}",
            f"DR error: {err:.2f} yds",
        ]),
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(target_map)

    folium.PolyLine(
        [[last_lat, last_lon], [pred_lat, pred_lon]],
        color="red",
        weight=3,
        opacity=0.9,
        dash_array="4,6",
        tooltip=f"DR predicted movement | error={err:.2f} yds",
    ).add_to(target_map)



def add_header(dm: DualMap, row: pd.Series):
    winner = row.get("winner", "NA")
    html = f"""
    <div style="
        position: fixed;
        top: 10px;
        left: 50px;
        z-index: 9999;
        background-color: white;
        border: 2px solid #444;
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 13px;
        line-height: 1.35;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        max-width: 540px;
    ">
        <b>Indexed Result Comparison</b><br>
        idx: {row.get('idx', 'NA')} &nbsp;|&nbsp; row_id: {row.get('row_id', 'NA')}<br>
        MMSI: {row.get('MMSI', 'NA')} &nbsp;|&nbsp; voyage_id: {row.get('voyage_id', 'NA')}<br>
        anchor_time: {row.get('anchor_time', 'NA')}<br>
        pred_time: {row.get('pred_time', 'NA')}<br>
        TCN error: {float(row.get('tcn_error_yds', float('nan'))):.2f} yds &nbsp;|&nbsp;
        DR error: {float(row.get('dr_error_yds', float('nan'))):.2f} yds &nbsp;|&nbsp;
        Improvement: {float(row.get('improvement_yds', float('nan'))):.2f} yds<br>
        Winner: <b>{winner}</b><br>
        <span style="color:gray;">Gray</span> = previous trajectory &nbsp;|&nbsp;
        <span style="color:green;">Green</span> = true next point &nbsp;|&nbsp;
        <span style="color:blue;">Blue</span> = TCN &nbsp;|&nbsp;
        <span style="color:red;">Red</span> = DR
    </div>
    """
    dm.get_root().html.add_child(folium.Element(html))



def build_dual_map(row: pd.Series, prior_df: pd.DataFrame) -> DualMap:
    center_lat = float(row["true_lat"])
    center_lon = float(row["true_lon"])

    dm = DualMap(location=[center_lat, center_lon], zoom_start=11)

    add_prior_track(dm.m1, prior_df)
    add_prior_track(dm.m2, prior_df)

    add_anchor_and_truth(dm.m1, row)
    add_anchor_and_truth(dm.m2, row)

    add_tcn_prediction(dm.m1, row)
    add_dr_prediction(dm.m2, row)

    folium.map.Marker(
        [center_lat, center_lon],
        icon=folium.DivIcon(html='<div style="font-size: 14px; color: blue;"><b>LEFT: TCN</b></div>')
    ).add_to(dm.m1)
    folium.map.Marker(
        [center_lat, center_lon],
        icon=folium.DivIcon(html='<div style="font-size: 14px; color: red;"><b>RIGHT: DR</b></div>')
    ).add_to(dm.m2)

    add_header(dm, row)
    return dm



def save_map_for_index(
    results_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    idx: int | None = None,
    row_id: str | None = None,
    max_previous_points: int = MAX_PREVIOUS_POINTS,
) -> Path:
    results_df = load_results(results_path)
    test_df = load_test_data(test_path)
    row = resolve_row(results_df, idx=idx, row_id=row_id)
    prior_df = get_prior_track(test_df, row, max_previous_points=max_previous_points)

    dm = build_dual_map(row, prior_df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_idx = str(row.get("idx", "NA"))
    safe_row_id = str(row.get("row_id", "NA")).replace("/", "_")
    out_path = output_dir / f"result_idx_{safe_idx}_rowid_{safe_row_id}.html"
    dm.save(str(out_path))
    return out_path


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a side-by-side Folium map for one indexed TCN vs DR result."
    )
    parser.add_argument("--results", type=str, default=RESULTS_PATH, help="Path to results_df.csv")
    parser.add_argument("--test", type=str, default=TEST_PATH, help="Path to test_data.csv")
    parser.add_argument("--outdir", type=str, default=OUTPUT_DIR, help="Directory to save HTML maps")
    parser.add_argument("--idx", type=int, default=None, help="Value from the results_df 'idx' column")
    parser.add_argument("--row_id", type=str, default=None, help="Alternative lookup by row_id")
    parser.add_argument(
        "--max_previous_points",
        type=int,
        default=MAX_PREVIOUS_POINTS,
        help="Maximum number of prior trajectory points to plot. Uses the most recent N points.",
    )
    return parser.parse_args()



def main():
    args = parse_args()
    out_path = save_map_for_index(
        results_path=args.results,
        test_path=args.test,
        output_dir=args.outdir,
        idx=args.idx,
        row_id=args.row_id,
        max_previous_points=args.max_previous_points,
    )
    print(f"Saved map to: {out_path}")


if __name__ == "__main__":
    main()
