from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from folium.plugins import MarkerCluster
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    # ----------------------------
    # Project paths (relative to capstone/)
    # ----------------------------
    train_path: str = "data/raw/train_data.csv"
    test_path: str = "data/raw/test_data.csv"
    experiments_dir: str = "experiments"

    # ----------------------------
    # Data / model setup
    # ----------------------------
    seq_len: int = 19
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.25
    kernel_size: int = 3
    tcn_channels: tuple = (32, 64, 64)
    dilations: tuple = (1, 2, 4)

    # ----------------------------
    # Early stopping
    # ----------------------------
    max_epochs: int = 100
    min_epochs: int = 10
    loss_stability_threshold: float = 0.02

    # ----------------------------
    # Runtime
    # ----------------------------
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # Core columns
    # ----------------------------
    track_id_col: str = "voyage_id"
    vessel_id_col: str = "MMSI"
    row_id_col: str = "row_id"
    time_col: str = "TIME"
    lat_col: str = "LAT"
    lon_col: str = "LON"
    speed_col: str = "SPEED"
    cog_col: str = "COG"
    heading_col: str = "HEADING"
    dt_col: str = "dt"
    num_pings_col: str = "num_pings"

    # ----------------------------
    # Constants
    # ----------------------------
    earth_radius_m: float = 6_371_000.0
    knots_to_mps: float = 0.514444
    meters_to_yards: float = 1.0936132983377078

    # ----------------------------
    # Reporting / artifacts
    # ----------------------------
    histogram_bins: int = 200
    map_max_points: int = 150
    save_comparison_map: bool = True
    save_top_gap_maps: bool = True
    top_gap_map_count: int = 15
    dr_horizon_mode: str = "input_dt"  # options: input_dt, target_dt


CFG = Config()


# ============================================================
# LOGGING
# ============================================================

class RunLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.log_path, "w", encoding="utf-8")

    def log(self, message: str = ""):
        print(message)
        self.f.write(message + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ============================================================
# BASIC UTILITIES
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_lr_for_name(x: float) -> str:
    return f"{x:.0e}".replace("+0", "").replace("+", "")


def make_run_name(cfg: Config) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ch = "-".join(str(x) for x in cfg.tcn_channels)
    dil = "-".join(str(x) for x in cfg.dilations)
    return (
        f"{timestamp}"
        f"_seq{cfg.seq_len}"
        f"_bs{cfg.batch_size}"
        f"_lr{format_lr_for_name(cfg.learning_rate)}"
        f"_do{cfg.dropout}"
        f"_ks{cfg.kernel_size}"
        f"_ch{ch}"
        f"_dil{dil}"
    )


def config_to_jsonable_dict(cfg: Config) -> dict:
    d = asdict(cfg)
    d["tcn_channels"] = list(cfg.tcn_channels)
    d["dilations"] = list(cfg.dilations)
    return d


def append_experiment_summary(summary_row: dict, experiments_dir: Path):
    summary_path = experiments_dir / "experiment_summary.csv"
    row_df = pd.DataFrame([summary_row])

    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(summary_path, index=False)


# ============================================================
# DATA LOADING / PREP
# ============================================================

def load_data(path: str, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")

    numeric_cols = [
        cfg.lat_col, cfg.lon_col, cfg.speed_col,
        cfg.cog_col, cfg.heading_col, cfg.dt_col,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[cfg.dt_col] = df[cfg.dt_col].fillna(0.0)
    df.loc[df[cfg.dt_col] < 0, cfg.dt_col] = 0.0
    df.loc[df[cfg.dt_col] > 1800, cfg.dt_col] = 0.0

    df[cfg.heading_col] = df[cfg.heading_col].fillna(df[cfg.cog_col])

    required = [
        cfg.track_id_col,
        cfg.vessel_id_col,
        cfg.row_id_col,
        cfg.num_pings_col,
        cfg.time_col,
        cfg.lat_col,
        cfg.lon_col,
        cfg.speed_col,
        cfg.cog_col,
        cfg.heading_col,
        cfg.dt_col,
    ]
    df = df.dropna(subset=required).copy()

    df[cfg.vessel_id_col] = df[cfg.vessel_id_col].astype(str)
    df[cfg.track_id_col] = df[cfg.track_id_col].astype(str)
    df[cfg.row_id_col] = df[cfg.row_id_col].astype(str)

    df = df.sort_values([cfg.track_id_col, cfg.time_col]).reset_index(drop=True)
    return df


def add_features_and_targets(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    df["COG_rad"] = np.deg2rad(df[cfg.cog_col] % 360.0)
    df["COG_cos"] = np.cos(df["COG_rad"])
    df["COG_sin"] = np.sin(df["COG_rad"])

    df["HEADING_rad"] = np.deg2rad(df[cfg.heading_col] % 360.0)
    df["HEADING_cos"] = np.cos(df["HEADING_rad"])
    df["HEADING_sin"] = np.sin(df["HEADING_rad"])

    df["next_lat"] = df.groupby(cfg.track_id_col)[cfg.lat_col].shift(-1)
    df["next_lon"] = df.groupby(cfg.track_id_col)[cfg.lon_col].shift(-1)
    df["target_dt"] = df.groupby(cfg.track_id_col)[cfg.dt_col].shift(-1)
    df["next_time"] = df.groupby(cfg.track_id_col)[cfg.time_col].shift(-1)

    df["dlat"] = df["next_lat"] - df[cfg.lat_col]
    df["dlon"] = df["next_lon"] - df[cfg.lon_col]

    df = df.dropna(subset=["dlat", "dlon", "next_lat", "next_lon", "target_dt", "next_time"]).reset_index(drop=True)
    return df


FEATURE_COLS = [
    "LAT", "LON", "SPEED", "dt",
    "COG_cos", "COG_sin", "HEADING_cos", "HEADING_sin",
]

TARGET_COLS = ["dlat", "dlon"]


def build_sequences(df: pd.DataFrame, cfg: Config):
    X_list, y_list, last_pos_list, meta_rows = [], [], [], []

    for _, group in df.groupby(cfg.track_id_col, sort=False):
        group = group.sort_values(cfg.time_col).reset_index(drop=True)

        if len(group) <= cfg.seq_len:
            continue

        X_vals = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_vals = group[TARGET_COLS].to_numpy(dtype=np.float32)
        latlon_vals = group[[cfg.lat_col, cfg.lon_col]].to_numpy(dtype=np.float32)

        for i in range(len(group) - cfg.seq_len):
            last_idx = i + cfg.seq_len - 1
            pred_idx = i + cfg.seq_len

            X_seq = X_vals[i:i + cfg.seq_len]
            y_next = y_vals[last_idx]  # delta from anchor row to the true next row
            last_pos = latlon_vals[last_idx]

            X_list.append(X_seq)
            y_list.append(y_next)
            last_pos_list.append(last_pos)

            meta_rows.append({
                "row_id": group.loc[pred_idx, cfg.row_id_col],
                "MMSI": group.loc[pred_idx, cfg.vessel_id_col],
                "voyage_id": group.loc[pred_idx, cfg.track_id_col],
                "anchor_time": group.loc[last_idx, cfg.time_col],
                "pred_time": group.loc[pred_idx, cfg.time_col],
                "input_dt": float(group.loc[last_idx, cfg.dt_col]),
                "target_dt": float(group.loc[pred_idx, cfg.dt_col]),
                "num_pings": group.loc[pred_idx, cfg.num_pings_col],
                "last_speed": float(group.loc[last_idx, cfg.speed_col]),
                "last_cog": float(group.loc[last_idx, cfg.cog_col]),
                "last_heading": float(group.loc[last_idx, cfg.heading_col]),
            })

    if not X_list:
        raise ValueError("No sequences were built. Check seq_len and input data.")

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    last_pos = np.stack(last_pos_list).astype(np.float32)
    meta_df = pd.DataFrame(meta_rows)

    return X, y, last_pos, meta_df


def scale_X_train_test(X_train: np.ndarray, X_test: np.ndarray):
    n_train, seq_len, n_features = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(n_train * seq_len, n_features))

    X_train_scaled = scaler.transform(
        X_train.reshape(n_train * seq_len, n_features)
    ).reshape(n_train, seq_len, n_features)

    X_test_scaled = scaler.transform(
        X_test.reshape(n_test * seq_len, n_features)
    ).reshape(n_test, seq_len, n_features)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32), scaler


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# MODEL
# ============================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.final_act = nn.ELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_act(out + res)


class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, channels, kernel_size, dilations, dropout):
        super().__init__()
        layers = []
        in_ch = input_dim
        for out_ch, dilation in zip(channels, dilations):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.tcn(x)
        z_last = z[:, :, -1]
        return self.fc(z_last)


# ============================================================
# TRAIN / PREDICT
# ============================================================

def within_2_percent(a, b, threshold=0.02):
    denom = max((a + b) / 2, 1e-12)
    return abs(a - b) / denom <= threshold


def train_model(model, train_loader, cfg: Config, logger: RunLogger):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.MSELoss()

    train_losses = []
    stopped_early = False

    for epoch in range(cfg.max_epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(cfg.device)
            y_batch = y_batch.to(cfg.device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        logger.log(f"Epoch {epoch+1:03d}/{cfg.max_epochs} | train_loss={avg_loss:.8f}")

        if len(train_losses) >= 3 and (epoch + 1) >= cfg.min_epochs:
            l1, l2, l3 = train_losses[-3:]
            if (
                within_2_percent(l1, l2, cfg.loss_stability_threshold)
                and within_2_percent(l2, l3, cfg.loss_stability_threshold)
                and within_2_percent(l1, l3, cfg.loss_stability_threshold)
            ):
                logger.log(
                    f"\nStopping early after epoch {epoch+1}: last 3 losses are all within "
                    f"{cfg.loss_stability_threshold * 100:.1f}% of each other."
                )
                stopped_early = True
                break

    return {
        "train_losses": train_losses,
        "epochs_trained": len(train_losses),
        "stopped_early": stopped_early,
        "final_train_loss": float(train_losses[-1]),
    }


def predict_model(model, loader, cfg: Config):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(cfg.device)
            preds.append(model(X_batch).cpu().numpy())
    return np.vstack(preds)


# ============================================================
# METRICS / RECONSTRUCTION
# ============================================================

def reconstruct_latlon(last_positions: np.ndarray, deltas: np.ndarray):
    pred_lat = last_positions[:, 0] + deltas[:, 0]
    pred_lon = last_positions[:, 1] + deltas[:, 1]
    return np.column_stack([pred_lat, pred_lon]).astype(np.float32)


def haversine_m(lat1, lon1, lat2, lon2, R=6_371_000.0):
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def forward_dead_reckoning(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    speed_knots: np.ndarray,
    cog_deg: np.ndarray,
    dt_seconds: np.ndarray,
    R: float,
    knots_to_mps: float,
) -> np.ndarray:
    lat1 = np.deg2rad(lat_deg)
    lon1 = np.deg2rad(lon_deg)
    brng = np.deg2rad(cog_deg % 360.0)
    d = np.maximum(dt_seconds, 0.0) * np.maximum(speed_knots, 0.0) * knots_to_mps
    ang_dist = d / R

    sin_lat1 = np.sin(lat1)
    cos_lat1 = np.cos(lat1)
    sin_ang = np.sin(ang_dist)
    cos_ang = np.cos(ang_dist)

    lat2 = np.arcsin(sin_lat1 * cos_ang + cos_lat1 * sin_ang * np.cos(brng))
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * sin_ang * cos_lat1,
        cos_ang - sin_lat1 * np.sin(lat2),
    )

    lon2 = (lon2 + np.pi) % (2 * np.pi) - np.pi
    return np.column_stack([np.rad2deg(lat2), np.rad2deg(lon2)]).astype(np.float32)


def choose_dr_horizon(meta_df: pd.DataFrame, cfg: Config) -> np.ndarray:
    if cfg.dr_horizon_mode == "target_dt":
        return meta_df["target_dt"].to_numpy(dtype=np.float32)
    return meta_df["input_dt"].to_numpy(dtype=np.float32)


def compute_outlier_stats(errors: pd.Series) -> dict:
    q1 = float(errors.quantile(0.25))
    median = float(errors.quantile(0.50))
    q3 = float(errors.quantile(0.75))
    iqr = float(q3 - q1)

    lower_bound = float(q1 - 1.5 * iqr)
    upper_bound = float(q3 + 1.5 * iqr)

    outliers = errors[(errors < lower_bound) | (errors > upper_bound)]
    num_outliers = int(len(outliers))
    outlier_pct = float(100 * num_outliers / len(errors))

    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "num_outliers": num_outliers,
        "outlier_pct": outlier_pct,
    }


def summarize_error(error_yds: np.ndarray) -> dict:
    return {
        "rmse_yds": float(np.sqrt(np.mean(error_yds ** 2))),
        "mae_yds": float(np.mean(error_yds)),
        "median_yds": float(np.median(error_yds)),
        "p95_yds": float(np.percentile(error_yds, 95)),
    }


# ============================================================
# SAVING / PLOTTING
# ============================================================

def save_config(cfg: Config, run_dir: Path):
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_jsonable_dict(cfg), f, indent=2)


def save_train_loss_history(train_losses: List[float], run_dir: Path):
    loss_df = pd.DataFrame({
        "epoch": np.arange(1, len(train_losses) + 1),
        "train_loss": train_losses,
    })
    loss_df.to_csv(run_dir / "train_loss_history.csv", index=False)


def save_results(results_df: pd.DataFrame, run_dir: Path) -> Path:
    results_path = run_dir / "results_df.csv"
    results_df.to_csv(results_path, index=False)
    return results_path


def save_metrics_summary(metrics_summary: dict, run_dir: Path):
    with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)


def print_model_comparison(tcn_summary: dict, dr_summary: dict, logger: RunLogger):
    logger.log("\n=== FINAL TEST METRICS (YARDS) ===")
    logger.log(
        f"TCN | RMSE={tcn_summary['rmse_yds']:.2f} | MAE={tcn_summary['mae_yds']:.2f} | "
        f"Median={tcn_summary['median_yds']:.2f} | P95={tcn_summary['p95_yds']:.2f}"
    )
    logger.log(
        f"DR  | RMSE={dr_summary['rmse_yds']:.2f} | MAE={dr_summary['mae_yds']:.2f} | "
        f"Median={dr_summary['median_yds']:.2f} | P95={dr_summary['p95_yds']:.2f}"
    )


def print_results_preview(results_df: pd.DataFrame, logger: RunLogger, n: int = 5):
    cols = [
        "idx", "row_id", "MMSI", "voyage_id", "anchor_time", "pred_time",
        "input_dt", "target_dt", "tcn_error_yds", "dr_error_yds", "improvement_yds", "winner",
    ]
    logger.log("\n=== RESULTS PREVIEW ===")
    logger.log(results_df[cols].head(n).to_string(index=False))


def print_outlier_stats(name: str, stats: dict, logger: RunLogger):
    logger.log(f"\n=== {name} ERROR DISTRIBUTION SUMMARY (YARDS) ===")
    logger.log(f"Q1: {stats['q1']:.2f}")
    logger.log(f"Median: {stats['median']:.2f}")
    logger.log(f"Q3: {stats['q3']:.2f}")
    logger.log(f"IQR: {stats['iqr']:.2f}")
    logger.log(f"Lower bound: {stats['lower_bound']:.2f}")
    logger.log(f"Upper bound: {stats['upper_bound']:.2f}")
    logger.log(f"Number of outliers: {stats['num_outliers']}")
    logger.log(f"Percent of outliers: {stats['outlier_pct']:.2f}%")


def plot_comparison_histogram(results_df: pd.DataFrame, run_dir: Path, bins: int = 100):
    plt.figure(figsize=(10, 6))
    plt.hist(results_df["tcn_error_yds"], bins=bins, alpha=0.6, label="TCN")
    plt.hist(results_df["dr_error_yds"], bins=bins, alpha=0.6, label="Dead Reckoning")
    plt.xlabel("Error (yards)")
    plt.ylabel("Frequency")
    plt.title("TCN vs Dead Reckoning Error Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "comparison_error_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_improvement_histogram(results_df: pd.DataFrame, run_dir: Path, bins: int = 100):
    plt.figure(figsize=(10, 6))
    plt.hist(results_df["improvement_yds"], bins=bins)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Improvement in error (yards) = DR error - TCN error")
    plt.ylabel("Frequency")
    plt.title("Improvement of TCN Over Dead Reckoning")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "improvement_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()


def select_map_rows(results_df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(results_df) <= max_points:
        return results_df.copy()
    top = results_df.nlargest(max_points, "abs_improvement_yds")
    return top.sort_values("anchor_time").reset_index(drop=True)


def make_popup_html(row: pd.Series) -> str:
    return f"""
    <b>row_id:</b> {row['row_id']}<br>
    <b>MMSI:</b> {row['MMSI']}<br>
    <b>voyage_id:</b> {row['voyage_id']}<br>
    <b>anchor_time:</b> {row['anchor_time']}<br>
    <b>pred_time:</b> {row['pred_time']}<br>
    <b>input_dt:</b> {row['input_dt']:.1f} s<br>
    <b>target_dt:</b> {row['target_dt']:.1f} s<br>
    <b>TCN error:</b> {row['tcn_error_yds']:.2f} yds<br>
    <b>DR error:</b> {row['dr_error_yds']:.2f} yds<br>
    <b>Improvement:</b> {row['improvement_yds']:.2f} yds<br>
    <b>Winner:</b> {row['winner']}<br>
    """


def save_comparison_map(results_df: pd.DataFrame, out_path: Path, max_points: int = 150):
    map_df = select_map_rows(results_df, max_points)
    center_lat = float(map_df["last_lat"].mean())
    center_lon = float(map_df["last_lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, control_scale=True)

    anchors_group = folium.FeatureGroup(name="Anchor / Current Point", show=True)
    true_group = folium.FeatureGroup(name="True Next Point", show=True)
    tcn_group = folium.FeatureGroup(name="TCN Prediction", show=True)
    dr_group = folium.FeatureGroup(name="Dead Reckoning Prediction", show=True)
    path_group = folium.FeatureGroup(name="Movement Paths", show=True)

    anchor_cluster = MarkerCluster(name="Anchor Cluster").add_to(anchors_group)
    true_cluster = MarkerCluster(name="True Cluster").add_to(true_group)
    tcn_cluster = MarkerCluster(name="TCN Cluster").add_to(tcn_group)
    dr_cluster = MarkerCluster(name="DR Cluster").add_to(dr_group)

    for _, row in map_df.iterrows():
        popup = folium.Popup(make_popup_html(row), max_width=350)

        folium.CircleMarker(
            location=[row["last_lat"], row["last_lon"]],
            radius=4,
            color="black",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"Anchor | {row['row_id']}",
            popup=popup,
        ).add_to(anchor_cluster)

        folium.CircleMarker(
            location=[row["true_lat"], row["true_lon"]],
            radius=5,
            color="green",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"True | {row['row_id']}",
            popup=popup,
        ).add_to(true_cluster)

        folium.CircleMarker(
            location=[row["tcn_pred_lat"], row["tcn_pred_lon"]],
            radius=5,
            color="blue",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"TCN | {row['row_id']} | {row['tcn_error_yds']:.1f} yds",
            popup=popup,
        ).add_to(tcn_cluster)

        folium.CircleMarker(
            location=[row["dr_pred_lat"], row["dr_pred_lon"]],
            radius=5,
            color="red",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"DR | {row['row_id']} | {row['dr_error_yds']:.1f} yds",
            popup=popup,
        ).add_to(dr_cluster)

        folium.PolyLine(
            [[row["last_lat"], row["last_lon"]], [row["true_lat"], row["true_lon"]]],
            color="green",
            weight=2,
            opacity=0.8,
            tooltip=f"True path | {row['row_id']}",
        ).add_to(path_group)

        folium.PolyLine(
            [[row["last_lat"], row["last_lon"]], [row["tcn_pred_lat"], row["tcn_pred_lon"]]],
            color="blue",
            weight=2,
            opacity=0.8,
            dash_array="5,8",
            tooltip=f"TCN path | {row['row_id']}",
        ).add_to(path_group)

        folium.PolyLine(
            [[row["last_lat"], row["last_lon"]], [row["dr_pred_lat"], row["dr_pred_lon"]]],
            color="red",
            weight=2,
            opacity=0.8,
            dash_array="2,6",
            tooltip=f"DR path | {row['row_id']}",
        ).add_to(path_group)

    anchors_group.add_to(m)
    true_group.add_to(m)
    tcn_group.add_to(m)
    dr_group.add_to(m)
    path_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_path))
    return out_path


def save_top_gap_case_maps(results_df: pd.DataFrame, out_dir: Path, n_maps: int = 15):
    out_dir.mkdir(parents=True, exist_ok=True)
    top_rows = results_df.nlargest(n_maps, "abs_improvement_yds")
    saved = []

    for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
        center_lat = float(np.mean([row["last_lat"], row["true_lat"], row["tcn_pred_lat"], row["dr_pred_lat"]]))
        center_lon = float(np.mean([row["last_lon"], row["true_lon"], row["tcn_pred_lon"], row["dr_pred_lon"]]))
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, control_scale=True)

        popup = folium.Popup(make_popup_html(row), max_width=350)

        folium.Marker(
            [row["last_lat"], row["last_lon"]],
            tooltip="Anchor / Current Point",
            popup=popup,
            icon=folium.Icon(color="black", icon="play"),
        ).add_to(m)

        folium.Marker(
            [row["true_lat"], row["true_lon"]],
            tooltip="True Next Point",
            popup=popup,
            icon=folium.Icon(color="green", icon="ok"),
        ).add_to(m)

        folium.Marker(
            [row["tcn_pred_lat"], row["tcn_pred_lon"]],
            tooltip=f"TCN Prediction | {row['tcn_error_yds']:.1f} yds",
            popup=popup,
            icon=folium.Icon(color="blue", icon="flash"),
        ).add_to(m)

        folium.Marker(
            [row["dr_pred_lat"], row["dr_pred_lon"]],
            tooltip=f"DR Prediction | {row['dr_error_yds']:.1f} yds",
            popup=popup,
            icon=folium.Icon(color="red", icon="remove"),
        ).add_to(m)

        folium.PolyLine(
            [[row["last_lat"], row["last_lon"]], [row["true_lat"], row["true_lon"]]],
            color="green", weight=3, tooltip="True movement"
        ).add_to(m)
        folium.PolyLine(
            [[row["last_lat"], row["last_lon"]], [row["tcn_pred_lat"], row["tcn_pred_lon"]]],
            color="blue", weight=3, dash_array="5,8", tooltip="TCN movement"
        ).add_to(m)
        folium.PolyLine(
            [[row["last_lat"], row["last_lon"]], [row["dr_pred_lat"], row["dr_pred_lon"]]],
            color="red", weight=3, dash_array="2,6", tooltip="DR movement"
        ).add_to(m)

        safe_row_id = str(row["row_id"]).replace("/", "_")
        out_file = out_dir / f"gap_case_{rank:02d}_{safe_row_id}.html"
        m.save(str(out_file))
        saved.append(str(out_file))

    return saved


# ============================================================
# PIPELINE
# ============================================================

def run_experiment(cfg: Config):
    set_seed(cfg.seed)

    experiments_dir = Path(cfg.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    run_name = make_run_name(cfg)
    run_dir = experiments_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    logger = RunLogger(run_dir / "run_log.txt")

    try:
        save_config(cfg, run_dir)
        logger.log(f"Run directory: {run_dir}")
        logger.log(f"Device: {cfg.device}")
        logger.log(f"DR horizon mode: {cfg.dr_horizon_mode}")

        # ----------------------------
        # Load + prep
        # ----------------------------
        train_df = add_features_and_targets(load_data(cfg.train_path, cfg), cfg)
        test_df = add_features_and_targets(load_data(cfg.test_path, cfg), cfg)

        X_train, y_train, last_pos_train, meta_train = build_sequences(train_df, cfg)
        X_test, y_test, last_pos_test, meta_test = build_sequences(test_df, cfg)

        X_test_unscaled = X_test.copy()
        X_train, X_test, feature_scaler = scale_X_train_test(X_train, X_test)

        train_ds = SequenceDataset(X_train, y_train)
        test_ds = SequenceDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

        model = TCN(
            input_dim=X_train.shape[-1],
            output_dim=y_train.shape[-1],
            channels=cfg.tcn_channels,
            kernel_size=cfg.kernel_size,
            dilations=cfg.dilations,
            dropout=cfg.dropout,
        ).to(cfg.device)

        # ----------------------------
        # Train / predict
        # ----------------------------
        train_info = train_model(model, train_loader, cfg, logger)
        pred_test = predict_model(model, test_loader, cfg)

        # ----------------------------
        # Reconstruct TCN
        # ----------------------------
        tcn_pred_latlon = reconstruct_latlon(last_pos_test, pred_test)
        true_latlon = reconstruct_latlon(last_pos_test, y_test)

        tcn_error_m = haversine_m(
            tcn_pred_latlon[:, 0], tcn_pred_latlon[:, 1],
            true_latlon[:, 0], true_latlon[:, 1],
            cfg.earth_radius_m,
        )
        tcn_error_yds = tcn_error_m * cfg.meters_to_yards

        # ----------------------------
        # Dead reckoning baseline
        # ----------------------------
        dr_dt = choose_dr_horizon(meta_test, cfg)
        dr_pred_latlon = forward_dead_reckoning(
            lat_deg=last_pos_test[:, 0],
            lon_deg=last_pos_test[:, 1],
            speed_knots=meta_test["last_speed"].to_numpy(dtype=np.float32),
            cog_deg=meta_test["last_cog"].to_numpy(dtype=np.float32),
            dt_seconds=dr_dt,
            R=cfg.earth_radius_m,
            knots_to_mps=cfg.knots_to_mps,
        )
        dr_error_m = haversine_m(
            dr_pred_latlon[:, 0], dr_pred_latlon[:, 1],
            true_latlon[:, 0], true_latlon[:, 1],
            cfg.earth_radius_m,
        )
        dr_error_yds = dr_error_m * cfg.meters_to_yards

        # ----------------------------
        # Results dataframe
        # ----------------------------
        dt_idx = FEATURE_COLS.index("dt")
        results_rows = []
        for i in range(len(pred_test)):
            tcn_err = float(tcn_error_yds[i])
            dr_err = float(dr_error_yds[i])
            improvement = dr_err - tcn_err
            if tcn_err < dr_err:
                winner = "TCN"
            elif dr_err < tcn_err:
                winner = "DR"
            else:
                winner = "TIE"

            results_rows.append({
                "idx": i,
                "row_id": meta_test.iloc[i]["row_id"],
                "MMSI": meta_test.iloc[i]["MMSI"],
                "voyage_id": meta_test.iloc[i]["voyage_id"],
                "anchor_time": meta_test.iloc[i]["anchor_time"],
                "pred_time": meta_test.iloc[i]["pred_time"],
                "num_pings": meta_test.iloc[i]["num_pings"],
                "input_dt": float(meta_test.iloc[i]["input_dt"]),
                "target_dt": float(meta_test.iloc[i]["target_dt"]),
                "delta_t_from_last_x": float(X_test_unscaled[i, -1, dt_idx]),
                "dr_horizon_dt": float(dr_dt[i]),
                "last_speed": float(meta_test.iloc[i]["last_speed"]),
                "last_cog": float(meta_test.iloc[i]["last_cog"]),
                "last_heading": float(meta_test.iloc[i]["last_heading"]),
                "last_lat": float(last_pos_test[i, 0]),
                "last_lon": float(last_pos_test[i, 1]),
                "tcn_pred_dlat": float(pred_test[i, 0]),
                "tcn_pred_dlon": float(pred_test[i, 1]),
                "true_dlat": float(y_test[i, 0]),
                "true_dlon": float(y_test[i, 1]),
                "tcn_pred_lat": float(tcn_pred_latlon[i, 0]),
                "tcn_pred_lon": float(tcn_pred_latlon[i, 1]),
                "dr_pred_lat": float(dr_pred_latlon[i, 0]),
                "dr_pred_lon": float(dr_pred_latlon[i, 1]),
                "true_lat": float(true_latlon[i, 0]),
                "true_lon": float(true_latlon[i, 1]),
                "tcn_error_m": float(tcn_error_m[i]),
                "dr_error_m": float(dr_error_m[i]),
                "tcn_error_yds": tcn_err,
                "dr_error_yds": dr_err,
                "improvement_yds": improvement,
                "abs_improvement_yds": abs(improvement),
                "winner": winner,
            })

        results_df = pd.DataFrame(results_rows)
        results_path = save_results(results_df, run_dir)

        # ----------------------------
        # Summaries
        # ----------------------------
        tcn_summary = summarize_error(results_df["tcn_error_yds"].to_numpy())
        dr_summary = summarize_error(results_df["dr_error_yds"].to_numpy())
        tcn_dist = compute_outlier_stats(results_df["tcn_error_yds"].dropna())
        dr_dist = compute_outlier_stats(results_df["dr_error_yds"].dropna())

        print_model_comparison(tcn_summary, dr_summary, logger)
        print_results_preview(results_df, logger)
        print_outlier_stats("TCN", tcn_dist, logger)
        print_outlier_stats("DR", dr_dist, logger)

        logger.log(f"\nSaved results to: {results_path}")
        logger.log(
            f"TCN better on {(results_df['winner'] == 'TCN').mean() * 100:.2f}% of test predictions | "
            f"DR better on {(results_df['winner'] == 'DR').mean() * 100:.2f}%"
        )

        # ----------------------------
        # Plots
        # ----------------------------
        plot_comparison_histogram(results_df, run_dir, bins=cfg.histogram_bins)
        plot_improvement_histogram(results_df, run_dir, bins=cfg.histogram_bins)
        save_train_loss_history(train_info["train_losses"], run_dir)

        # ----------------------------
        # Maps
        # ----------------------------
        comparison_map_path = None
        top_gap_map_files = []

        if cfg.save_comparison_map:
            comparison_map_path = save_comparison_map(
                results_df=results_df,
                out_path=run_dir / "tcn_vs_dr_comparison_map.html",
                max_points=cfg.map_max_points,
            )
            logger.log(f"Saved comparison map: {comparison_map_path}")

        if cfg.save_top_gap_maps:
            top_gap_map_files = save_top_gap_case_maps(
                results_df=results_df,
                out_dir=run_dir / "top_gap_maps",
                n_maps=cfg.top_gap_map_count,
            )
            logger.log(f"Saved {len(top_gap_map_files)} top-gap case maps.")

        # ----------------------------
        # Metrics summary
        # ----------------------------
        metrics_summary = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "seq_len": cfg.seq_len,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout,
            "kernel_size": cfg.kernel_size,
            "tcn_channels": list(cfg.tcn_channels),
            "dilations": list(cfg.dilations),
            "epochs_trained": train_info["epochs_trained"],
            "stopped_early": train_info["stopped_early"],
            "final_train_loss": train_info["final_train_loss"],
            "dr_horizon_mode": cfg.dr_horizon_mode,
            "num_test_predictions": int(len(results_df)),
            "tcn_rmse_yds": tcn_summary["rmse_yds"],
            "tcn_mae_yds": tcn_summary["mae_yds"],
            "tcn_median_yds": tcn_summary["median_yds"],
            "tcn_p95_yds": tcn_summary["p95_yds"],
            "dr_rmse_yds": dr_summary["rmse_yds"],
            "dr_mae_yds": dr_summary["mae_yds"],
            "dr_median_yds": dr_summary["median_yds"],
            "dr_p95_yds": dr_summary["p95_yds"],
            "tcn_q1_yds": tcn_dist["q1"],
            "tcn_q3_yds": tcn_dist["q3"],
            "tcn_iqr_yds": tcn_dist["iqr"],
            "dr_q1_yds": dr_dist["q1"],
            "dr_q3_yds": dr_dist["q3"],
            "dr_iqr_yds": dr_dist["iqr"],
            "tcn_outlier_percent": tcn_dist["outlier_pct"],
            "dr_outlier_percent": dr_dist["outlier_pct"],
            "mean_improvement_yds": float(results_df["improvement_yds"].mean()),
            "median_improvement_yds": float(results_df["improvement_yds"].median()),
            "tcn_win_percent": float((results_df["winner"] == "TCN").mean() * 100),
            "dr_win_percent": float((results_df["winner"] == "DR").mean() * 100),
            "results_df_path": str(results_path),
            "comparison_map_path": str(comparison_map_path) if comparison_map_path else "",
            "top_gap_map_count": len(top_gap_map_files),
            "train_loss_history_path": str(run_dir / "train_loss_history.csv"),
            "run_log_path": str(run_dir / "run_log.txt"),
        }
        save_metrics_summary(metrics_summary, run_dir)
        append_experiment_summary(metrics_summary, experiments_dir)

        logger.log("\nRun complete.")
        logger.log(f"Artifacts saved under: {run_dir}")

        return {
            "run_dir": run_dir,
            "results_df": results_df,
            "metrics_summary": metrics_summary,
            "train_losses": train_info["train_losses"],
            "epochs_trained": train_info["epochs_trained"],
            "comparison_map_path": comparison_map_path,
            "top_gap_map_files": top_gap_map_files,
        }

    finally:
        logger.close()


# ============================================================
# ONE-OFF MANUAL RUN
# ============================================================

if __name__ == "__main__":
    artifacts = run_experiment(CFG)
    print("\nMain comparison map:")
    print(artifacts["comparison_map_path"])
