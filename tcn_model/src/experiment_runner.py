from __future__ import annotations

import traceback
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from src.tcn_refactor import Config, run_experiment


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "experiments" / "runner_status.csv"


# ============================================================
# BASE CONFIG
# ============================================================

def get_base_config() -> Config:
    return Config(
        train_path="data/raw/gold_train_data_with_num_pings.csv",
        test_path="data/raw/gold_test_data_with_num_pings.csv",
        experiments_dir="experiments",
        seq_len=19,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.25,
        kernel_size=3,
        tcn_channels=(32, 64, 64),
        dilations=(1, 2, 4),
        max_epochs=100,
        min_epochs=5,
        loss_stability_threshold=0.02,
        histogram_bins=300,
        large_error_threshold=20000.0,
        save_outlier_maps=True,
    )


# ============================================================
# GENERAL MANUAL SWEEP BUILDER
# ============================================================

def build_manual_sweep(base_cfg: Config, param_grid: Dict[str, List]) -> List[Config]:
    """
    param_grid example:
    {
        "seq_len": [8, 12, 16],
        "learning_rate": [1e-3, 3e-4],
        "batch_size": [16, 32],
        "dropout": [0.10, 0.25],
    }
    """
    if not param_grid:
        return [base_cfg]

    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    cfgs = []
    for combo in product(*param_values):
        updates = dict(zip(param_names, combo))
        cfgs.append(replace(base_cfg, **updates))

    return cfgs


# ============================================================
# RUNNER HELPERS
# ============================================================

def config_signature(cfg: Config) -> str:
    return (
        f"seq={cfg.seq_len} | "
        f"bs={cfg.batch_size} | "
        f"lr={cfg.learning_rate} | "
        f"wd={cfg.weight_decay} | "
        f"dropout={cfg.dropout} | "
        f"ks={cfg.kernel_size} | "
        f"channels={cfg.tcn_channels} | "
        f"dilations={cfg.dilations}"
    )


def append_runner_status(row: Dict):
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([row])

    if SUMMARY_PATH.exists():
        existing = pd.read_csv(SUMMARY_PATH)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(SUMMARY_PATH, index=False)


def run_config_list(configs: Iterable[Config], sweep_name: str):
    configs = list(configs)
    print(f"\n=== RUNNING SWEEP: {sweep_name} ===")
    print(f"Total runs: {len(configs)}")

    for i, cfg in enumerate(configs, start=1):
        print("\n" + "=" * 70)
        print(f"Run {i}/{len(configs)}")
        print(config_signature(cfg))
        print("=" * 70)

        try:
            artifacts = run_experiment(cfg)
            metrics = artifacts["metrics_summary"]

            append_runner_status({
                "sweep_name": sweep_name,
                "status": "success",
                "seq_len": cfg.seq_len,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "dropout": cfg.dropout,
                "kernel_size": cfg.kernel_size,
                "tcn_channels": str(cfg.tcn_channels),
                "dilations": str(cfg.dilations),
                "epochs_trained": metrics["epochs_trained"],
                "stopped_early": metrics["stopped_early"],
                "final_train_loss": metrics["final_train_loss"],
                "rmse_m": metrics["rmse_m"],
                "mae_m": metrics["mae_m"],
                "median_m": metrics["median_m"],
                "p95_m": metrics["p95_m"],
                "outlier_count": metrics["outlier_count"],
                "outlier_percent": metrics["outlier_percent"],
                "large_error_count": metrics["large_error_count"],
                "run_name": metrics["run_name"],
                "run_dir": metrics["run_dir"],
                "error_message": "",
            })

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"\nFAILED: {err}")
            traceback.print_exc()

            append_runner_status({
                "sweep_name": sweep_name,
                "status": "failed",
                "seq_len": cfg.seq_len,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "dropout": cfg.dropout,
                "kernel_size": cfg.kernel_size,
                "tcn_channels": str(cfg.tcn_channels),
                "dilations": str(cfg.dilations),
                "epochs_trained": None,
                "stopped_early": None,
                "final_train_loss": None,
                "rmse_m": None,
                "mae_m": None,
                "median_m": None,
                "p95_m": None,
                "outlier_count": None,
                "outlier_percent": None,
                "large_error_count": None,
                "run_name": "",
                "run_dir": "",
                "error_message": err,
            })


# ============================================================
# MAIN
# ============================================================

def main():
    base_cfg = get_base_config()

    # Manually define the exact values you want to test
    param_grid = {
        "seq_len": [4, 8, 12, 16, 24],
        "learning_rate": [1e-4],
        "batch_size": [256],
        "dropout": [0.30],
        "kernel_size": [5],
        "tcn_channels": [(64, 128, 256), (128, 256, 512)],
        "dilations": [(1, 2, 4, 8)]
    }

    configs = build_manual_sweep(base_cfg, param_grid)
    run_config_list(configs, sweep_name="manual_param_grid")


if __name__ == "__main__":
    main()

#  python3 -m src.experiment_runner
