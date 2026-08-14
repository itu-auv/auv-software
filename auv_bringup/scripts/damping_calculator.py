#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
damping_calculator_wrench.py

Fits linear + quadratic damping from wrench-step test CSV files.
This does not need to be a ROS node; it can live under scripts/ in a ROS
package and run as a standalone Python executable with rosrun.

Model:
    tau = dL * nu + dQ * abs(nu) * nu + bias

Expected CSV columns:
    t, axis_name, phase, target,
    cmd_force_x, cmd_force_y, cmd_torque_z,
    odom_lin_x, odom_lin_y, odom_lin_z,
    odom_ang_x, odom_ang_y, odom_ang_z,
    roll, pitch, yaw,
    wrench_force_x, wrench_force_y, wrench_force_z,
    wrench_torque_x, wrench_torque_y, wrench_torque_z

Columns used:
    surge: nu=odom_lin_x,  tau=cmd_force_x   or wrench_force_x
    sway : nu=odom_lin_y,  tau=cmd_force_y   or wrench_force_y
    yaw  : nu=odom_ang_z,  tau=cmd_torque_z  or wrench_torque_z

Outputs:
    - Fit values in the terminal
    - One PNG fit plot per axis under figures/
"""

import argparse
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


EPS = np.finfo(float).eps
AXES_WANTED = ("surge", "sway", "yaw")


def date_str() -> str:
    return datetime.now().strftime("%Y%m%d")


# ===================== SETTINGS =====================
DATA_DIR = str(Path("~/damping_tests") / date_str())
OUTPUT_DIR = None  # Use DATA_DIR when this is None.

USE_ONLY_RECORD = True  # Use only the RECORD phase.
FIT_BIAS = False  # tau = dL*nu + dQ*|nu|nu + bias
TRIM_PCT = 10.0  # Trimmed mean percentage.
MIN_N = 50  # Minimum sample count.

# Use commanded tau on the tested axis. False uses logged wrench.
USE_COMMANDED_TAU = False

# Trim the edges of the RECORD phase.
TRIM_START_SEC = 1.0
TRIM_END_SEC = 1.0

# Drop samples when cross-axis motion is too large.
ENABLE_CROSS_AXIS_FILTER = True
CROSS_VEL_RATIO_MAX = 0.35  # Linear cross-axis / main-axis.
CROSS_YAW_RATE_MAX = 0.35  # Yaw rate / main-axis scale.

# Plotting.
SHOW_PLOTS = False  # Also show the plot on screen in addition to saving PNG.


class Config:
    def __init__(self, args: argparse.Namespace):
        self.data_dir = Path(args.data_dir).expanduser().resolve()
        self.output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else self.data_dir
        )

        self.use_only_record = args.use_only_record
        self.fit_bias = args.fit_bias
        self.trim_pct = args.trim_pct
        self.min_n = args.min_n
        self.use_commanded_tau = args.use_commanded_tau

        self.trim_start_sec = args.trim_start_sec
        self.trim_end_sec = args.trim_end_sec

        self.enable_cross_axis_filter = args.enable_cross_axis_filter
        self.cross_vel_ratio_max = args.cross_vel_ratio_max
        self.cross_yaw_rate_max = args.cross_yaw_rate_max

        self.show_plots = args.show_plots


class FitResult:
    def __init__(
        self,
        d_l: float,
        d_q: float,
        bias: float,
        r2: float,
        nu: np.ndarray,
        tau: np.ndarray,
    ):
        self.d_l = float(d_l)
        self.d_q = float(d_q)
        self.bias = float(bias)
        self.r2 = float(r2)
        self.nu = np.asarray(nu, dtype=float)
        self.tau = np.asarray(tau, dtype=float)


def trimmed_mean(x: np.ndarray, trim_pct: float) -> float:
    """Python equivalent of the MATLAB trimmedMean helper."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x.reshape(-1))
    n = x.size
    if n == 0:
        return float("nan")

    k = int(math.floor(n * (trim_pct / 100.0) / 2.0))
    if n <= 2 * k:
        return float(np.nanmean(x))
    return float(np.nanmean(x[k : n - k]))


def require_columns(df: pd.DataFrame, file_name: str) -> None:
    must_have = [
        "axis_name",
        "phase",
        "target",
        "cmd_force_x",
        "cmd_force_y",
        "cmd_torque_z",
        "odom_lin_x",
        "odom_lin_y",
        "odom_ang_z",
        "wrench_force_x",
        "wrench_force_y",
        "wrench_torque_z",
    ]
    missing = [c for c in must_have if c not in df.columns]
    if missing:
        available = ", ".join(map(str, df.columns))
        raise RuntimeError(
            f"Missing expected CSV columns. File={file_name}\n"
            f"Missing columns: {missing}\n"
            f"Available columns: {available}"
        )


def first_axis_name(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return str(df["axis_name"].iloc[0]).strip().lower()


def as_float_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def process_csv_file(csv_path: Path, cfg: Config) -> Optional[dict]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        warnings.warn(f"Could not read: {csv_path.name} ({exc})")
        return None

    if "axis_name" not in df.columns or "phase" not in df.columns:
        return None

    require_columns(df, csv_path.name)

    axis_name = first_axis_name(df)

    if cfg.use_only_record:
        phase = df["phase"].astype(str).str.upper()
        df = df.loc[phase == "RECORD"].copy()

    if len(df) < cfg.min_n:
        warnings.warn(f"Too few samples, skipped: {csv_path.name} (N={len(df)})")
        return None

    if "t" in df.columns:
        t = as_float_array(df["t"])
        t = t - t[0]
    else:
        t = np.arange(len(df), dtype=float)

    max_t = float(np.nanmax(t)) if t.size else 0.0
    keep_time = (t >= cfg.trim_start_sec) & (t <= (max_t - cfg.trim_end_sec))
    df = df.loc[keep_time].copy()
    t = t[keep_time]

    if len(df) < cfg.min_n:
        warnings.warn(
            f"Too few samples after trimming, skipped: {csv_path.name} (N={len(df)})"
        )
        return None

    if axis_name == "surge":
        nu = as_float_array(df["odom_lin_x"])
        tau = as_float_array(
            df["cmd_force_x"] if cfg.use_commanded_tau else df["wrench_force_x"]
        )
        cross1 = as_float_array(df["odom_lin_y"])
        cross2 = as_float_array(df["odom_ang_z"])

    elif axis_name == "sway":
        nu = as_float_array(df["odom_lin_y"])
        tau = as_float_array(
            df["cmd_force_y"] if cfg.use_commanded_tau else df["wrench_force_y"]
        )
        cross1 = as_float_array(df["odom_lin_x"])
        cross2 = as_float_array(df["odom_ang_z"])

    elif axis_name == "yaw":
        nu = as_float_array(df["odom_ang_z"])
        tau = as_float_array(
            df["cmd_torque_z"] if cfg.use_commanded_tau else df["wrench_torque_z"]
        )
        odom_lin_x = as_float_array(df["odom_lin_x"])
        odom_lin_y = as_float_array(df["odom_lin_y"])
        cross1 = np.hypot(odom_lin_x, odom_lin_y)
        cross2 = np.zeros_like(nu)

    else:
        warnings.warn(f"Unknown axis_name '{axis_name}', skipped: {csv_path.name}")
        return None

    ok = np.isfinite(nu) & np.isfinite(tau) & np.isfinite(cross1) & np.isfinite(cross2)
    nu = nu[ok]
    tau = tau[ok]
    cross1 = cross1[ok]
    cross2 = cross2[ok]

    if nu.size < cfg.min_n:
        warnings.warn(
            f"Too few samples after NaN cleanup, skipped: {csv_path.name} (N={nu.size})"
        )
        return None

    if cfg.enable_cross_axis_filter:
        main_scale = float(np.nanmax(np.abs(nu))) if nu.size else 1.0
        if main_scale < EPS:
            main_scale = 1.0

        if axis_name in ("surge", "sway"):
            mask_cross = (np.abs(cross1) <= cfg.cross_vel_ratio_max * main_scale) & (
                np.abs(cross2) <= cfg.cross_yaw_rate_max * main_scale
            )
        elif axis_name == "yaw":
            yaw_scale = float(np.nanmax(np.abs(nu) + EPS)) if nu.size else 1.0
            mask_cross = np.abs(cross1) <= cfg.cross_vel_ratio_max * yaw_scale
        else:
            mask_cross = np.ones_like(nu, dtype=bool)

        # Same as the MATLAB code: do not apply the filter if it is too aggressive.
        if np.count_nonzero(mask_cross) >= max(
            cfg.min_n, int(math.floor(0.4 * mask_cross.size))
        ):
            nu = nu[mask_cross]
            tau = tau[mask_cross]

    if nu.size < cfg.min_n:
        warnings.warn(
            f"Too few samples after cross-axis filtering, skipped: {csv_path.name} (N={nu.size})"
        )
        return None

    target_val = pd.to_numeric(df["target"], errors="coerce").iloc[0]

    return {
        "file": csv_path.name,
        "axis": axis_name,
        "target": float(target_val) if pd.notna(target_val) else float("nan"),
        "nu": trimmed_mean(nu, cfg.trim_pct),
        "tau": trimmed_mean(tau, cfg.trim_pct),
        "N": int(nu.size),
    }


def fit_axis(
    summary: pd.DataFrame, axis_name: str, fit_bias: bool
) -> Optional[FitResult]:
    data = summary.loc[summary["axis"] == axis_name]
    nu = data["nu"].to_numpy(dtype=float)
    tau = data["tau"].to_numpy(dtype=float)

    ok = np.isfinite(nu) & np.isfinite(tau)
    nu = nu[ok]
    tau = tau[ok]

    if nu.size < 3:
        return None

    cols = [nu, np.abs(nu) * nu]
    if fit_bias:
        cols.append(np.ones_like(nu))
    A = np.column_stack(cols)

    theta, *_ = np.linalg.lstsq(A, tau, rcond=None)
    tau_hat = A @ theta

    ss_res = float(np.sum((tau - tau_hat) ** 2))
    ss_tot = float(np.sum((tau - np.mean(tau)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, EPS)

    d_l = theta[0]
    d_q = theta[1]
    bias = theta[2] if fit_bias else 0.0

    return FitResult(d_l=d_l, d_q=d_q, bias=bias, r2=r2, nu=nu, tau=tau)


def ensure_matplotlib(show_plots: bool):
    cache_dir = Path("/tmp/matplotlib")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    if not show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_axis_fit(
    axis_name: str, result: FitResult, out_dir: Path, show_plots: bool
) -> None:
    plt = ensure_matplotlib(show_plots)

    nu = result.nu
    tau = result.tau

    nu_grid = np.linspace(float(np.min(nu)), float(np.max(nu)), 200)
    tau_grid = (
        result.d_l * nu_grid + result.d_q * np.abs(nu_grid) * nu_grid + result.bias
    )

    fig, ax = plt.subplots(figsize=(9, 5.4))
    (
        fig.canvas.manager.set_window_title(f"Damping Fit {axis_name}")
        if hasattr(fig.canvas, "manager")
        else None
    )

    ax.plot(nu, tau, "bo", markersize=7, linewidth=1.2, label="data (file mean)")
    ax.plot(nu_grid, tau_grid, "r-", linewidth=1.8, label="fit")
    ax.grid(True)
    ax.set_xlabel("nu")
    ax.set_ylabel("tau")
    ax.set_title(
        f"{axis_name} fit: dL={result.d_l:.6g}, dQ={result.d_q:.6g}, "
        f"bias={result.bias:.6g}, R^2={result.r2:.4f}"
    )
    ax.legend(loc="best")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"damping_fit_{axis_name}.png", dpi=300)

    if show_plots:
        plt.show(block=False)
    else:
        plt.close(fig)


def print_results(fit: Dict[str, FitResult]) -> None:
    for axis in AXES_WANTED:
        if axis not in fit:
            continue
        result = fit[axis]
        print(
            f"[{axis}] dL={result.d_l:.6g}  dQ={result.d_q:.6g}  "
            f"bias={result.bias:.6g}  R2={result.r2:.4f}"
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit linear + quadratic damping from wrench-step CSV files."
    )

    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help=f"Directory containing CSV files. Default: {DATA_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory where outputs are saved. Default: data-dir",
    )

    parser.set_defaults(use_only_record=USE_ONLY_RECORD)
    parser.add_argument(
        "--use-only-record",
        dest="use_only_record",
        action="store_true",
        help="Use only the RECORD phase.",
    )
    parser.add_argument(
        "--no-use-only-record",
        dest="use_only_record",
        action="store_false",
        help="Disable the RECORD filter and use all phases.",
    )

    parser.set_defaults(enable_cross_axis_filter=ENABLE_CROSS_AXIS_FILTER)
    parser.add_argument(
        "--cross-axis-filter",
        dest="enable_cross_axis_filter",
        action="store_true",
        help="Enable cross-axis filtering.",
    )
    parser.add_argument(
        "--no-cross-axis-filter",
        dest="enable_cross_axis_filter",
        action="store_false",
        help="Disable cross-axis filtering.",
    )

    parser.set_defaults(use_commanded_tau=USE_COMMANDED_TAU)
    parser.add_argument(
        "--use-commanded-tau",
        dest="use_commanded_tau",
        action="store_true",
        help="Use commanded force/torque for tau instead of logged wrench.",
    )
    parser.add_argument(
        "--use-logged-tau",
        dest="use_commanded_tau",
        action="store_false",
        help="Use logged wrench force/torque for tau.",
    )
    parser.set_defaults(fit_bias=FIT_BIAS)
    parser.add_argument(
        "--fit-bias",
        dest="fit_bias",
        action="store_true",
        help="Fit tau = dL*nu + dQ*|nu|nu + bias.",
    )
    parser.add_argument(
        "--no-fit-bias",
        dest="fit_bias",
        action="store_false",
        help="Remove the bias term from the fit.",
    )
    parser.set_defaults(show_plots=SHOW_PLOTS)
    parser.add_argument(
        "--show-plots",
        dest="show_plots",
        action="store_true",
        help="Show plots on screen. Avoid this on SSH/headless systems.",
    )
    parser.add_argument(
        "--no-show-plots",
        dest="show_plots",
        action="store_false",
        help="Do not show plots on screen; save PNG files only.",
    )

    parser.add_argument(
        "--trim-pct",
        type=float,
        default=TRIM_PCT,
        help=f"Trimmed mean percentage. Default: {TRIM_PCT:g}",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=MIN_N,
        help=f"Minimum sample count. Default: {MIN_N}",
    )
    parser.add_argument(
        "--trim-start-sec",
        type=float,
        default=TRIM_START_SEC,
        help=f"Seconds to trim from the start of RECORD. Default: {TRIM_START_SEC:g}",
    )
    parser.add_argument(
        "--trim-end-sec",
        type=float,
        default=TRIM_END_SEC,
        help=f"Seconds to trim from the end of RECORD. Default: {TRIM_END_SEC:g}",
    )
    parser.add_argument(
        "--cross-vel-ratio-max",
        type=float,
        default=CROSS_VEL_RATIO_MAX,
        help=f"Cross linear velocity / main-axis limit. Default: {CROSS_VEL_RATIO_MAX:g}",
    )
    parser.add_argument(
        "--cross-yaw-rate-max",
        type=float,
        default=CROSS_YAW_RATE_MAX,
        help=f"Cross yaw rate / main-axis scale limit. Default: {CROSS_YAW_RATE_MAX:g}",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = Config(args)

    if not cfg.data_dir.exists():
        print(f"ERROR: Directory not found: {cfg.data_dir}", file=sys.stderr)
        return 2

    csv_files = sorted(cfg.data_dir.glob("*.csv"))
    if not csv_files:
        print(
            f"ERROR: No CSV files found in this directory: {cfg.data_dir}",
            file=sys.stderr,
        )
        return 2

    rows = []
    for csv_path in csv_files:
        row = process_csv_file(csv_path, cfg)
        if row is not None:
            rows.append(row)

    if not rows:
        print(
            "ERROR: No data collected. Check the phase filter, column names, and RECORD rows.",
            file=sys.stderr,
        )
        return 1

    summary = pd.DataFrame(rows, columns=["file", "axis", "target", "nu", "tau", "N"])

    fit: Dict[str, FitResult] = {}
    for axis in AXES_WANTED:
        result = fit_axis(summary, axis, cfg.fit_bias)
        if result is not None:
            fit[axis] = result

    if not fit:
        print("ERROR: Could not produce a fit for any axis.", file=sys.stderr)
        return 1

    print_results(fit)

    figure_dir = cfg.output_dir / "figures"
    for axis, result in fit.items():
        plot_axis_fit(axis, result, figure_dir, cfg.show_plots)

    if cfg.show_plots:
        input("Press Enter to close the plots...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
