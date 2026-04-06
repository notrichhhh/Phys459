#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:40:10 2026

@author: Rich
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

base_dir = "rot=100_guassian_mdot=3.2_6000s_100times"

y_task = "y"
F_task = "F"

y_csv = os.path.join(base_dir, "phi0_" + y_task + ".csv")
F_csv = os.path.join(base_dir, "phi0_" + F_task + ".csv")
theta_csv = "theta_master.csv"

parts = base_dir.split("_")

# Directly choose theta indices
pole_index = 0
equator_index = 255

# Plot options
if_log_y = True
if_log_F = True
save_fig = True
fig_name = f"y_pole_y_equator_integratedF_{parts[2]}.png"

manual_ylim_top = None
manual_ylim_mid = None
manual_ylim_bot = None

figsize = (5, 8)
dpi = 150
line_width = 1.8

title_size = 15
label_size = 15
tick_size = 14


# ============================================================
# HELPERS
# ============================================================

def _load_theta(theta_csv):
    theta = np.loadtxt(theta_csv, delimiter=",")
    if theta.ndim == 2:
        theta = theta[0]
    return np.array(theta, dtype=float)


def _load_csv_flexible(data_csv):
    raw = np.loadtxt(data_csv, delimiter=",")

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if raw.shape[1] >= 3:
        write = raw[:, 0]
        sim_time = raw[:, 1]
        data = raw[:, 2:]
        return write, sim_time, data, True

    time_or_write = raw[:, 0]
    data = raw[:, 1:]
    return time_or_write, None, data, False


def _global_min_positive(arr):
    arr = np.asarray(arr, dtype=float)
    pos = arr[arr > 0]
    if pos.size == 0:
        raise RuntimeError("No positive values found → cannot take log.")
    return float(np.min(pos))


# ============================================================
# MAIN
# ============================================================

def main():
    theta = _load_theta(theta_csv)

    # -------------------------
    # Load y data
    # -------------------------
    x_y, sim_time_y, y_data, _ = _load_csv_flexible(y_csv)

    if y_data.shape[1] != theta.size:
        ntheta = min(y_data.shape[1], theta.size)
        print(f"[warn] y/theta mismatch → truncating to {ntheta}")
        theta_y = theta[:ntheta]
        y_data = y_data[:, :ntheta]
    else:
        theta_y = theta

    if pole_index < 0 or pole_index >= y_data.shape[1]:
        raise IndexError(f"pole_index={pole_index} out of bounds (Ntheta={y_data.shape[1]})")
    if equator_index < 0 or equator_index >= y_data.shape[1]:
        raise IndexError(f"equator_index={equator_index} out of bounds (Ntheta={y_data.shape[1]})")

    if sim_time_y is not None:
        t = sim_time_y
        xlabel = "t (s)"
    else:
        t = x_y
        xlabel = "time / frame index"
        print("[warn] no sim_time column in y CSV → using first column as time")

    y_pole = np.array(y_data[:, pole_index], dtype=float)
    y_equator = np.array(y_data[:, equator_index], dtype=float)

    theta_pole_val = float(theta_y[pole_index])
    theta_equator_val = float(theta_y[equator_index])

    # -------------------------
    # Load F data
    # -------------------------
    x_F, sim_time_F, F_data, _ = _load_csv_flexible(F_csv)

    if F_data.shape[1] != theta.size:
        ntheta = min(F_data.shape[1], theta.size)
        print(f"[warn] F/theta mismatch → truncating to {ntheta}")
        theta_F = theta[:ntheta]
        F_data = F_data[:, :ntheta]
    else:
        theta_F = theta

    # Optional consistency checks
    if len(t) != F_data.shape[0]:
        ntime = min(len(t), F_data.shape[0])
        print(f"[warn] y/F time-length mismatch → truncating to {ntime}")
        t = t[:ntime]
        y_pole = y_pole[:ntime]
        y_equator = y_equator[:ntime]
        F_data = F_data[:ntime, :]
        if sim_time_F is not None:
            sim_time_F = sim_time_F[:ntime]

    # -------------------------
    # Integrated flux
    # -------------------------
    sin_theta = np.sin(theta_F)
    denom = np.trapz(sin_theta, theta_F)

    if denom == 0:
        raise RuntimeError("Integral of sin(theta) is zero; check theta grid.")

    integrated_F = np.trapz(F_data * sin_theta[None, :], theta_F, axis=1) / denom

    # -------------------------
    # Log options
    # -------------------------
    if if_log_y:
        floor_y_pole = _global_min_positive(y_pole)
        floor_y_equator = _global_min_positive(y_equator)

        print(f"[log] y_pole floor    = {floor_y_pole:.3e}")
        print(f"[log] y_equator floor = {floor_y_equator:.3e}")

        y_pole_plot = np.log10(np.where(y_pole > floor_y_pole, y_pole, floor_y_pole))
        y_equator_plot = np.log10(np.where(y_equator > floor_y_equator, y_equator, floor_y_equator))

        ylabel_top = r"$\log_{10}(y)$"
        ylabel_mid = r"$\log_{10}(y)$"
    else:
        y_pole_plot = y_pole
        y_equator_plot = y_equator
        ylabel_top = r"$y$"
        ylabel_mid = r"$y$"

    if if_log_F:
        floor_F = _global_min_positive(integrated_F)
        print(f"[log] integrated_F floor = {floor_F:.3e}")
        integrated_F_plot = np.log10(np.where(integrated_F > floor_F, integrated_F, floor_F))
        ylabel_bot = r"$\log_{10}(F^*)$"
    else:
        integrated_F_plot = integrated_F
        ylabel_bot = r"F^*"

    # -------------------------
    # Plot
    # -------------------------
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Top: y at pole
    axes[0].plot(t, y_pole_plot, linewidth=line_width, color='C0')
    axes[0].set_ylabel(ylabel_top, fontsize=label_size)
    axes[0].set_title(
        rf"d) $y$ at pole",
        fontsize=title_size
    )
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis='x', labelsize=tick_size)
    axes[0].tick_params(axis='y', labelsize=tick_size)
    if manual_ylim_top is not None:
        axes[0].set_ylim(*manual_ylim_top)

    # Middle: y at equator
    axes[1].plot(t, y_equator_plot, linewidth=line_width, color='C1')
    axes[1].set_ylabel(ylabel_mid, fontsize=label_size)
    axes[1].set_title(
        rf"e) $y$ at equator",
        fontsize=title_size
    )
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis='x', labelsize=tick_size)
    axes[1].tick_params(axis='y', labelsize=tick_size)
    if manual_ylim_mid is not None:
        axes[1].set_ylim(*manual_ylim_mid)

    # Bottom: integrated F
    axes[2].plot(t, integrated_F_plot, linewidth=line_width, color='C2')
    axes[2].set_xlabel(xlabel, fontsize=label_size)
    axes[2].set_ylabel(ylabel_bot, fontsize=label_size)
    axes[2].set_title("f) $F^*$", fontsize=title_size)
    axes[2].grid(alpha=0.3)
    axes[2].tick_params(axis='x', labelsize=tick_size)
    axes[2].tick_params(axis='y', labelsize=tick_size)
    if manual_ylim_bot is not None:
        axes[2].set_ylim(*manual_ylim_bot)

    fig.tight_layout()

    if save_fig:
        fig.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        print(f"[plot] saved → {fig_name}")

    plt.show()

    # -------------------------
    # Info
    # -------------------------
    print(f"[info] pole index        = {pole_index}")
    print(f"[info] pole theta value  = {theta_pole_val:.12f} rad")
    print(f"[info] equator index     = {equator_index}")
    print(f"[info] equator theta val = {theta_equator_val:.12f} rad")
    print(f"[info] N time points     = {len(t)}")
    print(f"[info] N theta (y)       = {len(theta_y)}")
    print(f"[info] N theta (F)       = {len(theta_F)}")


if __name__ == "__main__":
    main()