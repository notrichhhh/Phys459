#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 23:13:25 2026

@author: Rich
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================



base_dir = "rot=100_guassian_mdot=3.2_6000s_100times"
data = 'y'
task = data

parts = base_dir.split("_")

data_csv = os.path.join(base_dir, "phi0_"+data+".csv")
theta_csv = "theta_master.csv"

# Directly choose theta index
theta_index = 5  # <<< YOU CONTROL THIS

# Plot options
if_log = True
save_fig = True
fig_name = data+f"_theta={theta_index}_{parts[2]}.png"

manual_ylim = None

figsize = (7, 5)
dpi = 150
line_width = 1.8


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
    x, sim_time, data, is_new_format = _load_csv_flexible(data_csv)

    # Safety check
    if theta_index < 0 or theta_index >= data.shape[1]:
        raise IndexError(f"theta_index={theta_index} out of bounds (Ntheta={data.shape[1]})")

    # Match theta size if needed
    if data.shape[1] != theta.size:
        ntheta = min(data.shape[1], theta.size)
        print(f"[warn] theta/data mismatch → truncating to {ntheta}")
        theta = theta[:ntheta]
        data = data[:, :ntheta]

    # Extract time axis
    if sim_time is not None:
        t = sim_time
        xlabel = "time (s)"
    else:
        t = x
        xlabel = "time / frame index"
        print("[warn] no sim_time column → using first column as time")

    # Extract data at chosen theta index
    y = np.array(data[:, theta_index], dtype=float)
    theta_val = float(theta[theta_index])

    # Log option
    if if_log:
        log_floor = _global_min_positive(y)
        print(f"[log] floor = {log_floor:.3e}")
        y_plot = np.log10(np.where(y > log_floor, y, log_floor))
        ylabel = f"log10({task})"
    else:
        y_plot = y
        ylabel = task

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(t, y_plot, linewidth=line_width)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{task} at theta index = {theta_index} (theta = {theta_val:.6f} rad)")

    if manual_ylim is not None:
        plt.ylim(*manual_ylim)

    plt.tight_layout()

    if save_fig:
        plt.savefig(fig_name, dpi=dpi)
        print(f"[plot] saved → {fig_name}")

    plt.show()

    print(f"[info] theta index = {theta_index}")
    print(f"[info] theta value = {theta_val:.12f} rad")
    print(f"[info] N time points = {len(t)}")


if __name__ == "__main__":
    main()