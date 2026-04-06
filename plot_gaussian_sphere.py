#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the first time increment of an axisymmetric phi0_<task>.csv
onto a full sphere by copying the theta profile uniformly in phi.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # remove if you want interactive plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# USER SETTINGS
# ============================================================

base_dir = "rot=100_guassian_mdot=3.2_6000s_100times"
data = 'ydot'
task = data

parts = base_dir.split("_")

data_csv = os.path.join(base_dir, "phi0_" + data + ".csv")
theta_csv = "theta_master.csv"

# Which time row to use
time_index = 0   # first time increment

# Sphere resolution in phi
nphi = 128

# Plot options
if_log = False
save_fig = True
fig_name = data + f"_sphere_tindex={time_index}_{parts[2]}.png"

figsize = (5, 5)
dpi = 150
title_size = 15
cbar_label_size = 14
cbar_tick_size = 13

cmap = plt.cm.viridis


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


def build_s2_coord_vertices(phi, theta):
    """
    Build vertex arrays from cell-centered phi/theta coordinates,
    matching the sphere-plotting logic style.
    """
    phi = np.ravel(phi)
    phi_vert = np.concatenate([phi, [2*np.pi]])
    phi_vert -= (phi[1] - phi[0]) / 2

    theta = np.ravel(theta)
    theta_mid = (theta[:-1] + theta[1:]) / 2
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])

    return np.meshgrid(phi_vert, theta_vert, indexing='ij')


# ============================================================
# MAIN
# ============================================================

def main():
    theta = _load_theta(theta_csv)
    x, sim_time, data_arr, is_new_format = _load_csv_flexible(data_csv)

    if time_index < 0 or time_index >= data_arr.shape[0]:
        raise IndexError(f"time_index={time_index} out of bounds (Ntimes={data_arr.shape[0]})")

    # Match theta size if needed
    if data_arr.shape[1] != theta.size:
        ntheta = min(data_arr.shape[1], theta.size)
        print(f"[warn] theta/data mismatch → truncating to {ntheta}")
        theta = theta[:ntheta]
        data_arr = data_arr[:, :ntheta]

    # Extract metadata for chosen row
    write_number = x[time_index]
    if sim_time is not None:
        t = float(sim_time[time_index])
    else:
        t = None
        print("[warn] no sim_time column → only write/frame index available")

    # Extract theta profile for chosen time
    profile_theta = np.array(data_arr[time_index, :], dtype=float)

    # Build axisymmetric 2D sphere data by copying in phi
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    sphere_data = np.tile(profile_theta, (nphi, 1))

    # Build sphere coordinates
    phi_vert, theta_vert = build_s2_coord_vertices(phi, theta)
    x_s = np.sin(theta_vert) * np.cos(phi_vert)
    y_s = np.sin(theta_vert) * np.sin(phi_vert)
    z_s = np.cos(theta_vert)

    # Color handling
    finite_vals = sphere_data[np.isfinite(sphere_data)]
    if finite_vals.size == 0:
        raise RuntimeError("No finite values found in selected time slice.")

    if if_log:
        vmin = _global_min_positive(sphere_data)
        pos_vals = finite_vals[finite_vals > 0]
        vmax = float(np.max(pos_vals))
        norm = LogNorm(vmin=vmin, vmax=vmax)

        print(f"[log] floor = {vmin:.3e}")

        data_plot = np.where(sphere_data > 0, sphere_data, np.nan)
        tmp = np.where(np.isfinite(data_plot), data_plot, vmin)
        fc = cmap(norm(tmp))
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        cbar_label = f"{task} [log]"
    else:
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
        if np.isclose(vmax, vmin):
            vmax = vmin + 1e-30

        tmp = np.clip(sphere_data, vmin, vmax)
        fc = cmap((tmp - vmin) / (vmax - vmin))
        mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        cbar_label = '$\dot{m} [\mathrm{g~cm^{-2}~s^{-1}}]$'

    mappable.set_array([])

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.00, 0.00, 0.80, 1.00], projection='3d')

    surf = ax.plot_surface(
        x_s, y_s, z_s,
        facecolors=fc,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False
    )

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    ax.axis('off')

    title = 'b) Gaussian Accretion on Sphere'

    ax.set_title(title, fontsize=title_size, pad=5)

    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.ticker import ScalarFormatter

    # Colorbar
    cax = fig.add_axes([0.83, 0.15, 0.03, 0.70])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(cbar_label, fontsize=cbar_label_size)
    cb.ax.tick_params(labelsize=cbar_tick_size)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # always scientific
    cb.ax.yaxis.set_major_formatter(formatter)
    cb.update_ticks()

    plt.tight_layout()

    if save_fig:
        plt.savefig(fig_name, dpi=dpi, bbox_inches='tight')
        print(f"[plot] saved → {fig_name}")

    plt.show()

    print(f"[info] task = {task}")
    print(f"[info] time_index = {time_index}")
    print(f"[info] write/frame = {write_number}")
    if t is not None:
        print(f"[info] sim_time = {t:.12e} s")
    print(f"[info] Ntheta = {len(theta)}")
    print(f"[info] Nphi = {nphi}")


if __name__ == "__main__":
    main()