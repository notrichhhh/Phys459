"""
Plot 1D phi=0 data from a CSV into PNG frames and optionally a movie.

Supports two CSV formats (no header):
  OLD:
    col 0: time_or_write
    col 1..: data(theta)

  NEW (recommended):
    col 0: write_number
    col 1: sim_time [s]
    col 2..: data(theta)

Usage:
  plot_phi0_data.py <task> [--data=<csv>] [--theta=<csv>] [--out=<dir>] [--movie=<mp4>] [--fps=<int>] [--every=<int>]

Options:
  --data=<csv>    Input data CSV [default: phi0_<task>.csv]
  --theta=<csv>   Theta master CSV [default: theta_master.csv]
  --out=<dir>     Output frames directory [default: frames_1D]
  --movie=<mp4>   If provided, also write an mp4 using ffmpeg
  --fps=<int>     Movie framerate [default: 48]
  --every=<int>   Use every Nth timestep [default: 1]
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docopt import docopt
from dedalus.tools.parallel import Sync

def sph_avg_axisym(theta, f_theta):
    """
    Numerical spherical average of axisymmetric f(theta) on [0, pi]:
        <f> = (1/2) ∫_0^π f(θ) sinθ dθ
    """
    theta = np.asarray(theta, dtype=float)
    f_theta = np.asarray(f_theta, dtype=float)

    # sort just in case theta isn't strictly increasing
    idx = np.argsort(theta)
    th = theta[idx]
    ff = f_theta[idx]

    return 0.5 * np.trapz(ff * np.sin(th), th)

'''
mdot_Edd = 8.8e4
Ntheta = 3000
theta = np.linspace(0.0, np.pi, Ntheta)
# Gaussian parameters
mu = np.pi/2 # center at equator
sigma = 0.5 # width of Gaussian
gaussian_profile = np.exp(-((theta - mu)**2) / (2*sigma**2))

my_mdot = gaussian_profile * mdot_Edd
ydot_sphere_ave_const = sph_avg_axisym(theta, my_mdot)
'''
ydot_sphere_ave_const = 40000

def _load_theta(theta_csv: str) -> np.ndarray:
    theta = np.loadtxt(theta_csv, delimiter=",")
    if theta.ndim == 2:
        theta = theta[0]
    return np.array(theta, dtype=float)


def _load_csv_flexible(data_csv: str):
    raw = np.loadtxt(data_csv, delimiter=",")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if raw.shape[1] < 2:
        raise RuntimeError(f"{data_csv} has too few columns: {raw.shape[1]}")

    # NEW format if >= 3 cols and second col looks like sim_time
    if raw.shape[1] >= 3:
        write = np.array(raw[:, 0], dtype=float)
        sim_time = np.array(raw[:, 1], dtype=float)
        data = np.array(raw[:, 2:], dtype=float)
        return write, sim_time, data, True

    # OLD format: one time-like col then data
    time_or_write = np.array(raw[:, 0], dtype=float)
    data = np.array(raw[:, 1:], dtype=float)
    return time_or_write, None, data, False


def _global_minmax(data: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(data)
    if not np.any(finite):
        return -1.0, 1.0
    ymin = float(np.nanmin(data[finite]))
    ymax = float(np.nanmax(data[finite]))
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    return ymin, ymax

def main():
    args = docopt(__doc__)
    task = args["<task>"]

    data_csv  = args["--data"]  or f"phi0_{task}.csv"
    theta_csv = args["--theta"] or "theta_master.csv"
    outdir    = args["--out"]   or "frames_1D"
    movie     = args["--movie"]
    fps       = int(args["--fps"] or 48)
    every     = int(args["--every"] or 1)

    if every != 1:
        raise RuntimeError("This plotting pipeline is configured to not skip any data. Use --every=1.")

    os.makedirs(outdir, exist_ok=True)

    theta = _load_theta(theta_csv)
    x0, sim_time, data, is_new = _load_csv_flexible(data_csv)

    # Robust handling of off-by-one / mismatch:
    if data.shape[1] != theta.size:
        n = min(data.shape[1], theta.size)
        print(f"[warn] theta/data mismatch: data cols={data.shape[1]} vs Ntheta={theta.size}. Truncating to {n}.")
        theta = theta[:n]
        data = data[:, :n]

    if task == "y":
        data_pos = np.where(np.isfinite(data) & (data > 0), data, np.nan)
        ymin, ymax = _global_minmax(data_pos)
    else:
        ymin, ymax = _global_minmax(data)

    frame = 0
    for i in range(0, data.shape[0], every):
        y = data[i, :].copy()
        y[~np.isfinite(y)] = np.nan

        plt.figure()
        if task == "y":
            y_plot = y.copy()
            y_plot[y_plot <= 0] = np.nan   # log-safe
            plt.plot(theta, y_plot, label=task)
            plt.yscale("log")
        else:
            plt.plot(theta, y, label=task)

        # Add spherical-average horizontal line
        if task == 'y_dot':
            if i == 0:
                ydot_sphere_ave_const = sph_avg_axisym(theta, y)
            plt.axhline(ydot_sphere_ave_const, linestyle="--", linewidth=1.2, label='spherical average')

        plt.xlabel("theta (rad)")
        plt.ylabel(task)

        if is_new:
            w = int(round(x0[i]))
            t = float(sim_time[i])
            plt.title(f"{task}   write={w:06d}   t={t:.6e} s")
        else:
            plt.title(f"{task}   frame={int(round(x0[i])):06d}")

        plt.xlim(theta.min(), theta.max())
        plt.ylim(ymin, ymax)
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"frame_{frame:06d}.png"), dpi=150)
        plt.close()
        frame += 1

    print(f"[plot] wrote {frame} frames to {outdir}/")

    if movie:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(outdir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            movie
        ]
        subprocess.run(cmd, check=True)
        print(f"[movie] wrote {movie}")


if __name__ == "__main__":
    with Sync() as sync:
        if sync.comm.rank == 0:
            main()
        sync.comm.Barrier()
