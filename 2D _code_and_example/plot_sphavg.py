#!/usr/bin/env python3
"""
Plot spherical average of ydot from a phi0 CSV made by make_phi0_csv.py.

Assumes phi0_y.csv layout is:
    row 0: theta grid (Ntheta entries)
    row 1..: data rows over time, where:
        col 0: time (or iteration)  (if present)
        remaining cols: ydot(theta) values (Ntheta entries)

If the file does NOT include time as first column, we fall back to using
row index as x-axis.

Spherical average (axisymmetric):
    <f>(t) = (1/2) ∫_0^pi f(θ,t) sinθ dθ
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sph_avg_axisym(theta, f_theta):
    theta = np.asarray(theta, dtype=float)
    f_theta = np.asarray(f_theta, dtype=float)

    idx = np.argsort(theta)
    th = theta[idx]
    ff = f_theta[idx]

    return 0.5 * np.trapz(ff * np.sin(th), th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="phi0_y.csv (ydot task) produced by make_phi0_csv.py")
    ap.add_argument("--out", default="sphavg_ydot.png", help="Output figure filename")
    ap.add_argument("--fit-frac", type=float, default=0.2,
                    help="Fraction of last samples used for plateau fit (default 0.2 = last 20%%)")
    ap.add_argument("--title", default="Spherical average of ydot", help="Plot title")
    args = ap.parse_args()

    data = np.loadtxt(args.csv, delimiter=",")

    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError(f"{args.csv}: expected at least 2 rows (theta row + >=1 data row). Got shape {data.shape}")

    theta = data[0, :]
    rows = data[1:, :]

    # Heuristic: does first column look like time?
    # If rows have one extra column compared to theta, treat col0 as time.
    # Otherwise treat all cols as theta-values and use index for time axis.
    has_time = (rows.shape[1] == theta.size + 1)

    if has_time:
        t = rows[:, 0]
        ydot = rows[:, 1:]
        xlab = "t"
    else:
        t = np.arange(rows.shape[0], dtype=float)
        ydot = rows
        xlab = "index"

    if ydot.shape[1] != theta.size:
        raise ValueError(
            f"theta size = {theta.size}, but data has {ydot.shape[1]} columns of ydot. "
            f"(has_time={has_time}). Check your CSV format."
        )

    sph = np.array([sph_avg_axisym(theta, ydot[i, :]) for i in range(ydot.shape[0])])

    # Plateau fit = mean of last fit_frac samples
    n = sph.size
    n_fit = max(1, int(np.ceil(args.fit_frac * n)))
    fit_slice = slice(n - n_fit, n)
    plateau = float(np.mean(sph[fit_slice]))
    plateau_std = float(np.std(sph[fit_slice], ddof=1)) if n_fit > 1 else 0.0

    # Save fit info
    info_path = args.out.rsplit(".", 1)[0] + "_fit.txt"
    with open(info_path, "w") as f:
        f.write(f"file: {args.csv}\n")
        f.write(f"n_samples: {n}\n")
        f.write(f"fit_frac: {args.fit_frac}\n")
        f.write(f"n_fit: {n_fit}\n")
        f.write(f"plateau_mean: {plateau:.16e}\n")
        f.write(f"plateau_std:  {plateau_std:.16e}\n")

    # Plot
    plt.figure()
    plt.plot(t, sph, lw=1.5, label=r"$\langle \dot{y}\rangle$")

    # draw plateau line across last segment
    plt.axhline(plateau, ls="--", lw=1.2, label=f"plateau fit (last {args.fit_frac:.0%})")

    plt.xlabel(xlab)
    plt.ylabel(r"$\frac{1}{2}\int_0^\pi \dot{y}(\theta)\sin\theta\,d\theta$")
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Wrote: {args.out}")
    print(f"Wrote: {info_path}")
    print(f"Plateau mean = {plateau:.6e}  (std over fit window = {plateau_std:.3e}, n_fit={n_fit})")


if __name__ == "__main__":
    main()
