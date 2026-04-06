#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys

# Usage:
#   python plot_max_vs_time.py phi0_u_theta.csv
#   python plot_max_vs_time.py phi0_y_dot.csv

if len(sys.argv) != 2:
    print("Usage: plot_max_vs_time.py <phi0_*.csv>")
    sys.exit(1)

csv = sys.argv[1]

# Load CSV
raw = np.loadtxt(csv, delimiter=",")
if raw.ndim == 1:
    raw = raw.reshape(1, -1)

# Detect format
# NEW format: write, sim_time, data...
# OLD format: time_or_write, data...
if raw.shape[1] >= 3:
    t = raw[:, 1]          # sim_time
    data = raw[:, 2:]      # theta values
else:
    t = raw[:, 0]
    data = raw[:, 1:]

# Compute max over theta at each time
maxval = np.nanmax(data, axis=1)

# Plot
plt.figure()
plt.plot(t, maxval)
plt.xlabel("time (s)")
plt.ylabel(r"max $u_\theta$")
plt.title(r"max $u_\theta$ vs time")
plt.grid(True)
plt.tight_layout()

out = csv.replace(".csv", "_max_vs_time.png")
plt.savefig(out, dpi=200)
plt.show()

print(f"[ok] wrote {out}")
