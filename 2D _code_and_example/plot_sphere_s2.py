"""
Plot sphere outputs with automatic global min/max detection
for an arbitrary Dedalus task.

Usage:
    plot_sphere_s2.py <task> <files>... [--output=<dir>] [--linear]

Options:
    <task>          Name of the task in the HDF5 "tasks" group (e.g. T, y, flux, u, u_theta, u_phi, y_dot).
    --output=<dir>  Output directory [default: ./frames]
    --linear        Use linear color scale instead of log (default is log scale)
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import ScalarFormatter, LogFormatterSciNotation


# ============================================================
# USER-CONTROLLED STYLE SETTINGS
# ============================================================
FIGSIZE = (16, 16)
DPI = 150

TIME_FONT_SIZE = 40
CBAR_LABEL_FONT_SIZE = 34
CBAR_TICK_FONT_SIZE = 32

TIME_TEXT_X = 0.02
TIME_TEXT_Y = 0.98

CBAR_LEFT = 0.82
CBAR_BOTTOM = 0.15
CBAR_WIDTH = 0.03
CBAR_HEIGHT = 0.70
# ============================================================


def build_s2_coord_vertices(phi, theta):
    """
    Build vertex arrays for plotting data on a sphere with pcolormesh-style
    faces, using the cell-centered phi/theta coordinates from Dedalus.
    """
    phi = phi.ravel()
    phi_vert = np.concatenate([phi, [2*np.pi]])
    phi_vert -= phi_vert[1] / 2

    theta = theta.ravel()
    theta_mid = (theta[:-1] + theta[1:]) / 2
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])

    return np.meshgrid(phi_vert, theta_vert, indexing='ij')


def scan_files_for_bounds(files, task, use_log=True):
    """
    Scan all given HDF5 files and return global min/max for the given task.

    Logic adapted from your original script, but generalized:
    - If use_log=True: only positive values are used for the bounds.
    - If use_log=False: all finite values (including negative) are used.
    """
    gmin, gmax = np.inf, -np.inf

    for fn in files:
        with h5py.File(fn, "r") as f:
            if "tasks" not in f or task not in f["tasks"]:
                print(f"[warn] {fn}: missing tasks/{task}; skipping")
                continue

            dset = f["tasks"][task]   # shape: (writes, phi, theta, ...)
            n = dset.shape[0]

            for i in range(n):
                arr = np.array(dset[(i, slice(None), slice(None))], dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue

                if use_log:
                    pos = arr[arr > 0.0]
                    if pos.size:
                        gmin = min(gmin, pos.min())
                        gmax = max(gmax, pos.max())
                else:
                    gmin = min(gmin, arr.min())
                    gmax = max(gmax, arr.max())

    if not np.isfinite(gmin) or not np.isfinite(gmax):
        raise RuntimeError(f"No finite data found for task '{task}' when scanning files.")

    # For LogNorm, ensure vmin > 0
    if use_log and gmin <= 0:
        gmin = np.nextafter(0.0, 1.0)

    print("=== GLOBAL BOUNDS (auto-detected) ===")
    print(f"TASK     : {task}")
    print(f"LOG-SCALE: {use_log}")
    print(f"VMIN     = {gmin:.8e}")
    print(f"VMAX     = {gmax:.8e}")

    return gmin, gmax


def main(filename, start, count, output, vmin, vmax, task, use_log):
    """Save plot of specified task for given range of analysis writes."""
    cmap = plt.cm.viridis
    savename_func = lambda write: f"write_{write:06d}.png"

    # Create figure
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')

    # Time overlay
    frame_label = fig.text(
        TIME_TEXT_X, TIME_TEXT_Y, "",
        ha='left', va='top',
        fontsize=TIME_FONT_SIZE
    )

    # Proper normalization for BOTH log and linear
    if use_log:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        dset = file['tasks'][task]
        phi = dset.dims[1][0][:].ravel()
        theta = dset.dims[2][0][:].ravel()
        phi_vert, theta_vert = build_s2_coord_vertices(phi, theta)
        x = np.sin(theta_vert) * np.cos(phi_vert)
        y = np.sin(theta_vert) * np.sin(phi_vert)
        z = np.cos(theta_vert)

        for index in range(start, start + count):
            data_slices = (index, slice(None), slice(None))
            data = np.array(dset[data_slices], dtype=float)
            data = np.where(np.isfinite(data), data, np.nan)

            if use_log:
                # sanitize for LogNorm: only positive values
                data_plot = np.where((data > 0) & np.isfinite(data), data, np.nan)

                if np.all(np.isnan(data_plot)):
                    tmp = np.full_like(data, vmin, dtype=float)
                else:
                    tmp = np.where(np.isfinite(data_plot), data_plot, vmin)

                fc = cmap(norm(tmp))

            else:
                data_plot = np.where(np.isfinite(data), data, np.nan)

                if np.all(np.isnan(data_plot)):
                    tmp = np.full_like(data, vmin, dtype=float)
                else:
                    tmp = np.clip(data_plot, vmin, vmax)

                fc = cmap(norm(tmp))

            if index == start:
                surf = ax.plot_surface(
                    x, y, z,
                    facecolors=fc,
                    cstride=1, rstride=1,
                    linewidth=0,
                    antialiased=False,
                    shade=False,
                    zorder=5
                )
                ax.set_box_aspect((1, 1, 1))
                ax.set_xlim(-0.7, 0.7)
                ax.set_ylim(-0.7, 0.7)
                ax.set_zlim(-0.7, 0.7)
                ax.axis('off')

                # Colorbar only on first frame
                from matplotlib.cm import ScalarMappable
                mappable = ScalarMappable(norm=norm, cmap=cmap)
                mappable.set_array([])

                cax = fig.add_axes([CBAR_LEFT, CBAR_BOTTOM, CBAR_WIDTH, CBAR_HEIGHT])
                cb = fig.colorbar(mappable, cax=cax)

                # Use real-value tick labels
                if use_log:
                    cb.formatter = LogFormatterSciNotation()
                else:
                    formatter = ScalarFormatter(useMathText=True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((-2, 2))
                    cb.formatter = formatter

                cb.update_ticks()
                cb.ax.tick_params(labelsize=CBAR_TICK_FONT_SIZE)

                label = task + (' [log]' if use_log else '')
                cb.set_label(label, rotation=90, va='center', fontsize=CBAR_LABEL_FONT_SIZE)

            else:
                surf.set_facecolors(fc.reshape(fc.size // 4, 4))

            # Save figure
            write_number = int(file['scales/write_number'][index]) if 'scales/write_number' in file else int(index)

            if 'scales/sim_time' in file:
                t = float(file['scales/sim_time'][index])
                frame_label.set_text(f"t = {t:.6e} s")
            else:
                frame_label.set_text(f"write = {write_number:06d}")

            savename = savename_func(write_number)
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=DPI)

    plt.close(fig)


if __name__ == "__main__":
    import pathlib
    from docopt import docopt
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    task = args['<task>']
    file_list = args['<files>']
    use_log = not args['--linear']

    # 1) Automatically scan files to find global bounds for this task
    VMIN, VMAX = scan_files_for_bounds(file_list, task=task, use_log=use_log)

    # 2) Set up output path
    output_path = pathlib.Path(args['--output']).absolute()
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()

    # 3) Visit writes, passing vmin/vmax + task into main
    post.visit_writes(
        file_list,
        main,
        output=output_path,
        vmin=VMIN,
        vmax=VMAX,
        task=task,
        use_log=use_log,
    )