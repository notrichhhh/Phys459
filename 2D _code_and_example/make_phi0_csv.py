"""
make_phi0_csv.py

Build a phi=0 (or phi index) time-evolving 1D profile vs theta for a given Dedalus task.

Output CSV format (no header):
    column 0 : write_number   (int-like; used to sync to 2D write_*.png)
    column 1 : sim_time [s]   (neutron-star simulation time)
    columns 2.. : data at different theta

MPI-safe:
    - All ranks participate in post.visit_writes()
    - Each rank writes a temporary part file
    - Rank 0 merges and sorts by write_number

Usage:
    make_phi0_csv.py <task> <files>...
        [--phi=<i>] [--theta-master=<csv>] [--out=<csv>]

Output:
    phi0_<task>.csv
"""

import os
import h5py
import numpy as np
from docopt import docopt
from dedalus.tools import post
from dedalus.tools.parallel import Sync


def _get_write_and_time(f, start, count):
    if "scales/write_number" not in f:
        raise RuntimeError("Missing scales/write_number in snapshot file.")
    write = np.array(f["scales/write_number"][start:start + count], dtype=float)

    if "scales/sim_time" in f:
        sim_time = np.array(f["scales/sim_time"][start:start + count], dtype=float)
    else:
        # Fallback: monotonic index
        sim_time = np.arange(start, start + count, dtype=float)

    return write, sim_time


def _append_csv(path, arr):
    with open(path, "ab") as fp:
        np.savetxt(fp, arr, delimiter=",")


def main(filename, start, count, task, phi_index, theta_master, out_part):
    # Load theta_master only to set expected size; we still take data from snapshots.
    theta = np.loadtxt(theta_master, delimiter=",")
    if theta.ndim == 2:
        theta = theta[0]
    ntheta = int(theta.size)

    with h5py.File(filename, "r") as f:
        dset = f["tasks"][task]

        # Actual theta dimension from dataset (more reliable)
        ntheta_dset = int(dset.shape[2])
        ntheta_use = min(ntheta, ntheta_dset)

        # cols: write_number, sim_time, theta-data...
        block = np.empty((count, 2 + ntheta_use), dtype=float)

        for j, index in enumerate(range(start, start + count)):
            data = np.array(dset[(index, slice(None), slice(None))], dtype=float)
            # use only first ntheta_use points if there's an off-by-one mismatch
            block[j, 2:] = data[phi_index, :ntheta_use]

        write, sim_time = _get_write_and_time(f, start, count)
        block[:, 0] = write
        block[:, 1] = sim_time

    _append_csv(out_part, block)


if __name__ == "__main__":
    args = docopt(__doc__)
    task = args["<task>"]
    files = args["<files>"]
    phi_index = int(args["--phi"] or 0)
    theta_master = args["--theta-master"] or "theta_master.csv"
    out = args["--out"] or f"phi0_{task}.csv"

    with Sync() as sync:
        rank = sync.comm.rank
        size = sync.comm.size

        out_part = f"{out}.rank{rank:05d}.part"
        if os.path.exists(out_part):
            os.remove(out_part)

        sync.comm.Barrier()

        # IMPORTANT: all ranks participate (fixes the half-frames issue)
        post.visit_writes(
            files,
            main,
            task=task,
            phi_index=phi_index,
            theta_master=theta_master,
            out_part=out_part,
        )

        sync.comm.Barrier()

        if rank == 0:
            parts = [f"{out}.rank{r:05d}.part" for r in range(size)]
            rows = []

            for p in parts:
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    arr = np.loadtxt(p, delimiter=",")
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    rows.append(arr)

            if not rows:
                raise RuntimeError("No data written to any part files. Check task name and inputs.")

            merged = np.vstack(rows)

            # Sort by write_number (col 0) to restore correct order
            merged = merged[np.argsort(merged[:, 0])]

            with open(out, "wb") as fp:
                np.savetxt(fp, merged, delimiter=",")

            for p in parts:
                if os.path.exists(p):
                    os.remove(p)

        sync.comm.Barrier()

    if Sync().comm.rank == 0:
        print(f"[phi0] wrote {out}")
