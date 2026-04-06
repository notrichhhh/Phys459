"""
Usage:
  make_theta_master_csv.py <task> <files>... [--out=<csv>]

Options:
  --out=<csv>   Output master theta CSV filename [default: theta_master.csv]
"""

import h5py
import numpy as np
from docopt import docopt
from dedalus.tools import post
from dedalus.tools.parallel import Sync


def _write_theta_from_file(filename, task, out):
    with h5py.File(filename, "r") as f:
        dset = f["tasks"][task]
        theta = np.array(dset.dims[2][0][:].ravel(), dtype=float)
    np.savetxt(out, theta.reshape(1, -1), delimiter=",")
    print(f"[theta] wrote {out} (Ntheta={theta.size})")


def _visit_first_write(filename, start, count, task, out):
    # Only need one file/visit to get theta grid.
    _write_theta_from_file(filename, task, out)
    # Stop further visits by raising a controlled exception would be messy;
    # visit_writes will continue but we gate by a flag on rank 0 outside.


if __name__ == "__main__":
    args = docopt(__doc__)
    task = args["<task>"]
    files = args["<files>"]
    out = args["--out"] or "theta_master.csv"

    with Sync() as sync:
        if sync.comm.rank == 0:
            # Use the first file listed; consistent and fast.
            _write_theta_from_file(files[0], task, out)
        sync.comm.Barrier()
