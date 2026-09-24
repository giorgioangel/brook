# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Skeletonize a small synthetic label volume with Brook.

Needs an NVIDIA GPU of compute capability 8.0 or newer and an installed Brook package
(`python -m pip install brook-cu12`). No data file is needed:

    python examples/quickstart.py
"""

import numpy as np

import brook


def make_labels(n=96):
    """A ball with a tube leaving it (label 1) and two straight tubes (labels 2 and 3)."""
    z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    labels = np.zeros((n, n, n), dtype=np.uint32, order="F")
    labels[(x - 34) ** 2 + (y - 34) ** 2 + (z - 34) ** 2 <= 24**2] = 1
    labels[29:41, 29:41, 34:91] = 1
    labels[67:77, 10:86, 67:77] = 2
    labels[10:86, 72:82, 10:19] = 3
    return labels


def main():
    labels = make_labels()
    skeletons = brook.skeletonize(
        labels,
        teasar_params=brook.DEFAULT_TEASAR_PARAMS,
        anisotropy=(16, 16, 40),  # voxel size, e.g. in nanometres
        dust_threshold=1000,
        fix_branching=True,
        fix_borders=True,
        progress=False,
    )
    print(f"{len(skeletons)} labels")
    for label, skeleton in sorted(skeletons.items()):
        print(
            f"label {label}: {len(skeleton.vertices)} vertices, {len(skeleton.edges)} edges, "
            f"cable length {skeleton.cable_length():.0f}"
        )
    print(f"total: {sum(len(s.vertices) for s in skeletons.values())} vertices")


if __name__ == "__main__":
    main()
