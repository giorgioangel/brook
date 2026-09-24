"""Time complete skeletonization calls on a NumPy label volume and print a JSON record."""
# The published 512^3 numbers use benchmarks/connectomics.npy.ckl.gz from Kimimaro
# (https://github.com/seung-lab/kimimaro, revision fee3b020177f0a1499238358ec4785a759f9d8d0),
# saved in Fortran order with crackle-codec and NumPy:
#
#   labels = crackle.load("benchmarks/connectomics.npy.ckl.gz")
#   np.save("full512.npy", np.asfortranarray(labels))
#   np.save("crop256.npy", np.asfortranarray(labels[:256, :256, :256]))
#
# The SHA-256 of the voxel bytes, recorded as input_sha256, is
# 4f5ad1c03f6fa0478a5c332a1ff51cf7636a83a59c87e1d086e07fb0a5fc77d6 for full512.npy and
# 60a1fba0158105fbd137d8fc470dfd2d1469b05f16a34eb6298e6b0bd64b8b7a for crop256.npy.
# Brook and Kimimaro 5.8.1 were installed in separate environments, both with NumPy 2.4.6:
#
#   python examples/benchmark.py full512.npy --backend brook
#   taskset -c 0 python examples/benchmark.py full512.npy --backend kimimaro --parallel 1
#   python examples/benchmark.py full512.npy --backend kimimaro --parallel 32
#
# and again with --no-fix-branching.
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume", type=Path)
    parser.add_argument("--backend", choices=("brook", "kimimaro"), default="brook")
    parser.add_argument("--parallel", type=int, default=1, help="Kimimaro worker count")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--fix-branching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-skeletons", type=Path, help="Export the final output outside timing")
    parser.add_argument("--memory-profile", type=Path, help="Sample process memory during the first warmup only")
    args = parser.parse_args()
    if args.memory_profile and args.warmups < 1:
        parser.error("memory profiling requires at least one warmup")
    if args.runs < 1 or args.warmups < 0:
        parser.error("runs must be positive and warmups nonnegative")
    labels = np.load(args.volume, allow_pickle=False)
    if labels.ndim != 3:
        parser.error("volume must be three-dimensional")
    backend = importlib.import_module(args.backend)
    options = dict(
        teasar_params=dict(scale=1.5, const=300, pdrf_exponent=4, pdrf_scale=100000,
                          soma_detection_threshold=1100, soma_acceptance_threshold=3500,
                          soma_invalidation_scale=1.0, soma_invalidation_const=300),
        anisotropy=(16, 16, 40), dust_threshold=1000,
        fix_borders=True, fix_branching=args.fix_branching, progress=False,
    )
    if args.backend == "brook":
        options["streaming"] = False
    else:
        options["parallel"] = args.parallel
    seconds, summaries = [], []
    for run in range(args.warmups + args.runs):
        monitor = None
        if run == 0 and args.memory_profile:
            monitor = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("profile_memory.py")),
                 "--pid", str(os.getpid()), "--output", str(args.memory_profile),
                 "--gpu" if args.backend == "brook" else "--no-gpu"],
                stdout=subprocess.PIPE, text=True,
            )
            if monitor.stdout.readline().strip() != "ready":
                raise RuntimeError("memory sampler failed to start")
        start = time.perf_counter()
        completed = False
        try:
            skeletons = backend.skeletonize(labels, **options)
            completed = True
        finally:
            elapsed = time.perf_counter() - start
            if monitor is not None:
                monitor.terminate()
                if monitor.wait() != 0:
                    raise RuntimeError("memory sampler failed")
                args.memory_profile.with_suffix(".window.json").write_text(json.dumps(
                    dict(start=start, end=start + elapsed, backend=args.backend, shape=labels.shape,
                         options=options, sampled_phase="warmup", completed=completed), indent=2) + "\n")
        print(f"{args.backend} {'warmup' if run < args.warmups else 'timed'} "
              f"{run + 1}: {elapsed:.3f} s", file=sys.stderr, flush=True)
        if run >= args.warmups:
            seconds.append(elapsed)
            summaries.append(dict(
                labels=sorted(int(k) for k in skeletons),
                vertices=sum(len(s.vertices) for s in skeletons.values()),
                edges=sum(len(s.edges) for s in skeletons.values()),
                cable_length=sum(float(s.cable_length()) for s in skeletons.values()),
            ))
        if args.save_skeletons and run == args.warmups + args.runs - 1:
            ordered = sorted(skeletons.items())
            offsets = np.cumsum([0] + [len(s.vertices) for _, s in ordered])
            np.savez_compressed(
                args.save_skeletons,
                labels=np.array([k for k, _ in ordered], dtype=np.uint64),
                offsets=offsets,
                vertices=np.concatenate([s.vertices for _, s in ordered]) if ordered else np.empty((0, 3)),
                edges=np.concatenate([s.edges.astype(np.int64) + offsets[i]
                                      for i, (_, s) in enumerate(ordered)]) if ordered else np.empty((0, 2), np.int64),
                edge_labels=np.concatenate([np.full(len(s.edges), k, np.uint64)
                                            for k, s in ordered]) if ordered else np.empty(0, np.uint64),
            )
        del skeletons
    print(json.dumps(dict(
        backend=args.backend, version=backend.__version__ if args.backend == "brook" else importlib.metadata.version(args.backend),
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        warmup_memory_profiled=bool(args.memory_profile),
        shape=labels.shape, dtype=str(labels.dtype), fortran_order=bool(labels.flags.f_contiguous),
        input_sha256=hashlib.sha256(labels.tobytes(order="F")).hexdigest(),
        options=options, warmups=args.warmups, seconds=seconds,
        median_seconds=statistics.median(seconds), outputs=summaries,
    ), indent=2))


if __name__ == "__main__":
    main()
