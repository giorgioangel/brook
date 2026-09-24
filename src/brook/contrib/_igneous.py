# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Brook integration for Igneous.

Skeletonization and the helper algorithms use Brook; forge="kimimaro" uses the Kimimaro module
that Igneous imports.
"""
from __future__ import annotations

import contextlib
import threading

_state = {}
_gpu_lock = threading.RLock()            # one task on the device at a time (see `execute`)
stats = {"skeletonize_calls": 0, "skeletonize_seconds": 0.0}     # since the last install()


class _Engine:
    """Kimimaro's interface backed by Brook skeletonization and postprocessing."""

    def __init__(self, kimimaro, forge="brook"):
        self._kimimaro, self._forge = kimimaro, forge

    def skeletonize(self, *args, **kwargs):
        import time

        t = time.perf_counter()
        try:
            if self._forge == "kimimaro":
                return self._kimimaro.skeletonize(*args, **kwargs)
            from .._intake import skeletonize

            kwargs.setdefault("progress", False)
            _state.pop("listing", None)
            with _gpu_lock:
                t = time.perf_counter()
                return skeletonize(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t
            with _gpu_lock:
                stats["skeletonize_calls"] += 1
                stats["skeletonize_seconds"] += dt

    def postprocess(self, *args, **kwargs):
        from .._helpers import postprocess

        return postprocess(*args, **kwargs)

    def __getattr__(self, name):
        from .. import _helpers

        if name == "cross_sectional_area_single":
            return _cross_sectional_area_single
        if name in {"postprocess_many", "join_close_components", "synapses_to_targets",
                    "cross_sectional_area", "oversegment", "connect_points"}:
            function = getattr(_helpers, name)
            if name in ("connect_points", "oversegment"):
                def locked(*args, **kwargs):
                    with _gpu_lock:
                        return function(*args, **kwargs)
                return locked
            return function
        # No automatic fallback: Kimimaro is used only when selected with forge="kimimaro".
        raise AttributeError(name)


class _Held:
    """What `_Crackle.compress` hands back in place of a crackle stream: the array itself."""

    def __init__(self, array):
        self.array = array


class _Crackle:
    """`crackle`, as seen by `igneous.tasks.skeleton`. SkeletonTask compresses the whole cutout just
    before skeletonization and passes a callback that decompresses it, to save memory while the
    engine holds the labels. A Brook call is short, and the compression round trip can take longer
    than the skeletonization, so a 3D label array is held as is; everything else goes to crackle."""

    def __init__(self, crackle):
        self._crackle = crackle

    def compress(self, labels, *args, **kwargs):
        import numpy as np

        if isinstance(labels, np.ndarray) and labels.ndim == 3 and not args and not kwargs:
            return _Held(labels)
        return self._crackle.compress(labels, *args, **kwargs)

    def decompress(self, binary, *args, **kwargs):
        if isinstance(binary, _Held):
            return binary.array
        return self._crackle.decompress(binary, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._crackle, name)


def _fragment_names(task):
    """Return this task's fragment files from a cached skeleton-directory listing."""
    import bisect

    from cloudfiles import CloudFiles

    key = (task.cloudpath, task.vol.skeleton.path)
    with _gpu_lock:
        names = _state.setdefault("listing", {}).get(key)
        if names is None:
            names = sorted(CloudFiles(task.cloudpath, progress=False).list(prefix=task.vol.skeleton.path + "/"))
            _state["listing"][key] = names
    prefix = f"{task.vol.skeleton.path}/{task.prefix}"
    lo = bisect.bisect_left(names, prefix)
    hi = bisect.bisect_left(names, prefix + "\U0010ffff")
    return names[lo:hi]


def _cross_sectional_area_single(binimg, skel, roi=None, anisotropy=None,
                               smoothing_window=1, progress=False, in_place=False,
                               multipass=False, repair_contacts=False,
                               visualize_section_planes=False, step=1):
    from .. import _helpers as helpers
    from .._runtime import engine

    assert step > 0
    assert smoothing_window > 0
    origin = (0, 0, 0) if roi is None else tuple(roi.minpt)
    result = engine().cross_sectional_area(
        binimg, [helpers._area_record(skel, cropped=True)], anisotropy=anisotropy,
        smoothing_window=smoothing_window, step=step, multipass=multipass,
        repair_contacts=repair_contacts, visualize_section_planes=visualize_section_planes,
        origin=origin, _shape_is_crop=True, _return_processed=True,
    )[0]
    helpers._attach_area(skel, result, multipass, repair_contacts, visualize_section_planes)
    helpers._property(skel, helpers._XS)
    helpers._property(skel, helpers._XS_CONTACT)
    return skel


def _fuse(skels, *, boxes=None, resolution=None, crop=0, float32_only=False):
    from osteoid import Skeleton
    from osteoid.exceptions import SkeletonAttributeMixingError

    from .._helpers import _cpu_result, _merged_edge_dtype
    from .._runtime import engine

    inputs = list(skels)
    result = engine().cpu_fuse(
        [(s.vertices, s.edges, s.radii, s.transform, s.space) for s in inputs],
        boxes=boxes, resolution=resolution, crop=crop,
    )
    kept = result["attribute_inputs"]
    schema = inputs[kept[0]].extra_attributes if kept else Skeleton().extra_attributes
    for index in kept[1:]:
        if inputs[index].extra_attributes != schema:
            raise SkeletonAttributeMixingError("The extended vertex attributes are not uniformly defined.")
    if float32_only:
        import numpy as np

        schema = [a for a in schema if a["data_type"] == "float32"]
        if not any(a["id"] in ("radius", "radii") for a in schema):
            result["radii"] = np.full(len(result["vertices"]), -1, dtype=np.float32)
    else:
        schema = None
    return _cpu_result(result, inputs, schema=schema,
                       edge_dtype=_merged_edge_dtype([inputs[i].edges.dtype for i in kept]))


def _within_length(skeleton, maximum, invert_exceeds=False):
    from .._runtime import engine

    if maximum is None:
        return True
    result = engine().cable_length_info(skeleton.vertices, skeleton.edges, skeleton.radii,
                                       maximum, skeleton.transform, skeleton.space)
    # osteoid's cable_length calls physical_space(copy=False), which converts the skeleton to
    # physical space in place; do the same. The conversion itself is done in C++.
    if skeleton.space != "physical":
        skeleton.vertices = result["vertices"]
        skeleton.space = result["space"]
    return not result["exceeds"] if invert_exceeds else result["within"]


def _postprocess(skeletons, dust_threshold, tick_threshold, max_cable_length, device, cpu_sharded=False):
    from .._helpers import postprocess

    todo = {label: skeleton for label, skeleton in skeletons.items()
            if _within_length(skeleton, max_cable_length, cpu_sharded)}
    if not todo:
        return dict(skeletons)
    if device:
        from .._device_post import postprocess as canonical_postprocess
        from .._packed import PackedSkeletons

        with _gpu_lock:
            done = canonical_postprocess(PackedSkeletons.from_skeletons(todo, (1, 1, 1)),
                                          dust_threshold, tick_threshold).to_skeletons()
        for label, skeleton in done.items():
            skeleton.transform, skeleton.space = skeletons[label].transform, skeletons[label].space
    else:
        done = {label: postprocess(skeleton, dust_threshold, tick_threshold)
                for label, skeleton in todo.items()}
    return {label: done.get(label, skeleton) for label, skeleton in skeletons.items()}


def _device_postprocess(skeletons, dust_threshold, tick_threshold, max_cable_length):
    return _postprocess(skeletons, dust_threshold, tick_threshold, max_cable_length, True)


def install(postprocess="kimimaro", forge="brook", hold_labels=True):
    """Make Igneous skeleton tasks use Brook; Igneous keeps storage and task I/O.

    `postprocess="kimimaro"` applies Kimimaro's postprocessing rules on the CPU;
    `postprocess="device"` applies Brook's GPU postprocessing, with Brook's tie rules.
    `forge="kimimaro"` runs Igneous's own Kimimaro for skeletonization; it is used only when
    selected explicitly.
    """
    if postprocess not in ("kimimaro", "device"):
        raise ValueError("postprocess must be 'kimimaro' or 'device'")
    if forge not in ("brook", "kimimaro"):
        raise ValueError("forge must be 'brook' or 'kimimaro'")
    import igneous.tasks.skeleton as its

    stats.update(skeletonize_calls=0, skeletonize_seconds=0.0)
    if "kimimaro" not in _state:
        _state["kimimaro"] = its.kimimaro
        _state["crackle"] = its.crackle
        _state["sharded"] = its.ShardedSkeletonMergeTask.process_skeletons
        _state["unsharded"] = its.UnshardedSkeletonMergeTask.execute
        _state["fuse"] = its.UnshardedSkeletonMergeTask.fuse_skeletons
    its.kimimaro = _Engine(_state["kimimaro"], forge)
    its.crackle = _Crackle(_state["crackle"]) if forge == "brook" and hold_labels else _state["crackle"]
    device = postprocess == "device"

    def fuse_skeletons(self, fragments):
        fragments = list(fragments)
        boxes = [[box.minpt, box.maxpt] for box, _ in fragments] if self.crop > 0 else None
        return _fuse([s for _, s in fragments], boxes=boxes,
                     resolution=self.vol.resolution if boxes is not None else None, crop=self.crop)

    def process_skeletons(self, unfused_skeletons, in_place=False):
        fused = {}
        for label, fragments in unfused_skeletons.items():
            fused[label] = _fuse(fragments, float32_only=True)
            fused[label].id = label
        done = _postprocess(fused, self.dust_threshold, self.tick_threshold, self.max_cable_length, device,
                            cpu_sharded=not device)
        out = unfused_skeletons if in_place else {}
        for label, skeleton in done.items():
            skeleton.id = label
            out[label] = skeleton.to_precomputed()
        return out

    def execute_merge(self):
        from cloudfiles import CloudFiles
        from cloudvolume import CloudVolume

        if device:
            self.vol = CloudVolume(self.cloudpath, cdn_cache=False, cache_locking=False)
        else:
            self.vol = CloudVolume(self.cloudpath, cdn_cache=False)
        self.vol.mip = self.vol.skeleton.meta.mip
        names = _fragment_names(self) if device else self.get_filenames()
        fused = {label: self.fuse_skeletons(fragments)
                 for label, fragments in self.get_skeletons_by_segid(names).items()}
        done = _postprocess(fused, self.dust_threshold, self.tick_threshold, self.max_cable_length, device)
        outputs = []
        for label, skeleton in done.items():
            skeleton.id = label
            outputs.append(skeleton)
        self.vol.skeleton.upload(outputs)
        if self.delete_fragments:
            CloudFiles(self.cloudpath, progress=True).delete(names)

    its.UnshardedSkeletonMergeTask.fuse_skeletons = fuse_skeletons
    its.ShardedSkeletonMergeTask.process_skeletons = process_skeletons
    its.UnshardedSkeletonMergeTask.execute = execute_merge


def uninstall():
    """Restore the Igneous functions that `install` replaced."""
    if "kimimaro" not in _state:
        return
    import igneous.tasks.skeleton as its

    _state.pop("listing", None)
    its.kimimaro = _state.pop("kimimaro")
    its.crackle = _state.pop("crackle")
    its.ShardedSkeletonMergeTask.process_skeletons = _state.pop("sharded")
    its.UnshardedSkeletonMergeTask.execute = _state.pop("unsharded")
    its.UnshardedSkeletonMergeTask.fuse_skeletons = _state.pop("fuse")


def execute(tasks, concurrency=3, progress=False):
    """Run Igneous tasks concurrently so I/O overlaps serialized GPU work.

    Call inside :func:`engine` or after :func:`install`. Each in-flight task
    retains its cutout in host memory. Returns the number of tasks run; the
    first exception is raised after running tasks finish.
    """
    from concurrent.futures import ThreadPoolExecutor

    tasks = list(tasks)
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=len(tasks), desc="Tasks")
    def run(task):
        task.execute()
        if progress:
            bar.update(1)
    try:
        with ThreadPoolExecutor(max_workers=max(int(concurrency), 1)) as pool:
            for future in [pool.submit(run, task) for task in tasks]:
                future.result()
    finally:
        if progress:
            bar.close()
    return len(tasks)


@contextlib.contextmanager
def engine(postprocess="kimimaro", forge="brook", hold_labels=True):
    """Context manager: `install` on entry, `uninstall` on exit."""
    install(postprocess, forge, hold_labels)
    try:
        yield
    finally:
        uninstall()
