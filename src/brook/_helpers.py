# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""Kimimaro-compatible helpers: Python adapters around the compiled engine."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

_XS = {"id": "cross_sectional_area", "data_type": "float32", "num_components": 1}
_XS_CONTACT = {"id": "cross_sectional_area_contacts", "data_type": "uint8", "num_components": 1}


def _engine():
    from ._runtime import engine

    return engine()


def _context():
    from ._runtime import context

    return context()


def _is_crackle(labels):
    # Detect Crackle lazily without importing its optional codec.
    return any(base.__module__.startswith("crackle") for base in type(labels).__mro__)


def _host_labels(labels):
    return labels[:] if _is_crackle(labels) else np.asarray(labels)


def _skeleton_list(skeletons):
    if hasattr(skeletons, "vertices"):
        return [skeletons]
    if isinstance(skeletons, dict):
        return list(skeletons.values())
    return list(skeletons)


def _property(skeleton, descriptor):
    if not any(attribute["id"] == descriptor["id"] for attribute in skeleton.extra_attributes):
        skeleton.extra_attributes.append(descriptor)


def _area_record(skeleton, *, cropped=False):
    return {
        "id": 1 if cropped else (skeleton.id if skeleton.id is not None else 0),
        "vertices": skeleton.vertices,
        "edges": skeleton.edges,
        "physical": skeleton.space == "physical",
        "areas": getattr(skeleton, "cross_sectional_area", None),
        "contacts": getattr(skeleton, "cross_sectional_area_contacts", None),
    }


def _attach_area(skeleton, result, multipass, repair_contacts, visualize):
    areas, contacts = result[:2]
    processed = result[-1]
    reuse = repair_contacts or (multipass and hasattr(skeleton, "cross_sectional_area"))
    for name, value in (("cross_sectional_area", areas), ("cross_sectional_area_contacts", contacts)):
        existing = getattr(skeleton, name, None)
        if not processed and existing is not None:
            continue
        if existing is not None and reuse:
            # Preserve Kimimaro's aliases to preexisting property arrays.
            np.copyto(existing, value, casting="unsafe")
        else:
            setattr(skeleton, name, value)
    if visualize and processed:
        import microviewer

        microviewer.view(result[2], seg=True)


def _area_aliases(skeletons, multipass, repair_contacts):
    seen = set()
    owners = set()
    for skeleton in skeletons:
        if id(skeleton) in seen:
            return True
        seen.add(id(skeleton))
        if multipass or repair_contacts:
            for name in ("cross_sectional_area", "cross_sectional_area_contacts"):
                array = getattr(skeleton, name, None)
                if array is None:
                    continue
                while isinstance(getattr(array, "base", None), np.ndarray):
                    array = array.base
                if id(array) in owners:
                    return True
                owners.add(id(array))
    return False


def cross_sectional_area(labels, skeletons, anisotropy=None, smoothing_window=1,
                         progress=False, in_place=False, fill_holes=False,
                         multipass=False, repair_contacts=False,
                         visualize_section_planes=False, step=1):
    """Kimimaro's `cross_sectional_area`: set the `cross_sectional_area` and
    `cross_sectional_area_contacts` vertex attributes. Runs on the CPU."""
    assert step > 0
    assert smoothing_window > 0
    engine = _engine()
    objects = _skeleton_list(skeletons)
    options = dict(anisotropy=anisotropy, smoothing_window=smoothing_window,
                   step=step, multipass=multipass, repair_contacts=repair_contacts,
                   visualize_section_planes=visualize_section_planes,
                   _return_processed=True)
    if _is_crackle(labels):
        # Iterate exact binary crops, without an extra halo, hole filling, or
        # renumbering.
        boxes = labels.bounding_boxes()
        iterator = labels.each(crop=True, labels=list(skeletons.keys()))
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Cross Section Analysis Paths")
        for label, image in iterator:
            skeleton = skeletons[label]
            origin = tuple(axis.start for axis in boxes[label])
            result = engine.cross_sectional_area(
                image, [_area_record(skeleton, cropped=True)], origin=origin,
                _shape_is_crop=True, **options,
            )[0]
            _attach_area(skeleton, result, multipass, repair_contacts, visualize_section_planes)
    else:
        image = np.asarray(labels)
        if _area_aliases(objects, multipass, repair_contacts):
            # Duplicate object/property aliases make sequential mutation observable.
            # Compiled mutation of labels must still happen once, after all shapes.
            for skeleton in objects:
                result = engine.cross_sectional_area(image, [_area_record(skeleton)],
                                                     fill_holes=fill_holes, **options)[0]
                _attach_area(skeleton, result, multipass, repair_contacts, visualize_section_planes)
            if in_place:
                engine.cross_sectional_area(image, [], in_place=True, **options)
        else:
            results = engine.cross_sectional_area(image, [_area_record(s) for s in objects],
                                                  fill_holes=fill_holes, in_place=in_place, **options)
            for skeleton, result in zip(objects, results):
                _attach_area(skeleton, result, multipass, repair_contacts, visualize_section_planes)
    for skeleton in objects:
        _property(skeleton, _XS)
        _property(skeleton, _XS_CONTACT)
        # Labels that were not visited get the defaults that skipped ndarray labels get.
        if not hasattr(skeleton, "cross_sectional_area"):
            skeleton.cross_sectional_area = np.full(len(skeleton.vertices), -1, dtype=np.float32)
        if not hasattr(skeleton, "cross_sectional_area_contacts"):
            skeleton.cross_sectional_area_contacts = np.zeros(len(skeleton.vertices), dtype=np.uint8)
    return skeletons


def oversegment(labels, skeletons, anisotropy=None, progress=False, fill_holes=False,
                in_place=False, downsample=0):
    """Kimimaro's `oversegment`. Runs on the CPU; `fill_holes=True` uses the GPU."""
    if fill_holes:
        return _context().oversegment(_host_labels(labels), skeletons, anisotropy=anisotropy,
                                      progress=progress, fill_holes=fill_holes,
                                      in_place=in_place, downsample=downsample)
    return _engine()._oversegment_host(_host_labels(labels), skeletons, anisotropy=anisotropy,
                                      progress=progress, fill_holes=fill_holes,
                                      in_place=in_place, downsample=downsample)


def connect_points(labels, start, end, anisotropy=(1, 1, 1), fill_holes=False, in_place=False,
                   pdrf_scale=100000, pdrf_exponent=4):
    """Trace the centerline between two points of a binary image; a drop-in for `kimimaro.connect_points`.

    Takes Kimimaro's arguments and returns one `osteoid.Skeleton` whose vertices run from `end` to
    `start` in physical coordinates (`space="physical"`). As in Kimimaro 5.8.1, `fill_holes` and
    `in_place` are accepted for compatibility and ignored: holes are not filled and `labels` is not modified.
    """
    from osteoid import Skeleton

    from . import DimensionError

    image = _host_labels(labels)
    while image.ndim > 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim > 3:
        raise DimensionError(f"{image.ndim} dimensions is not supported (max 3).")
    while image.ndim < 3:
        image = image[..., np.newaxis]
    core_engine = _context().connect_points(image, start, end, anisotropy=anisotropy,
                                       pdrf_scale=pdrf_scale, pdrf_exponent=pdrf_exponent)
    # The compiled path already multiplies the vertices by the anisotropy, so they are physical, as
    # Kimimaro marks its result. Kimimaro 5.8.1 does that scaling in place on the uint32 path, which
    # NumPy's same_kind casting rejects; these float32 values are what kimimaro.skeletonize's
    # np.multiply(..., dtype=np.float32) gives for the same path.
    return Skeleton(core_engine.vertices, core_engine.edges, radii=core_engine.radii, space="physical")


def synapses_to_targets(labels, synapses, progress=False):
    """Kimimaro's `synapses_to_targets`, computed on the CPU."""
    return _engine().synapses_to_targets(_host_labels(labels), synapses, progress=progress)


def _merged_edge_dtype(dtypes):
    # osteoid 0.7's simple_merge and consolidate keep the promoted edge dtype; Skeleton() has uint32.
    return np.result_type(*dtypes) if dtypes else np.dtype(np.uint32)


def _join_edge_dtype(result, inputs):
    # Kimimaro's join: an input with several components splits into uint32 pieces, the others
    # keep their edge dtype, and a connecting edge promotes the result with uint32.
    dtype = _merged_edge_dtype([np.uint32 if i < 0 else inputs[i].edges.dtype for i in result["pieces"]])
    return np.result_type(dtype, np.uint32) if result["joined"] else dtype


def _cpu_result(result, inputs, *, post_id=None, post=False, schema=None, edge_dtype=None):
    from osteoid import Skeleton

    owner = int(result["metadata_owner"])
    edges = result["edges"]
    if edge_dtype is not None:
        edges = edges.astype(edge_dtype, copy=False)
    if owner < 0:
        skeleton = Skeleton(result["vertices"], edges, radii=result["radii"],
                            space=result["space"], transform=result["transform"], extra_attributes=schema)
    else:
        prototype = inputs[owner]
        attributes = prototype.extra_attributes if schema is None else schema
        skeleton = Skeleton(result["vertices"], edges, radii=result["radii"],
                            segid=prototype.id, extra_attributes=attributes,
                            space=result["space"], transform=result["transform"])
        sources = result["sources"]
        allowed = set(result.get("attribute_inputs", range(len(inputs))))
        for attribute in attributes:
            name = attribute["id"]
            if name in ("radius", "radii"):
                setattr(skeleton, name, result["radii"])
                continue
            # Gather vertex attributes through the source indices returned by the compiled join.
            template = getattr(prototype, name)
            buffers = [getattr(source, name) if i in allowed and len(source.vertices) and len(source.edges)
                       else np.empty((len(source.vertices), *template.shape[1:]), dtype=template.dtype)
                       for i, source in enumerate(inputs)]
            values = buffers[0] if len(buffers) == 1 else np.concatenate(buffers, axis=0)
            setattr(skeleton, name, values[sources])
    if post:
        skeleton.id = post_id
    return skeleton


def postprocess(skeleton, dust_threshold=1500, tick_threshold=3500):
    """Kimimaro's `postprocess` for one skeleton (dust, loops, joins, ticks), computed on the CPU;
    returns a new `osteoid.Skeleton`."""
    result = _engine().cpu_postprocess(
        skeleton.vertices, skeleton.edges, skeleton.radii,
        dust_threshold=dust_threshold, tick_threshold=tick_threshold,
        transform=skeleton.transform, space=skeleton.space,
    )
    return _cpu_result(result, [skeleton], post=True, post_id=skeleton.id)


def join_close_components(skeletons, radius=None, restrict_by_radius=False):
    """Kimimaro's `join_close_components`: join skeletons into one through their nearest vertices,
    within `radius` (physical units; None: no limit). Runs on the CPU."""
    from osteoid.exceptions import SkeletonAttributeMixingError

    try:
        inputs = list(iter(skeletons))
    except TypeError:
        inputs = [skeletons]
    result = _engine().cpu_join(
        [(s.vertices, s.edges, s.radii, s.transform, s.space) for s in inputs],
        radius=radius, restrict_by_radius=restrict_by_radius,
    )
    nonempty = [s for s in inputs if len(s.vertices) and len(s.edges)]
    if nonempty:
        schema = nonempty[0].extra_attributes
        for skeleton in nonempty[1:]:
            if skeleton.extra_attributes != schema:
                raise SkeletonAttributeMixingError("The extended vertex attributes are not uniformly defined.")
    return _cpu_result(result, inputs, edge_dtype=_join_edge_dtype(result, inputs))


def _default_workers():
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, available // 2)


def postprocess_many(skeletons, dust_threshold=1500, tick_threshold=3500,
                     parallel=None, progress=False):
    """`postprocess` for a dict or list of skeletons, in `parallel` threads (default: half the
    available CPUs); returns the same kind of container, in input order."""
    is_dict = isinstance(skeletons, dict)
    items = list(skeletons.items()) if is_dict else list(enumerate(skeletons))
    workers = _default_workers() if parallel is None or int(parallel) <= 0 else int(parallel)
    workers = min(workers, len(items))
    if workers <= 1:
        done = {key: postprocess(skeleton, dust_threshold, tick_threshold) for key, skeleton in items}
    else:
        ordered = sorted(items, key=lambda item: -int(item[1].vertices.shape[0]))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(postprocess, skeleton, dust_threshold, tick_threshold): key
                       for key, skeleton in ordered}
            completed = as_completed(futures)
            if progress:
                from tqdm import tqdm

                completed = tqdm(completed, total=len(futures), desc="Postprocessing")
            done = {futures[future]: future.result() for future in completed}
    return {key: done[key] for key, _ in items} if is_dict else [done[key] for key, _ in items]
