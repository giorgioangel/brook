"""Exercise the installed distribution."""
import ctypes
import hashlib
import importlib.metadata
import inspect
import itertools
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from osteoid import Skeleton

import brook


def _distribution():
    """The installed distribution that provides the brook package: brook, or a name per CUDA major
    version (brook-cu12, brook-cu13) where the plain name is not available."""
    names = importlib.metadata.packages_distributions().get("brook", [])
    assert len(names) == 1, names
    return names[0]


def test_installed_layout_version_defaults():
    root = Path(brook.__file__).resolve().parent
    assert list(root.glob("_core*.so")), root
    assert (root / "_core.pyi").is_file() and (root / "py.typed").is_file()
    assert (root / ".libs/libbrook.so.0").is_file()
    assert (root / ".libs/libbrook_xs3d.so.1").is_file()
    assert (root / "_third_party/xs3d/CMakeLists.txt").is_file()
    assert (root / "_source/cpp/src/skeletonize.cu").is_file()
    assert (root / "licenses/xs3d/COPYING.LESSER").is_file()
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py", "_device_post.py", "_helpers.py", "_intake.py", "_packed.py", "_pool.py", "_runtime.py",
        "batch.py", "device.py", "device_post.py", "intake.py", "packed.py", "pool.py", "post.py", "preamble.py",
        "profile.py", "streaming.py", "trace.py"}
    library = ctypes.CDLL(str(root / ".libs/libbrook.so.0"))
    library.brook_version.restype = ctypes.c_char_p
    # One version for the Python package, the distribution metadata and the C library.
    assert brook.__version__ == importlib.metadata.version(_distribution()) == library.brook_version().decode()
    assert inspect.signature(brook.skeletonize).parameters["teasar_params"].default is brook.DEFAULT_TEASAR_PARAMS
    from brook.trace import trace_defaults

    defaults = trace_defaults()
    assert defaults["scale"] == 10 and defaults["pdrf_scale"] == 5000
    assert defaults["manual_targets_before"] is None and defaults["void_check"] is None
    assert brook.DEFAULT_TEASAR_PARAMS["scale"] == 1.5
    requirements = importlib.metadata.requires(_distribution()) or []
    base = [value for value in requirements if "extra ==" not in value]
    assert not any(name in value.lower() for value in base for name in ["cupy", "kimimaro", "scipy", "dijkstra3d"])


def test_bundled_cuda_runtime():
    """A wheel that bundles the CUDA runtime loads its own copy. Builds with
    BROOK_BUNDLE_CUDA_RUNTIME=OFF (conda) use the environment's runtime instead."""
    libs = (Path(brook.__file__).resolve().parent / ".libs").resolve()
    bundled = sorted(libs.glob("libcudart.so.*"))
    if os.environ.get("BROOK_TEST_EXPECT_BUNDLED_CUDART") == "1":
        assert bundled, f"no libcudart.so.* in {libs}"
    elif not bundled:
        pytest.skip("this build does not bundle the CUDA runtime")
    if not Path("/proc/self/maps").is_file():
        pytest.skip("needs /proc/self/maps")
    # A fresh interpreter, so that no runtime loaded earlier in this process (for example by torch)
    # can stand in for the bundled one, and without LD_LIBRARY_PATH, which the loader searches
    # before _core's RUNPATH.
    code = ("import pathlib, brook._core\n"
            "for line in pathlib.Path('/proc/self/maps').read_text().splitlines():\n"
            "    if '/libcudart.so' in line: print(line.split(maxsplit=5)[-1])")
    env = {key: value for key, value in os.environ.items() if key not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    mapped = set(subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True,
                                text=True, env=env).stdout.splitlines())
    assert mapped and all(Path(path).resolve().parent == libs for path in mapped), mapped


def test_cpu_helpers():
    skel = Skeleton(np.array([[0, 0, 0], [1, 0, 0], [5, 0, 0], [11, 0, 0]], np.float32),
                    np.array([[0, 1], [2, 3]], np.uint32), radii=np.full(4, 100, np.float32), segid=1337)
    joined = brook.join_close_components(skel)
    expected = np.array([[0, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(joined.edges, expected)
    processed = brook.postprocess(skel, dust_threshold=0, tick_threshold=0)
    np.testing.assert_array_equal(processed.edges, expected.astype(np.uint32))
    assert processed.id == 1337
    assert list(brook.postprocess_many({1337: skel}, dust_threshold=0, tick_threshold=0, parallel=1)) == [1337]
    labels = np.array([0, 7, 0, 7], np.uint64).reshape((2, 2, 1), order="F")
    targets = brook.synapses_to_targets(labels, {7: [((1, .5, 0), 3)]})
    assert targets == {(1, 0, 0): 3}
    assert all(isinstance(value, np.integer) for point in targets for value in point)


def test_cpu_join_matches_osteoid_components():
    # Expected values from Kimimaro 5.8.4 with osteoid 0.7.3. Components are taken in the order
    # of their first edge, so the duplicate at (1, 0, 0) keeps the radius of vertex 4.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0], [1, 0, 0]], np.float32)
    skel = Skeleton(vertices, np.array([[2, 3], [3, 4], [0, 1]], np.uint32),
                    radii=np.array([1, 1, 2, 2, 5], np.float32))
    joined = brook.join_close_components(skel)
    np.testing.assert_array_equal(joined.vertices, vertices[:4])
    np.testing.assert_array_equal(joined.edges, [[0, 1], [1, 3], [2, 3]])
    np.testing.assert_array_equal(joined.radii, [1, 5, 2, 2])

    # Edge dtypes: an input with several components gives uint32 pieces, others keep their
    # dtype; merged pieces promote, and a connecting edge promotes with uint32.
    square = np.array([[0, 0, 0], [1, 0, 0], [5, 5, 5], [6, 5, 5]], np.float32)
    expected = {np.int64: ["uint32", "int64", "int64", "int64", "int64"],
                np.int32: ["uint32", "int64", "int32", "int64", "int32"],
                np.uint16: ["uint32", "uint32", "uint16", "uint32", "uint16"],
                np.uint32: ["uint32"] * 5}
    for dtype, dtypes in expected.items():
        multi = Skeleton(square, np.array([[0, 1], [2, 3]], dtype))
        single = Skeleton(square[:2] + 100, np.array([[0, 1]], dtype))
        near = Skeleton(square[:2] + 3, np.array([[0, 1]], dtype))
        cases = [([multi], None), ([multi, single], None), ([single], None), ([single, near], None),
                 ([single, near], 1.0)]
        got = [str(brook.join_close_components(group, radius=radius).edges.dtype) for group, radius in cases]
        assert got == dtypes, dtype
    assert brook.join_close_components([Skeleton()]).edges.dtype == np.uint32


def test_cpu_igneous_fuse_matches_osteoid():
    from brook.contrib import _igneous

    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 0]], np.float32)
    for first, second in ((np.uint32, np.uint32), (np.int64, np.uint32), (np.int32, np.uint32)):
        skels = [Skeleton(vertices, np.array([[0, 1], [1, 2], [2, 3]], first),
                          radii=np.arange(4, dtype=np.float32)),
                 Skeleton(vertices + 1, np.array([[0, 1], [0, 0]], second),
                          radii=np.arange(4, 8, dtype=np.float32)),
                 Skeleton()]
        expected = Skeleton.simple_merge([s.clone() for s in skels]).consolidate()
        fused = _igneous._fuse([s.clone() for s in skels])
        assert fused.edges.dtype == expected.edges.dtype, first
        np.testing.assert_array_equal(fused.vertices, expected.vertices)
        np.testing.assert_array_equal(fused.edges, expected.edges)
        np.testing.assert_array_equal(fused.radii, expected.radii)
    assert _igneous._fuse([Skeleton()]).edges.dtype == np.uint32


def test_cpu_area_frozen_fixture():
    labels = np.full((13, 15, 17), 7, np.uint8, order="F")
    vertices = np.column_stack([np.arange(13), np.full(13, 7), np.full(13, 8)]).astype(np.float32)
    edges = np.column_stack([np.arange(12), np.arange(1, 13)]).astype(np.uint32)
    skel = Skeleton(vertices, edges, segid=7, space="physical")
    assert brook.cross_sectional_area(labels, skel) is skel
    np.testing.assert_array_equal(skel.cross_sectional_area, np.full(13, 255, np.float32))
    contacts = np.full(13, 60, np.uint8)
    contacts[0], contacts[-1] = 61, 62
    np.testing.assert_array_equal(skel.cross_sectional_area_contacts, contacts)


def test_cpu_packed_scheduling_and_explicit_external_forge():
    from brook.contrib import igneous
    from brook.packed import PackedSkeletons

    packed = PackedSkeletons.from_skeletons({}, (1, 1, 1))
    assert not packed.on_device and not packed.to_skeletons()
    assert pickle.loads(pickle.dumps(packed)).labels.dtype == np.int64
    assert list(brook.iter_chunks((5, 5, 5), (5, 5, 5))) == [(slice(0, 5), slice(0, 5), slice(0, 5))]
    assert brook.chunk_shape_for((100, 300, 5000), 4, 24 << 30)[:2] == (100, 300)

    class Owner:
        calls = 0
        def skeletonize(self, *args, **kwargs):
            self.calls += 1
            return {"owner": True}

    owner = Owner()
    selected = igneous._Engine(owner, forge="kimimaro")
    assert selected.skeletonize(None) == {"owner": True} and owner.calls == 1
    with pytest.raises(AttributeError):
        _ = selected.unsupported_helper
    assert owner.calls == 1


def test_cpu_oversegment_without_context(monkeypatch):
    from brook import _runtime

    def forbidden_context(*args, **kwargs):
        raise AssertionError("host oversegment must not initialize CUDA")

    monkeypatch.setattr(_runtime, "context", forbidden_context)
    labels = np.ones((5, 1, 1), np.uint8)
    skel = Skeleton(np.array([[0, 0, 0], [4, 0, 0]], np.float64),
                    np.array([[0, 1]], np.uint32), segid=1)
    skel.custom = {"data": np.array([2, 3])}
    features, copied = brook.oversegment(labels, skel)
    np.testing.assert_array_equal(features[:, 0, 0], np.array([1, 1, 2, 2, 2], np.uint8))
    np.testing.assert_array_equal(copied.segments, np.array([1, 2], np.uint8))
    assert features.dtype == copied.segments.dtype == np.uint8
    assert copied.vertices.dtype == np.float64 and copied.id == skel.id
    assert copied is not skel and not hasattr(skel, "segments")
    assert not np.shares_memory(copied.custom["data"], skel.custom["data"])


@pytest.mark.gpu
def test_gpu_public_core_helpers_and_protocols(gpu):
    from brook import device_post
    from brook.device import context
    from brook.packed import merge_fragments
    from brook.trace import trace

    labels = np.ones((16, 5, 5), np.uint16, order="C")
    params = dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4)
    options = dict(teasar_params=params, dust_threshold=0, progress=False, fix_borders=False)
    reference = brook.skeletonize(labels, streaming=False, **options)
    streamed = brook.skeletonize(labels, streaming=True, **options)
    assert list(reference) == list(streamed) == [1]
    for name in ["vertices", "edges", "radii"]:
        np.testing.assert_array_equal(getattr(reference[1], name), getattr(streamed[1], name))
    packed = brook.skeletonize(labels, streaming=False, output="packed_device", **options)
    assert packed.on_device and packed.vertices.__cuda_array_interface__["shape"][1] == 3
    vertices = np.from_dlpack(packed.vertices, device="cpu", copy=True)
    np.testing.assert_array_equal(vertices, reference[1].vertices)
    assert len(merge_fragments([packed], [(0, 0, 0)], on_device=True)) == 1
    assert len(device_post.postprocess(packed, 0, 0)) == 1
    path = brook.connect_points(labels, (1, 2, 2), (14, 2, 2))
    assert len(path.vertices) > 1
    fields, copied = brook.oversegment(labels, reference[1])
    assert fields.max() > 0 and copied is not reference[1]
    mask = context().upload(labels.astype(bool))
    distance = context().edt(mask, (1, 1, 1), True)
    component = trace(mask, distance, **params)
    assert len(component.edges) and component.id is None and component.space == "voxel"
    assert brook.warmup() is None


def test_cpu_connect_points_signature():
    parameters = inspect.signature(brook.connect_points).parameters
    assert list(parameters) == ["labels", "start", "end", "anisotropy", "fill_holes", "in_place",
                                "pdrf_scale", "pdrf_exponent"]
    assert [parameters[name].default for name in list(parameters)[3:]] == [(1, 1, 1), False, False, 100000, 4]
    from brook import intake

    assert (intake.connect_points, intake.synapses_to_targets, intake.DimensionError) == (
        brook.connect_points, brook.synapses_to_targets, brook.DimensionError)          # kimimaro.intake's names


@pytest.mark.gpu
def test_gpu_connect_points_kimimaro_contract(gpu):
    # Expected values are kimimaro.connect_points 5.8.1 results. Its last line, `vertices *= anisotropy`
    # on the uint32 path, raises under NumPy's same_kind casting; the values below use that line as
    # kimimaro.skeletonize writes it, np.multiply(vertices, anisotropy, dtype=np.float32).
    from brook.contrib import igneous

    labels = np.zeros((12, 10, 4), np.uint32)
    labels[1:11, 1:4], labels[1:11, 6:9], labels[8:11, 1:9] = 7, 7, 7
    anisotropy = np.float32([4, 4, 40])
    path = [[2, 7, 2], [3, 7, 1], [4, 7, 1], [5, 7, 1], [6, 7, 1], [7, 7, 1], [8, 7, 1], [9, 6, 1], [9, 5, 1],
            [9, 4, 1], [9, 3, 1], [8, 2, 1], [7, 2, 1], [6, 2, 1], [5, 2, 1], [4, 2, 1], [3, 2, 1], [2, 2, 1]]
    skeleton = brook.connect_points(labels, (2, 2, 1), (2, 7, 2), anisotropy=(4, 4, 40))
    assert skeleton.space == "physical" and skeleton.id is None
    assert skeleton.vertices.dtype == skeleton.radii.dtype == np.float32 and skeleton.edges.dtype == np.uint32
    np.testing.assert_array_equal(skeleton.vertices, np.multiply(path, anisotropy, dtype=np.float32))
    np.testing.assert_array_equal(skeleton.edges, np.column_stack([np.arange(17), np.arange(1, 18)]))
    np.testing.assert_array_equal(skeleton.radii, np.full(18, 8, np.float32))
    np.testing.assert_array_equal(skeleton.vertex_types, np.zeros(18, np.uint8))
    np.testing.assert_array_equal(skeleton.transform, np.eye(3, 4, dtype=np.float32))
    assert [attribute["id"] for attribute in skeleton.extra_attributes] == ["radius", "vertex_types"]
    # The fifth positional argument is fill_holes, as in Kimimaro, not pdrf_scale.
    positional = brook.connect_points(labels, (2, 2, 1), (2, 7, 2), (4, 4, 40), True)
    np.testing.assert_array_equal(positional.vertices, skeleton.vertices)
    forwarded = igneous._Engine(None).connect_points(labels, (2, 2, 1), (2, 7, 2), (4, 4, 40), fill_holes=True, in_place=True)
    np.testing.assert_array_equal(forwarded.vertices, skeleton.vertices)
    weighted = brook.connect_points(labels, (2, 2, 1), (2, 7, 2), (4, 4, 40), False, False, 5000, 16)
    path = [[2, 7, 2], [3, 7, 1], [4, 7, 1], [5, 7, 1], [6, 7, 1], [7, 6, 1], [8, 5, 1], [9, 4, 1], [8, 3, 1],
            [7, 2, 1], [6, 2, 1], [5, 2, 1], [4, 2, 1], [3, 2, 1], [2, 2, 1]]
    np.testing.assert_array_equal(weighted.vertices, np.multiply(path, anisotropy, dtype=np.float32))
    np.testing.assert_array_equal(weighted.radii, np.float32([8, 8, 8, 8, 8, 4, 4, 8, 5.656854152679443, 8, 8, 8, 8, 8, 8]))
    # A sealed cavity: as in Kimimaro 5.8.1, fill_holes does not fill it (the filled tube gives the
    # straight path along (x, 4, 4)) and in_place leaves the input unchanged.
    tube = np.zeros((20, 9, 9), np.uint8)
    tube[:, 1:8, 1:8] = 1
    tube[3:17, 3:6, 3:6] = 0
    original = tube.copy()
    hollow = brook.connect_points(tube, (0, 4, 4), (19, 4, 4), fill_holes=True, in_place=True)
    path = [[19, 4, 4], [18, 3, 3], *([x, 2, 2] for x in range(17, 1, -1)), [1, 3, 3], [0, 4, 4]]
    np.testing.assert_array_equal(hollow.vertices, np.float32(path))
    root2, root3 = np.sqrt(np.float32(2)), np.sqrt(np.float32(3))
    np.testing.assert_array_equal(hollow.radii, np.float32([1, 2, root3, *[root2] * 14, root3, 2, 1]))
    np.testing.assert_array_equal(tube, original)
    # Physical space is what Kimimaro's cross_sectional_area expects of these vertices.
    mask = (labels > 0).astype(np.uint8)
    section = brook.connect_points(mask, (2, 2, 1), (2, 7, 2), anisotropy=(4, 4, 40))
    section.id = 1
    brook.cross_sectional_area(mask, section, anisotropy=(4, 4, 40))
    areas = np.float32([1447.1820068359375, 1929.5760498046875, 1920, 1920, 1920, 1920, 5120, 3620.38671875,
                        1920, 1920, 6400, 2715.2900390625, 1920, 1920, 1920, 1920, 1920, 1920])
    np.testing.assert_array_equal(section.cross_sectional_area, areas)
    np.testing.assert_array_equal(section.cross_sectional_area_contacts, np.uint8([32] + [48] * 17))


@pytest.mark.gpu
@pytest.mark.parametrize("order", ["C", "F"])
def test_gpu_connect_points_layout(gpu, order):
    # Kimimaro copies the mask to Fortran order before its EDT; for this ball the float32 EDT
    # depends on the axis order, so a C-order pass would give 3.2999999 at the centre.
    from brook.device import context

    grid = np.indices((7, 7, 7)) - 3
    ball = np.array((grid ** 2).sum(axis=0) <= 6, np.uint8, order=order)
    skeleton = brook.connect_points(ball, (1, 2, 2), (5, 4, 4), anisotropy=(1.1, 1.3, 1.7))
    path = [[5, 4, 4], [4, 3, 3], [3, 3, 3], [2, 3, 3], [1, 2, 2]]
    np.testing.assert_array_equal(skeleton.vertices, np.multiply(path, np.float32([1.1, 1.3, 1.7]), dtype=np.float32))
    np.testing.assert_array_equal(skeleton.radii, np.float32([1.100000023841858, 2.200000047683716, 3.3000001907348633,
                                                              2.200000047683716, 1.100000023841858]))
    # The low-level binding takes the same Fortran-order EDT.
    low_level = context().connect_points(ball, (1, 2, 2), (5, 4, 4), anisotropy=(1.1, 1.3, 1.7))
    np.testing.assert_array_equal(low_level.vertices, skeleton.vertices)
    np.testing.assert_array_equal(low_level.radii, skeleton.radii)


@pytest.mark.gpu
def test_gpu_worker_default_warmup(gpu):
    labels = np.ones((12, 5, 5), np.uint8, order="F")
    options = dict(dust_threshold=0, fix_borders=False, progress=False,
                   teasar_params=dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4))
    expected = brook.skeletonize(labels, **options)
    with brook.GpuPool(devices=[0]) as pool:
        result = pool.map([(0, labels)], **options)
    assert result[0][0] == 0
    for name in ["vertices", "edges", "radii"]:
        np.testing.assert_array_equal(getattr(result[0][1][1], name), getattr(expected[1], name))


def _components(skeleton):
    parent = list(range(len(skeleton.vertices)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in skeleton.edges.tolist():
        parent[find(a)] = find(b)
    return len({find(i) for i in range(len(parent))})


@pytest.mark.gpu
@pytest.mark.parametrize("lockstep", ["0", "1"])
@pytest.mark.parametrize("fix_branching", [True, False])
def test_gpu_soma_paths_stay_attached(gpu, monkeypatch, lockstep, fix_branching):
    # A ball with six processes. Paths are trimmed within the soma radius; with branching correction
    # they are target-first, so a rail end inside the radius must be replaced by the soma root.
    monkeypatch.setenv("BROOK_LOCKSTEP", lockstep)
    size, centre = 72, 36
    grid = np.indices((size,) * 3) - centre
    labels = ((grid ** 2).sum(axis=0) <= 12 ** 2).astype(np.uint8)
    for axis in range(3):
        tube = [slice(centre - 1, centre + 2)] * 3
        tube[axis] = slice(2, size - 2)
        labels[tuple(tube)] = 1
    labels = np.asfortranarray(labels)
    params = dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4, soma_detection_threshold=6,
                  soma_acceptance_threshold=9, soma_invalidation_scale=1.0, soma_invalidation_const=2)
    skeleton = brook.skeletonize(labels, teasar_params=params, dust_threshold=0, fix_branching=fix_branching,
                                 fix_borders=False, progress=False, streaming=False)[1]
    offsets = np.abs(skeleton.vertices - centre)
    assert (offsets.max(axis=1) > 30).sum() >= 6, "every process is traced"
    assert np.any(np.all(skeleton.vertices == centre, axis=1)), "the soma root is a vertex"
    assert _components(skeleton) == 1


@pytest.mark.gpu
def test_gpu_device_resident_input(gpu):
    # A device-resident label volume (CUDA array interface) skeletonizes without a host round trip
    # and gives the host-input result, whether the device layout is Fortran or C order.
    from brook.device import context

    n = 48
    z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    labels = np.zeros((n, n, n), np.uint32)
    labels[(x - 20) ** 2 + (y - 20) ** 2 <= 9] = 1
    labels[30:34, 6:42, 30:34] = 2
    options = dict(teasar_params=dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4),
                   dust_threshold=10, fix_borders=False, progress=False)
    expected = brook.skeletonize(np.asfortranarray(labels), **options)
    ctx = context()
    fortran = ctx.upload(np.asfortranarray(labels))                 # Fortran-contiguous device array
    c_order = ctx.upload_tensor(np.ascontiguousarray(labels))       # C-contiguous device array
    assert fortran.__cuda_array_interface__["strides"] != c_order.__cuda_array_interface__["strides"]
    binary = brook.skeletonize(np.asfortranarray(labels > 0), **options)     # one label: both objects merge
    for device_input, reference in ((fortran, expected), (c_order, expected), (ctx.upload(np.asfortranarray(labels > 0)), binary)):
        result = brook.skeletonize(device_input, **options)
        assert list(result) == list(reference)
        for label in reference:
            for name in ["vertices", "edges", "radii"]:
                np.testing.assert_array_equal(getattr(result[label], name), getattr(reference[label], name))
    np.testing.assert_array_equal(fortran.get(), np.asfortranarray(labels))   # the input is left untouched
    packed = brook.skeletonize(fortran, output="packed_device", **options)
    assert packed.on_device and len(packed) == len(expected)
    with pytest.raises(NotImplementedError):
        brook.skeletonize(fortran, object_ids=[1], **options)


def _batch_samples():
    # Mixed shapes and dtypes, with and without a soma-sized body, on the host and on the device.
    n = 40
    z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    ball = np.zeros((n, n, n), np.uint32)
    ball[(x - 14) ** 2 + (y - 14) ** 2 + (z - 14) ** 2 <= 100] = 1
    ball[12:17, 12:17, 14:38] = 1
    ball[28:32, 4:36, 28:32] = 2
    tube = np.zeros((24, 30, 50), np.uint16)
    tube[10:14, 12:16, 3:47] = 5
    tube[2:22, 20:24, 20:24] = 9
    sheet = np.zeros((6, 64, 64), np.uint8)
    sheet[2:4, 4:60, 4:60] = 1
    sheet[2:4, 30:34, :] = 0
    return [ball, tube, sheet]


@pytest.mark.gpu
def test_gpu_batch_identity(gpu):
    from brook.device import context

    samples = _batch_samples()
    params = dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4, soma_detection_threshold=6,
                  soma_acceptance_threshold=8, soma_invalidation_scale=1., soma_invalidation_const=0)
    options = dict(teasar_params=params, dust_threshold=10, fix_borders=True, progress=False)
    expected = [brook.skeletonize(np.asfortranarray(s), streaming=False, **options) for s in samples]
    ctx = context()
    inputs = [samples[0], ctx.upload(np.asfortranarray(samples[1])), ctx.upload_tensor(np.ascontiguousarray(samples[2]))]
    for order in ([0, 1, 2], [2, 0, 1]):
        batch = brook.skeletonize_batch([inputs[i] for i in order], **options)
        assert len(batch) == 3 and batch.on_device
        assert batch.s_off.tolist() == np.cumsum([0] + [len(expected[i]) for i in order]).tolist()
        for position, i in enumerate(order):
            result = batch[position].to_skeletons()
            assert list(result) == list(expected[i])
            for label in expected[i]:
                for name in ["vertices", "edges", "radii"]:
                    np.testing.assert_array_equal(getattr(result[label], name), getattr(expected[i][label], name))
        assert batch.sample.tolist() == [p for p, i in enumerate(order) for _ in expected[i]]
    listed = brook.skeletonize_batch(inputs, output="skeletons", **options)
    assert [list(d) for d in listed] == [list(e) for e in expected]
    dense = np.stack([samples[0], samples[0] * 0 + (samples[0] > 0) * 3])          # (B, Z, Y, X) host batch
    stacked = brook.skeletonize_batch(dense, output="packed", **options)
    assert not stacked.on_device and len(stacked) == 2 and stacked.labels.tolist()[:2] == [1, 2]
    flat = ctx.upload_tensor(np.ascontiguousarray(dense.reshape(-1, *dense.shape[2:])))   # a 3-D device buffer ...

    class DenseView:                                                                   # ... presented as (B, Z, Y, X)
        __cuda_array_interface__ = dict(flat.__cuda_array_interface__)
        __cuda_array_interface__["shape"] = dense.shape
        __cuda_array_interface__["strides"] = (dense.shape[1] * flat.strides[0],) + tuple(flat.strides)

    dense_device = brook.skeletonize_batch(DenseView(), **options)
    assert dense_device.labels.tolist() == stacked.labels.tolist()
    np.testing.assert_array_equal(dense_device.vertices.get(), stacked.vertices)
    padded = stacked.padded()
    assert padded["vertices"].shape[:2] == (2, 2) and padded["mask"].any()


@pytest.mark.gpu
def test_gpu_batch_sub_batching(gpu, monkeypatch):
    # A tiny memory budget forces one sub-batch per sample; results must not change.
    samples = _batch_samples() * 2
    options = dict(teasar_params=dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4, soma_detection_threshold=6,
                                      soma_acceptance_threshold=8, soma_invalidation_scale=1., soma_invalidation_const=0),
                   dust_threshold=10, fix_borders=True, progress=False)
    whole = brook.skeletonize_batch(samples, output="packed", **options)
    monkeypatch.setenv("BROOK_BATCH_BUDGET", "1")
    split = brook.skeletonize_batch(samples, output="packed", **options)
    from brook.device import context
    assert context().stats["batch_sub_batches"] == len(samples)
    assert whole.labels.tolist() == split.labels.tolist() and whole.s_off.tolist() == split.s_off.tolist()
    for name in ["vertices", "edges", "radii", "v_off", "e_off"]:
        np.testing.assert_array_equal(getattr(whole, name), getattr(split, name))


@pytest.mark.gpu
def test_gpu_batch_uniform_shape(gpu, monkeypatch):
    # Samples of one shape take the stacked path: shared preamble, one trace. Same results as
    # per-sample calls, including colliding label ids, dust, borders and a soma-sized body.
    from brook.device import context

    canvas = (40, 64, 64)
    samples = []
    for source in _batch_samples():
        sample = np.zeros(canvas, np.int32)
        z, y, x = source.shape
        part = source[:38, :61, :59]
        sample[2:2 + part.shape[0], 3:3 + part.shape[1], 5:5 + part.shape[2]] = part
        sample[36:38, 50:60, 50:60] = 7          # a component below the dust threshold
        samples.append(sample)
    samples.append(samples[0].copy())            # identical labels in two samples: ids must not collide
    params = dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4, soma_detection_threshold=6,
                  soma_acceptance_threshold=8, soma_invalidation_scale=1., soma_invalidation_const=0)
    options = dict(teasar_params=params, dust_threshold=150, fix_borders=True, progress=False)
    expected = [brook.skeletonize(np.asfortranarray(s), streaming=False, **options) for s in samples]
    batch = brook.skeletonize_batch(samples, **options)
    assert context().stats["batch_uniform"] == 1
    assert len(batch) == len(samples)
    for i, reference in enumerate(expected):
        result = batch[i].to_skeletons()
        assert list(result) == list(reference), i
        for label in reference:
            for name in ["vertices", "edges", "radii"]:
                np.testing.assert_array_equal(getattr(result[label], name), getattr(reference[label], name))
    monkeypatch.setenv("BROOK_BATCH_UNIFORM", "0")           # the generic path gives the same batch
    generic = brook.skeletonize_batch(samples, output="packed", **options)
    host = batch.to_host()
    assert generic.labels.tolist() == host.labels.tolist()
    for name in ["vertices", "edges", "radii", "v_off", "e_off", "s_off"]:
        np.testing.assert_array_equal(getattr(generic, name), getattr(host, name))


def _targeted_samples():
    """Samples of one shape whose manual targets change what the plain trace finds.

    Sample 0 forces a path to the tip of a stub the invalidation ball swallows; sample 1 reaches
    that tip as an `after` target, which takes its component out of the lockstep trace; sample 2
    has no targets at all; sample 3 mixes points that land on background with one written from the
    far edge (a negative coordinate); sample 4
    is a soma-sized body, where the trace inserts the root ahead of the manual targets.
    """
    canvas = (28, 40, 40)
    samples, before, after = [], [], []

    def stubbed_bar():
        volume = np.zeros(canvas, np.uint16)
        volume[10:14, 10:14, 4:36] = 1                   # a bar across the volume
        volume[10:14, 14:20, 20:24] = 1                  # a stub the bar's invalidation swallows
        return volume

    samples.append(stubbed_bar()); before.append([(12, 19, 22)]); after.append([])
    samples.append(stubbed_bar()); before.append([]); after.append([(12, 19, 22)])
    plain = np.zeros(canvas, np.uint16)
    plain[10:14, 10:14, 4:36] = 3; plain[20:26, 24:34, 24:34] = 4
    samples.append(plain); before.append([]); after.append([])
    mixed = stubbed_bar(); mixed[2:6, 30:36, 30:36] = 2
    # (12, -21, 22) is the stub tip again, counted from the far edge of the sample's own shape
    samples.append(mixed); before.append([(1, 1, 1), (12, -21, 22)]); after.append([(2, 2, 2)])
    z, y, x = np.meshgrid(np.arange(28), np.arange(40), np.arange(40), indexing="ij")
    body = np.zeros(canvas, np.uint16)
    body[(x - 14) ** 2 + (y - 14) ** 2 + (z - 14) ** 2 <= 100] = 5
    body[12:16, 12:16, 14:38] = 5
    samples.append(body); before.append([(14, 14, 37)]); after.append([])
    return samples, before, after


@pytest.mark.gpu
def test_gpu_batch_extra_targets(gpu, monkeypatch):
    # Manual targets are per sample, in the sample's own coordinates: the stacked path, the
    # generic path and forced sub-batching all give what per-sample calls give.
    from brook.device import context

    samples, before, after = _targeted_samples()
    params = dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4, soma_detection_threshold=6,
                  soma_acceptance_threshold=8, soma_invalidation_scale=1., soma_invalidation_const=0)
    options = dict(teasar_params=params, dust_threshold=10, fix_borders=True, progress=False)
    expected = [brook.skeletonize(np.asfortranarray(sample), streaming=False, extra_targets_before=b,
                                  extra_targets_after=a, **options)
                for sample, b, a in zip(samples, before, after)]
    plain = [brook.skeletonize(np.asfortranarray(sample), streaming=False, **options) for sample in samples]
    for i in [0, 1, 3, 4]:           # the targets must matter, or the comparison proves nothing
        assert any(not np.array_equal(expected[i][label].vertices, plain[i][label].vertices) for label in plain[i]), i
    assert all(np.array_equal(expected[2][label].vertices, plain[2][label].vertices) for label in plain[2])

    def compare(tag, uniform, batch_samples=samples, batch_before=before, batch_after=after, reference=expected):
        batch = brook.skeletonize_batch(batch_samples, extra_targets_before=batch_before,
                                        extra_targets_after=batch_after, **options)
        assert dict(context().stats).get("batch_uniform", 0) == uniform, tag
        assert len(batch) == len(reference)
        for i, one in enumerate(reference):
            result = batch[i].to_skeletons()
            assert list(result) == list(one), (tag, i)
            for label in one:
                for name in ["vertices", "edges", "radii"]:
                    np.testing.assert_array_equal(getattr(result[label], name), getattr(one[label], name),
                                                  err_msg=f"{tag}: sample {i}, label {label}, {name}")

    compare("stacked", 1)
    monkeypatch.setenv("BROOK_BATCH_BUDGET", "1")             # one sample per sub-batch
    compare("stacked, sub-batched", 1)
    assert dict(context().stats)["batch_sub_batches"] == len(samples)
    monkeypatch.setenv("BROOK_BATCH_UNIFORM", "0")            # the generic path
    compare("generic, sub-batched", 0)
    monkeypatch.delenv("BROOK_BATCH_BUDGET")
    compare("generic", 0)
    monkeypatch.delenv("BROOK_BATCH_UNIFORM")

    # Mixed shapes: each sample's points are read in its own frame — these two are out of range
    # for the other sample, so a point that reached the wrong sample would be rejected.
    mixed = _batch_samples()
    mixed_before = [None, [(12, 14, 45)], []]
    mixed_after = [[], [], [(3, 40, 55)]]
    mixed_expected = [brook.skeletonize(np.asfortranarray(sample), streaming=False, extra_targets_before=b or [],
                                        extra_targets_after=a, **options)
                      for sample, b, a in zip(mixed, mixed_before, mixed_after)]
    compare("mixed shapes", 0, mixed, mixed_before, mixed_after, mixed_expected)

    # A sample that is one full-volume component cannot be stacked: the batch resumes on the
    # generic path from there, and every sample keeps its own targets across the switch.
    monkeypatch.setenv("BROOK_BATCH_BUDGET", "1")              # the samples before it stay stacked
    resumed = samples[:2] + [np.full(samples[0].shape, 6, np.uint16)]
    resumed_before = [before[0], before[1], [(0, 0, 0)]]
    resumed_after = [after[0], after[1], []]
    resumed_expected = [brook.skeletonize(np.asfortranarray(sample), streaming=False, extra_targets_before=b,
                                          extra_targets_after=a, **options)
                        for sample, b, a in zip(resumed, resumed_before, resumed_after)]
    assert len(resumed_expected[2]) == 1                        # the full-volume sample is skeletonized
    compare("stacked then generic", 0, resumed, resumed_before, resumed_after, resumed_expected)
    assert dict(context().stats)["batch_sub_batches"] == 3      # two stacked samples, then one generic
    monkeypatch.delenv("BROOK_BATCH_BUDGET")

    with pytest.raises(ValueError, match="per sample"):        # one flat list of points for the batch
        brook.skeletonize_batch(mixed, extra_targets_before=[(1, 2, 3), (4, 5, 6)], **options)
    with pytest.raises(ValueError, match="per sample"):
        brook.skeletonize_batch(mixed, extra_targets_after=np.array([[1, 2, 3]]), **options)
    with pytest.raises(ValueError, match="entries for 3 samples"):
        brook.skeletonize_batch(mixed, extra_targets_before=[[], []], **options)
def _cavity_samples():
    # Samples of one shape with enclosed cavities and avocado configurations: a slab with a background
    # cavity, a pit fully enclosed by a fruit, a pit open to one face beside a cavity, two fruits with
    # pits whose ids interleave. The pits' distance fields exceed the detection threshold (6 / 2.5).
    hole = np.zeros((40, 40, 40), np.uint32)
    hole[4:36, 4:36, 4:20] = 1
    hole[14:26, 14:26, 8:16] = 0
    hole[6:34, 30:34, 24:36] = 2
    avocado = np.zeros((40, 40, 40), np.uint32)
    avocado[2:38, 2:38, 2:30] = 3
    avocado[10:30, 10:30, 8:24] = 4
    avocado[4:36, 33:37, 31:39] = 5
    mixed = np.zeros((40, 40, 40), np.uint32)
    mixed[3:37, 3:37, 3:37] = 6
    mixed[8:20, 8:20, 3:15] = 7
    mixed[24:32, 24:32, 24:32] = 0
    two = np.zeros((40, 40, 40), np.uint16)
    two[2:38, 2:18, 2:38] = 8
    two[10:30, 6:14, 10:30] = 9
    two[2:38, 22:38, 2:38] = 10
    two[10:30, 26:34, 10:30] = 11
    return [hole, avocado, mixed, two]


_CAVITY_PARAMS = dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4, soma_detection_threshold=6,
                      soma_acceptance_threshold=8, soma_invalidation_scale=1., soma_invalidation_const=0)


def _assert_same_skeletons(result, reference, where):
    assert list(result) == list(reference), where
    for label in reference:
        for name in ["vertices", "edges", "radii"]:
            np.testing.assert_array_equal(getattr(result[label], name), getattr(reference[label], name), err_msg=f"{where} {label} {name}")


@pytest.mark.gpu
@pytest.mark.parametrize("fill_holes,fix_avocados", [(True, False), (False, True), (True, True)])
def test_gpu_batch_holes_and_avocados(gpu, monkeypatch, fill_holes, fix_avocados):
    # fill_holes and fix_avocados in batches: the stacked path (holes filled box by box on the stack,
    # avocados corrected sample by sample on it), the generic path and forced sub-batching all give
    # the per-sample results, including the merged pits, the reordered labels of the avocado stage
    # and the infinite radius the stage's unrefreshed distance field produces at a painted cavity.
    from brook.device import context

    samples = _cavity_samples()
    options = dict(teasar_params=_CAVITY_PARAMS, dust_threshold=10, fix_borders=True, progress=False,
                   fill_holes=fill_holes, fix_avocados=fix_avocados)
    expected = [brook.skeletonize(np.asfortranarray(s), streaming=False, **options) for s in samples]
    if fix_avocados:
        assert 4 not in expected[1] and 7 not in expected[2] and list(expected[3]) == [8, 10], "the pits merge"
    if fix_avocados and not fill_holes:
        assert np.isinf(expected[0][1].radii).sum() == 1, "a painted cavity of the hole sample is a vertex"

    def check(batch, where, extra=0):
        assert len(batch) == len(samples) + extra
        for i, reference in enumerate(expected):
            _assert_same_skeletons(batch[i].to_skeletons(), reference, f"{where} sample {i}")

    stacked = brook.skeletonize_batch(samples, output="packed", **options)
    stats = context().stats
    assert stats["batch_uniform"] == 1
    assert (stats.get("hole_filled_voxels", 0) > 0) == fill_holes
    assert ("avocado_passes" in stats) == fix_avocados
    check(stacked, "stacked")
    monkeypatch.setenv("BROOK_BATCH_UNIFORM", "0")
    generic = brook.skeletonize_batch(samples, output="packed", **options)
    assert context().stats["batch_sub_batches"] == 1
    check(generic, "generic")
    monkeypatch.setenv("BROOK_BATCH_BUDGET", "1")
    split = brook.skeletonize_batch(samples, output="packed", **options)
    assert context().stats["batch_sub_batches"] == len(samples)
    check(split, "sub-batched")
    monkeypatch.delenv("BROOK_BATCH_UNIFORM")                        # stacked, one sample per sub-batch
    split = brook.skeletonize_batch(samples, output="packed", **options)
    assert context().stats["batch_uniform"] == 1 and context().stats["batch_sub_batches"] == len(samples)
    check(split, "stacked sub-batched")
    monkeypatch.delenv("BROOK_BATCH_BUDGET")
    cropped = np.ascontiguousarray(samples[1][:, :, :30])           # mixed shapes: the generic path
    mixed = brook.skeletonize_batch(samples + [cropped], output="packed", **options)
    assert "batch_uniform" not in context().stats
    check(mixed, "mixed", extra=1)
    _assert_same_skeletons(mixed[4].to_skeletons(), brook.skeletonize(np.asfortranarray(cropped), streaming=False, **options), "mixed cropped")


@pytest.mark.gpu
@pytest.mark.parametrize("fix_branching", [True, False])
def test_gpu_avocado_lockstep_matches_component_tracing(gpu, monkeypatch, fix_branching):
    # Avocado-corrected volumes trace through the lockstep tracer with the component tracer's results:
    # a foreground voxel at distance zero (a painted cavity, whose field the stage does not refresh)
    # is +inf to both, in the penalty, the invalidation ball and the radius.
    from brook.device import context

    for detection in (6, 1000):                                      # with and without somata
        params = dict(_CAVITY_PARAMS, soma_detection_threshold=detection, soma_acceptance_threshold=detection + 2)
        options = dict(teasar_params=params, dust_threshold=10, fix_borders=True, fix_branching=fix_branching,
                       fix_avocados=True, progress=False, streaming=False)
        for sample in _cavity_samples():
            monkeypatch.setenv("BROOK_LOCKSTEP", "0")
            reference = brook.skeletonize(np.asfortranarray(sample), **options)
            assert "lockstep_labels" not in context().stats
            monkeypatch.delenv("BROOK_LOCKSTEP")
            result = brook.skeletonize(np.asfortranarray(sample), **options)
            stats = context().stats
            assert stats["lockstep_labels"] >= 1
            _assert_same_skeletons(result, reference, f"detection {detection}")
            if sample[4, 4, 4] == 1 and detection == 6:                # the hole sample: painted cavity voxels
                assert stats["lockstep_zero_distances"] > 0


@pytest.mark.gpu
def test_gpu_avocado_unreachable_cavity(gpu, monkeypatch):
    # Without a voxel graph: the avocado stage paints the slab's enclosed cavity without a merge, so the
    # field is not recomputed and the painted voxels keep distance zero, an infinite penalty. With
    # branching correction a DAF target inside the cavity has no path to a rail (dijkstra3d's railroad
    # returns none, and Kimimaro 5.8.1 repeats it until max_paths); the component tracer stops there,
    # and the lockstep hands the label to it.
    from brook.device import context

    labels = np.zeros((26, 39, 26), np.uint32)
    labels[2:24, 3:38, 1:23] = 1
    labels[8:18, 17:26, 9:16] = 0
    labels = np.asfortranarray(labels)
    options = dict(teasar_params=_CAVITY_PARAMS, dust_threshold=10, fix_avocados=True, progress=False, streaming=False)
    monkeypatch.setenv("BROOK_LOCKSTEP", "0")
    reference = brook.skeletonize(labels, **options)
    assert context().stats["component_unreachable_stops"] == 1 and context().stats["avocado_passes"] == 1
    monkeypatch.delenv("BROOK_LOCKSTEP")
    result = brook.skeletonize(labels, **options)
    assert context().stats["lockstep_labels"] == 1 and context().stats["ejected"] == 1
    _assert_same_skeletons(result, reference, "lockstep")
    assert list(result) == [1] and len(result[1].vertices) - len(result[1].edges) == 1


@pytest.mark.gpu
def test_gpu_railroad_without_reachable_rail(gpu):
    # Context.railroad returns the path from the target to the nearest rail (zero cost), and no path when
    # an infinite cost walls the target off from every rail, as dijkstra3d's railroad does. Every rail of
    # the plane costs 3: the lowest flat index among them, (0, 0, 0), fixes the last road voxel (1, 1, 1),
    # and the path joins that voxel's first rail in dijkstra3d's order, its -z face.
    from brook import _runtime
    from brook.device import context

    ctx = context()
    field = np.ones((9, 9, 9), np.float32, order="F")
    field[:, :, 0] = 0
    target = int(np.ravel_multi_index((4, 4, 4), field.shape, order="F"))
    workspace = _runtime.engine().TraceWorkspace(ctx, field.size)
    path = ctx.railroad(ctx.upload(field), target, workspace)
    road = [(4, 4, 4), (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 0)]
    assert path == [int(np.ravel_multi_index(p, field.shape, order="F")) for p in road]
    walled = field.copy(order="F")
    walled[2:7, 2:7, 2:7] = np.inf
    walled[3:6, 3:6, 3:6] = 1
    assert ctx.railroad(ctx.upload(walled), target, workspace) == []
    assert len(ctx.railroad(ctx.upload(field), target, workspace)) == 5          # the workspace is reusable


# dijkstra3d's neighbour order (compute_neighborhood): faces, the xy, yz and xz edges, then the corners
_D3_ORDER = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
             (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0), (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
             (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
             (-1, -1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1), (1, 1, 1)]


@pytest.mark.gpu
def test_gpu_railroad_join_order(gpu):
    # Every rail next to the last road voxel costs the same (entering a rail is free). The path joins the
    # one dijkstra3d's railroad joins: that voxel's first rail in dijkstra3d's neighbour order, among those
    # the voxel graph lets it step to. Expected values: dijkstra3d 1.15.2's railroad for every single rail,
    # every pair and all 26 around the target, and for every pair with the first one's direction cut from
    # the graph at the target (the lowest flat index differs on 143 of the 352 sets, Brook's own neighbour
    # table on 7).
    from brook import _runtime
    from brook.device import context

    ctx = context()
    shape, centre = (5, 5, 5), (2, 2, 2)

    def flat(p):
        return int(np.ravel_multi_index(p, shape, order="F"))

    near = [tuple(c + d for c, d in zip(centre, offset)) for offset in _D3_ORDER]
    workspace = _runtime.engine().TraceWorkspace(ctx, 125)
    for rails in [(i,) for i in range(26)] + list(itertools.combinations(range(26), 2)) + [tuple(range(26))]:
        field = np.ones(shape, np.float32, order="F")
        for i in rails:
            field[near[i]] = 0
        assert ctx.railroad(ctx.upload(field), flat(centre), workspace) == [flat(centre), flat(near[rails[0]])], rails
        if len(rails) == 2:
            graph = np.full(shape, 0x3FFFFFF, np.uint32, order="F")
            graph[centre] &= ~np.uint32(1 << _GRAPH_BITS[_D3_ORDER[rails[0]]])
            path = ctx.railroad(ctx.upload(field), flat(centre), workspace, voxel_graph=ctx.upload(graph))
            assert path == [flat(centre), flat(near[rails[1]])], rails


@pytest.mark.gpu
def test_gpu_avocado_stage_segmented(gpu):
    # The avocado stage on a stack of samples (its `segment`) equals the per-sample stage: labels
    # renumbered per sample, mapping, roots and the distance field, refreshed only where a sample changed.
    from brook.device import context

    ctx = context()
    samples = [s.astype(np.uint32) for s in _cavity_samples()]
    depth, voxels = samples[0].shape[2], samples[0].size
    per_sample, base, offset = [], [], 0
    for s in samples:
        cc, mapping, roots = ctx.connected_components(ctx.upload(np.asfortranarray(s)))
        dbf = ctx.edt(cc, (1., 1., 1.), False)
        per_sample.append(ctx._fix_avocados(cc, dbf, mapping, 6., (1., 1., 1.), False))
        base.append(offset)
        offset += int(s.max())
    stack = np.concatenate([np.where(s > 0, s + b, 0).astype(np.uint32) for s, b in zip(samples, base)], axis=2)
    cc, mapping, roots = ctx.connected_components(ctx.upload(np.asfortranarray(stack)))
    dbf = ctx.edt(cc, (1., 1., 1.), False, 3, depth)                 # z lines per sample, as the stacked path
    labels, distance, mapping, roots = ctx._fix_avocados(cc, dbf, mapping, 6., (1., 1., 1.), False, roots, depth)
    labels, distance = labels.get(), distance.get()
    assert ctx.stats["avocado_passes"] == 1
    first = 0
    for i, (local, field, m, r) in enumerate(per_sample):
        local, field, count = local.get(), field.get(), len(r)
        z = slice(i * depth, (i + 1) * depth)
        np.testing.assert_array_equal(labels[:, :, z], np.where(local > 0, local + first, 0))
        np.testing.assert_array_equal(distance[:, :, z], field)
        assert mapping[first + 1:first + count + 1] == [v + base[i] for v in m[1:]]
        assert roots[first:first + count] == [v + i * voxels for v in r]
        first += count
    assert first == len(roots) == len(mapping) - 1 == 7


def _avocado_rule_samples():
    # Kimimaro runs the avocado stage on Fortran-ordered arrays: its argmax takes the first maximum in
    # x-fastest order and fastremap renumbers the corrected labels by first voxel in that order. A pit (2)
    # has two lobes at distance 2, lobe A (centre (10, 3, 3)) inside a one-voxel fruit shell (1) and lobe B
    # (centre (3, 3, 10)) in the open, joined through a hole in the shell. A comes first, its rays meet the
    # fruit on all six sides and the pit merges; with x and z swapped B comes first and both labels stay.
    # The third sample's labels have their first voxels in the order 3, 7, 5.
    tie = np.zeros((16, 9, 16), np.uint32)
    tie[8:13, 1:6, 1:6] = 1
    tie[9:12, 2:5, 2:5] = 2
    tie[3:9, 2, 2] = 2
    tie[3, 2, 2:9] = 2
    tie[2:5, 2:5, 9:12] = 2
    order = np.zeros((16, 9, 16), np.uint32)
    order[0:3, 0:3, 10:13] = 5
    order[10:13, 0:3, 0:3] = 3
    order[5:8, 0:3, 5:8] = 7
    return [tie, np.ascontiguousarray(tie.transpose(2, 1, 0)), order]


_AVOCADO_RULE_PARAMS = dict(scale=1.5, const=1, pdrf_scale=100000, pdrf_exponent=4, soma_detection_threshold=4.5,
                            soma_acceptance_threshold=1e6, soma_invalidation_scale=1., soma_invalidation_const=0)
_AVOCADO_RULE_KEYS = [[2], [2, 1], [3, 7, 5]]      # kimimaro.skeletonize(sample, fix_avocados=True), any layout
_AVOCADO_REFRESH_ANISOTROPY = (1.1, 1.3, 2.0)


def _avocado_refresh_sample():
    # A pit filling a one-voxel fruit shell merges, and the stage refreshes the distance field. Kimimaro
    # computes it with edt.edt on its Fortran-ordered labels; with anisotropy (1.1, 1.3, 2) that pass order
    # gives 3.3000002 (0x40533334) on the centre line, where the C-order pass sequence gives 3.2999999.
    sample = np.zeros((7, 7, 7), np.uint32)
    sample[1:6, 1:6, 1:6] = 1
    sample[2:5, 2:5, 2:5] = 2
    return sample


def _avocado_graph_sample():
    # The refresh sample's layout in a 9^3 volume with the labels' own 26-connectivity graph, which cuts
    # the pit from the fruit. Kimimaro refreshes the merged field with that graph: the cut stays a
    # boundary, and the field keeps its values 0.5 (shell), 1.5 and 2.5 (pit centre).
    labels = np.zeros((9, 9, 9), np.uint32)
    labels[1:8, 1:8, 1:8] = 1
    labels[2:7, 2:7, 2:7] = 2
    none = np.zeros(labels.shape, bool)
    return np.asfortranarray(labels), np.asfortranarray(_voxel_graph(labels, none, none))


@pytest.mark.gpu
@pytest.mark.parametrize("lockstep", ["0", "1"])
@pytest.mark.parametrize("fix_branching", [True, False])
def test_gpu_avocado_ties_and_numbering_follow_kimimaro(gpu, monkeypatch, lockstep, fix_branching):
    # Which labels survive and their order, from Kimimaro 5.8.1, on host input of either layout, device
    # input of either layout, and every batch path.
    from brook.device import context

    monkeypatch.setenv("BROOK_LOCKSTEP", lockstep)
    ctx = context()
    options = dict(teasar_params=_AVOCADO_RULE_PARAMS, dust_threshold=0, fix_branching=fix_branching,
                   fix_avocados=True, progress=False)
    samples = _avocado_rule_samples()
    for sample, keys in zip(samples, _AVOCADO_RULE_KEYS):
        for volume in (np.asfortranarray(sample), np.ascontiguousarray(sample), sample.astype(np.uint8),
                       ctx.upload(np.asfortranarray(sample)), ctx.upload_tensor(np.ascontiguousarray(sample))):
            assert list(brook.skeletonize(volume, streaming=False, **options)) == keys
    batch = brook.skeletonize_batch(samples, output="skeletons", **options)
    assert [list(result) for result in batch] == _AVOCADO_RULE_KEYS
    assert context().stats.get("batch_uniform", 0) == (lockstep == "1")      # the stacked path
    monkeypatch.setenv("BROOK_BATCH_UNIFORM", "0")
    batch = brook.skeletonize_batch(samples, output="skeletons", **options)
    assert [list(result) for result in batch] == _AVOCADO_RULE_KEYS
    monkeypatch.setenv("BROOK_BATCH_BUDGET", "1")                    # generic, one sample per sub-batch
    batch = brook.skeletonize_batch(samples, output="skeletons", **options)
    assert [list(result) for result in batch] == _AVOCADO_RULE_KEYS
    monkeypatch.delenv("BROOK_BATCH_UNIFORM")                         # stacked (with lockstep), one sample per sub-batch
    batch = brook.skeletonize_batch(samples, output="skeletons", **options)
    assert [list(result) for result in batch] == _AVOCADO_RULE_KEYS
    stats = context().stats                                           # without lockstep every sample traces alone
    assert stats.get("batch_uniform", 0) == (lockstep == "1")
    assert stats["batch_sub_batches"] == (len(samples) if lockstep == "1" else 0)


@pytest.mark.gpu
def test_gpu_avocado_stage_follows_kimimaro(gpu):
    # The stage's labels, original-label mapping, roots and distance field, alone and on a stack of the
    # samples, against kimimaro.intake.engage_avocado_protection on the same volumes.
    from brook.device import context

    ctx = context()
    samples = _avocado_rule_samples()
    expected = [([0, 2], [168], {1: 1, 2: 1}, True),               # (mapping, roots, input label -> id, refreshed)
                ([0, 2, 1], [329, 1169], {2: 1, 1: 2}, False),
                ([0, 3, 7, 5], [10, 725, 1440], {3: 1, 7: 2, 5: 3}, False)]
    for sample, (mapping, roots, ids, refreshed) in zip(samples, expected):
        cc, m, r = ctx.connected_components(ctx.upload(np.asfortranarray(sample)))
        dbf = ctx.edt(cc, (1., 1., 1.), False)
        field = dbf.get()
        labels, distance, m, r = ctx._fix_avocados(cc, dbf, m, 4.5, (1., 1., 1.), False)
        labels = labels.get()
        assert (m, r) == (mapping, roots)
        np.testing.assert_array_equal(labels, np.select([sample == k for k in ids], list(ids.values()), 0))
        if refreshed:
            field = ctx.edt(ctx.upload(np.asfortranarray(labels)), (1., 1., 1.), False).get()
        np.testing.assert_array_equal(distance.get(), field)
    # The same samples stacked along z: numbered by first voxel across the stack, so the roots are sorted.
    depth, voxels = samples[0].shape[2], samples[0].size
    stack = np.concatenate([np.where(samples[0] > 0, samples[0], 0), np.where(samples[1] > 0, samples[1] + 2, 0),
                            np.where(samples[2] > 0, samples[2] + 4, 0)], axis=2).astype(np.uint32)
    cc, m, r = ctx.connected_components(ctx.upload(np.asfortranarray(stack)))
    dbf = ctx.edt(cc, (1., 1., 1.), False, 3, depth)
    labels, distance, m, r = ctx._fix_avocados(cc, dbf, m, 4.5, (1., 1., 1.), False, r, depth)
    assert m == [0, 2, 4, 3, 7, 11, 9]
    assert r == [168, 329 + voxels, 1169 + voxels, 10 + 2 * voxels, 725 + 2 * voxels, 1440 + 2 * voxels]
    # The anisotropic refresh.
    sample, aniso = _avocado_refresh_sample(), _AVOCADO_REFRESH_ANISOTROPY
    cc, m, r = ctx.connected_components(ctx.upload(np.asfortranarray(sample)))
    labels, distance, m, r = ctx._fix_avocados(cc, ctx.edt(cc, aniso, False), m, 5., aniso, False)
    labels, distance = labels.get(), distance.get()
    assert (m, r) == ([0, 1], [57])
    np.testing.assert_array_equal(labels, sample > 0)
    assert distance[3, 3, 2:5].view(np.uint32).tolist() == [0x40533334] * 3
    np.testing.assert_array_equal(distance, ctx.edt(ctx.upload(np.asfortranarray(labels)), aniso, False).get())


@pytest.mark.gpu
@pytest.mark.parametrize("lockstep", ["0", "1"])
def test_gpu_avocado_refresh_follows_kimimaro(gpu, monkeypatch, lockstep):
    # Kimimaro 5.8.1's skeleton of the refreshed sample, bit for bit, single and batched.
    from brook.device import context

    monkeypatch.setenv("BROOK_LOCKSTEP", lockstep)
    sample, aniso = _avocado_refresh_sample(), _AVOCADO_REFRESH_ANISOTROPY
    vertices = np.arange(1, 6, dtype=np.float32)[:, None] * np.array(aniso, np.float32)
    radii = [0x3f8ccccd, 0x400ccccd, 0x40533334, 0x400ccccd, 0x3f8ccccd]
    for fix_branching in (True, False):
        options = dict(teasar_params=dict(_AVOCADO_RULE_PARAMS, soma_detection_threshold=5), anisotropy=aniso,
                       dust_threshold=0, fix_branching=fix_branching, fix_avocados=True, progress=False)
        single = brook.skeletonize(np.ascontiguousarray(sample), streaming=False, **options)
        batch = brook.skeletonize_batch([sample, np.ascontiguousarray(sample)], output="skeletons", **options)
        assert context().stats.get("batch_uniform", 0) == (lockstep == "1")
        for result in [single] + batch:
            assert list(result) == [1]
            np.testing.assert_array_equal(result[1].vertices, vertices)
            np.testing.assert_array_equal(result[1].edges, [[0, 1], [1, 2], [2, 3], [3, 4]])
            assert result[1].radii.view(np.uint32).tolist() == radii


@pytest.mark.gpu
@pytest.mark.parametrize("lockstep", ["0", "1"])
def test_gpu_avocado_refresh_uses_voxel_graph(gpu, monkeypatch, lockstep):
    # With a voxel graph, Kimimaro 5.8.1 refreshes the merged field with the graph EDT: the stage's
    # labels, mapping, roots and field, and the skeleton's radii (all 0.5: the trace stays on the shell),
    # on host and device input and in a batch (the generic path).
    from brook.device import context

    monkeypatch.setenv("BROOK_LOCKSTEP", lockstep)
    ctx = context()
    labels, graph = _avocado_graph_sample()
    cc, m, r = ctx.graph_components(ctx.upload(labels), ctx.upload(graph))
    initial = ctx.graph_edt(cc, ctx.upload(graph), (1., 1., 1.), False)
    field = initial.get()
    corrected, distance, m, r = ctx._fix_avocados(cc, initial, m, 3., (1., 1., 1.), False, voxel_graph=ctx.upload(graph))
    corrected, distance = corrected.get(), distance.get()
    assert (m, r) == ([0, 1], [91])
    np.testing.assert_array_equal(corrected, labels > 0)
    values, counts = np.unique(distance, return_counts=True)
    assert (values.tolist(), counts.tolist()) == ([0., .5, 1.5, 2.5], [386, 316, 26, 1])
    np.testing.assert_array_equal(distance, field)                   # the cut keeps the initial field here
    params = dict(_AVOCADO_RULE_PARAMS, soma_detection_threshold=3)
    for fix_branching in (True, False):
        options = dict(teasar_params=params, dust_threshold=0, fix_branching=fix_branching, fix_avocados=True, progress=False)
        results = [brook.skeletonize(labels, voxel_graph=graph, streaming=False, **options),
                   brook.skeletonize(ctx.upload(labels), voxel_graph=ctx.upload(graph), **options)]
        results += brook.skeletonize_batch([labels, labels], voxel_graph=[graph, graph], output="skeletons", **options)
        assert context().stats.get("batch_uniform", 0) == 0
        for result in results:
            assert list(result) == [1]
            assert len(result[1].radii) and set(result[1].radii.tolist()) == {0.5}


@pytest.mark.gpu
def test_gpu_border_targets_match_serial(gpu, monkeypatch):
    # The face-stack border plan agrees with the independent serial implementation, ties included.
    from brook.device import context

    ctx = context()
    touched = False
    for sample in _batch_samples() + [np.asfortranarray(np.ones((5, 7, 9), np.uint8))]:
        cc, mapping, roots = ctx.connected_components(ctx.upload(np.asfortranarray(sample)))
        batched = ctx.border_targets(cc, len(roots), (4., 4., 40.))
        monkeypatch.setenv("BROOK_BORDER_BATCH", "0")
        serial = ctx.border_targets(cc, len(roots), (4., 4., 40.))
        monkeypatch.delenv("BROOK_BORDER_BATCH")
        assert [list(map(tuple, b)) for b in batched] == [list(map(tuple, s)) for s in serial]
        touched = touched or any(len(b) for b in batched)
    assert touched


@pytest.mark.gpu
def test_gpu_borrowed_input_and_streams(gpu):
    # A Fortran-contiguous device array is read in place, left untouched and released after the
    # call; C-order input falls back to the copy; explicit and interface-declared producer streams
    # are accepted.
    import sys

    from brook.device import context

    n = 48
    z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    labels = np.zeros((n, n, n), np.uint32)
    labels[(x - 20) ** 2 + (y - 20) ** 2 <= 9] = 1
    labels[30:34, 6:42, 30:34] = 2
    options = dict(teasar_params=dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4),
                   dust_threshold=10, fix_borders=False, progress=False)
    expected = brook.skeletonize(np.asfortranarray(labels), **options)
    ctx = context()
    fortran = ctx.upload(np.asfortranarray(labels))
    before = sys.getrefcount(fortran)
    for kwargs in (dict(), dict(borrow=False), dict(stream=1), dict(stream=2)):
        result = brook.skeletonize(fortran, **options, **kwargs)
        assert list(result) == list(expected)
        for label in expected:
            np.testing.assert_array_equal(result[label].vertices, expected[label].vertices)
    assert sys.getrefcount(fortran) == before                    # the borrow was released
    np.testing.assert_array_equal(fortran.get(), np.asfortranarray(labels))

    class Declared:                                              # CUDA array interface with a stream entry
        __cuda_array_interface__ = dict(fortran.__cuda_array_interface__, stream=1)

    assert list(brook.skeletonize(Declared(), **options)) == list(expected)
    c_order = ctx.upload_tensor(np.ascontiguousarray(labels))
    assert list(brook.skeletonize(c_order, borrow=True, **options)) == list(expected)
    batch = brook.skeletonize_batch([fortran, c_order], stream=1, **options)
    assert batch.labels.tolist() == list(expected) * 2


def test_contrib_torch_skeletonize_class_availability():
    # brook.contrib.torch imports torch lazily: the module itself always imports, and only
    # accessing Skeletonize needs torch installed (with a clear error otherwise).
    import brook.contrib.torch as bt

    assert set(bt.__all__) == {"skeletonize", "skeletonize_batch", "padded", "PackedSkeletonsT", "BatchedSkeletonsT"}
    with pytest.raises(AttributeError):
        _ = bt.not_a_real_attribute
    try:
        import torch
    except ImportError:
        with pytest.raises(ImportError, match="PyTorch"):
            _ = bt.Skeletonize
    else:
        assert issubclass(bt.Skeletonize, torch.nn.Module)
        assert bt.Skeletonize is bt.Skeletonize                 # built once, cached


@pytest.mark.gpu
def test_gpu_contrib_torch_skeletonize_matches_and_zero_copy(gpu):
    torch = pytest.importorskip("torch")
    from brook.contrib import torch as bt

    sample = _batch_samples()[0]
    params = dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4, soma_detection_threshold=6,
                  soma_acceptance_threshold=8, soma_invalidation_scale=1., soma_invalidation_const=0)
    options = dict(teasar_params=params, dust_threshold=10, fix_borders=True, progress=False)
    expected = brook.skeletonize(np.asfortranarray(sample), **options)

    def check(result):
        assert isinstance(result, bt.PackedSkeletonsT)
        assert result.vertices.is_cuda and result.edges.is_cuda and result.radii.is_cuda
        assert result.labels.device.type == "cpu" and result.labels.dtype == torch.int64
        assert result.labels.tolist() == list(expected)
        for i, label in enumerate(result.labels.tolist()):
            v0, v1 = int(result.v_off[i]), int(result.v_off[i + 1])
            e0, e1 = int(result.e_off[i]), int(result.e_off[i + 1])
            np.testing.assert_array_equal(result.vertices[v0:v1].cpu().numpy(), expected[label].vertices)
            np.testing.assert_array_equal(result.edges[e0:e1].cpu().numpy(), expected[label].edges)
            np.testing.assert_array_equal(result.radii[v0:v1].cpu().numpy(), expected[label].radii)

    # Default layout: a C-contiguous CUDA tensor is copied and transposed on the GPU.
    c_tensor = torch.as_tensor(sample, device="cuda")
    assert c_tensor.is_contiguous()
    check(bt.skeletonize(c_tensor, **options))

    # layout="fortran" rearranges the tensor so Brook borrows it in place (zero copy).
    check(bt.skeletonize(c_tensor, layout="fortran", **options))

    # A CPU tensor is uploaded automatically.
    check(bt.skeletonize(torch.as_tensor(sample), **options))

    # DLPack conversion of a Brook device array is zero-copy: same pointer, same device.
    packed = brook.skeletonize(np.asfortranarray(sample), output="packed_device", **options)
    direct = torch.from_dlpack(packed.vertices)
    assert direct.data_ptr() == packed.vertices.__cuda_array_interface__["data"][0]
    assert direct.device.type == "cuda"

    # The wrapped result's tensors outlive the Brook objects they came from.
    result = bt.skeletonize(c_tensor, **options)
    vertices, radii = result.vertices, result.radii
    before_v, before_r = vertices.clone(), radii.clone()
    del result, packed, direct
    import gc
    gc.collect()
    torch.cuda.synchronize()
    np.testing.assert_array_equal(vertices.cpu().numpy(), before_v.cpu().numpy())
    np.testing.assert_array_equal(radii.cpu().numpy(), before_r.cpu().numpy())


@pytest.mark.gpu
def test_gpu_contrib_torch_batch_and_padded(gpu):
    torch = pytest.importorskip("torch")
    from brook.contrib import torch as bt

    samples = _batch_samples()
    params = dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4, soma_detection_threshold=6,
                  soma_acceptance_threshold=8, soma_invalidation_scale=1., soma_invalidation_const=0)
    options = dict(teasar_params=params, dust_threshold=10, fix_borders=True, progress=False)

    expected = brook.skeletonize_batch(samples, **options).to_host()
    tensors = [torch.as_tensor(s, device="cuda") for s in samples]        # C-contiguous CUDA tensors
    result = bt.skeletonize_batch(tensors, **options)

    assert isinstance(result, bt.BatchedSkeletonsT) and len(result) == len(samples)
    assert result.vertices.is_cuda and result.edges.is_cuda and result.radii.is_cuda
    assert result.labels.device.type == "cpu" and result.sample.device.type == "cpu"
    assert result.labels.tolist() == expected.labels.tolist()
    assert result.sample.tolist() == expected.sample.tolist()
    assert result.s_off.tolist() == expected.s_off.tolist()
    np.testing.assert_array_equal(result.vertices.cpu().numpy(), expected.vertices)
    np.testing.assert_array_equal(result.radii.cpu().numpy(), expected.radii)

    # A dense (B, Z, Y, X) CUDA tensor of same-shaped samples takes the stacked fast path.
    same_shape = [samples[0], samples[0] * 0 + (samples[0] > 0) * 3]
    dense = torch.as_tensor(np.stack(same_shape), device="cuda")
    dense_result = bt.skeletonize_batch(dense, **options)
    dense_expected = brook.skeletonize_batch(np.stack(same_shape), **options).to_host()
    assert dense_result.labels.tolist() == dense_expected.labels.tolist()

    # padded(): dense CUDA tensors matching BatchedSkeletons.padded(), computed on the device.
    dense_expected_pad = expected.padded()
    pad = bt.padded(result)
    assert pad["vertices"].is_cuda and pad["radii"].is_cuda and pad["labels"].is_cuda and pad["mask"].is_cuda
    np.testing.assert_array_equal(pad["vertices"].cpu().numpy(), dense_expected_pad["vertices"])
    np.testing.assert_array_equal(pad["radii"].cpu().numpy(), dense_expected_pad["radii"])
    np.testing.assert_array_equal(pad["labels"].cpu().numpy(), dense_expected_pad["labels"])
    np.testing.assert_array_equal(pad["mask"].cpu().numpy(), dense_expected_pad["mask"])

    # Explicit bounds give the same masked region without needing a device-to-host sync to size it.
    bounded = bt.padded(result, max_skeletons=pad["labels"].shape[1], max_vertices=pad["vertices"].shape[2])
    np.testing.assert_array_equal(bounded["mask"].cpu().numpy(), dense_expected_pad["mask"])

    # result[i] is a torch view of sample i, matching brook's own per-sample view.
    for i in range(len(result)):
        view = result[i]
        host_view = expected[i]
        np.testing.assert_array_equal(view.vertices.cpu().numpy(), host_view.vertices)
        np.testing.assert_array_equal(view.labels.numpy(), host_view.labels)


@pytest.mark.gpu
def test_gpu_contrib_torch_module_stream_and_grad(gpu):
    torch = pytest.importorskip("torch")
    from brook.contrib import torch as bt

    sample = _batch_samples()[0]
    params = dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4, soma_detection_threshold=6,
                  soma_acceptance_threshold=8, soma_invalidation_scale=1., soma_invalidation_const=0)
    options = dict(teasar_params=params, dust_threshold=10, fix_borders=True, progress=False)
    expected = brook.skeletonize(np.asfortranarray(sample), **options)

    module = bt.Skeletonize(**options)
    assert isinstance(module, torch.nn.Module)

    def check(result):
        assert isinstance(result, bt.PackedSkeletonsT)
        for i, label in enumerate(result.labels.tolist()):
            v0, v1 = int(result.v_off[i]), int(result.v_off[i + 1])
            np.testing.assert_array_equal(result.vertices[v0:v1].cpu().numpy(), expected[label].vertices)

    # A CPU tensor is uploaded automatically; a single 3-D volume takes the skeletonize path;
    # explicit torch.no_grad() around the call is fine (forward already runs under no_grad()).
    with torch.no_grad():
        check(module(torch.as_tensor(sample)))

    # A non-default stream: brook.contrib.torch reads the *current* stream inside the call, so
    # work queued on a side stream is correctly ordered against Brook's own without extra care.
    side_stream = torch.cuda.Stream()
    with torch.cuda.stream(side_stream):
        gpu_tensor = torch.as_tensor(sample, device="cuda")
        stream_result = module(gpu_tensor)
    torch.cuda.synchronize()
    check(stream_result)

    # A list of 3-D tensors takes the skeletonize_batch path.
    batch_result = module([gpu_tensor, gpu_tensor])
    assert isinstance(batch_result, bt.BatchedSkeletonsT) and len(batch_result) == 2

    # A tensor that still requires grad is rejected with a clear, specific error (skeletonization
    # is combinatorial and has no gradient; a silent detach would be a worse footgun).
    grad_input = torch.zeros((8, 8, 8), dtype=torch.float32, device="cuda", requires_grad=True)
    with pytest.raises(RuntimeError, match="requires grad"):
        module(grad_input)
# Voxel connectivity graph bits (cc3d convention) for a step (dx, dy, dz) along the array's own axes.
_GRAPH_BITS = {(1, 0, 0): 0, (-1, 0, 0): 1, (0, 1, 0): 2, (0, -1, 0): 3, (0, 0, 1): 4, (0, 0, -1): 5,
               (1, 1, 0): 6, (-1, 1, 0): 7, (1, -1, 0): 8, (-1, -1, 0): 9,
               (1, 0, 1): 10, (-1, 0, 1): 11, (0, 1, 1): 12, (0, -1, 1): 13,
               (1, 0, -1): 14, (-1, 0, -1): 15, (0, 1, -1): 16, (0, -1, -1): 17,
               (1, 1, 1): 18, (-1, 1, 1): 19, (1, -1, 1): 20, (-1, -1, 1): 21,
               (1, 1, -1): 22, (-1, 1, -1): 23, (1, -1, -1): 24, (-1, -1, -1): 25}


def _voxel_graph(labels, side, region):
    """The 26-connectivity graph of `labels` (a bit per permitted step between equal labels) with
    every edge between `side` and its complement cut where both ends lie in `region`: a self-contact
    the labels alone cannot express."""
    graph = np.zeros(labels.shape, np.uint32)
    foreground = labels != 0
    for (dx, dy, dz), bit in _GRAPH_BITS.items():
        src = tuple(slice(max(0, -d), labels.shape[a] - max(0, d)) for a, d in enumerate((dx, dy, dz)))
        dst = tuple(slice(max(0, d), labels.shape[a] - max(0, -d)) for a, d in enumerate((dx, dy, dz)))
        cut = (side[src] != side[dst]) & region[src] & region[dst]
        permitted = foreground[src] & (labels[src] == labels[dst]) & ~cut
        graph[src] |= permitted.astype(np.uint32) << np.uint32(bit)
    return graph


def _graph_cases():
    """Self-touching volumes with the cut that separates the contact: (labels, side, region)."""
    bars = np.zeros((40, 40, 24), np.uint32)          # two bars of one label touching along a face
    bars[4:36, 8:16, 8:16] = 1
    bars[4:36, 16:24, 8:16] = 1
    side = np.zeros(bars.shape, bool)
    side[:, 16:, :] = True
    everywhere = np.ones(bars.shape, bool)
    partial = np.zeros(bars.shape, bool)                # the same bars, cut only up to x = 30: one component
    partial[:30] = True
    ball = np.zeros((40, 40, 40), np.uint32)            # a soma-sized ball; a process touches its surface
    x, y, z = np.indices(ball.shape)
    ball[(x - 14) ** 2 + (y - 14) ** 2 + (z - 14) ** 2 <= 100] = 1
    ball[12:17, 12:17, 24:38] = 1
    ball[28:32, 4:36, 28:32] = 2
    process = np.zeros(ball.shape, bool)
    process[12:17, 12:17, 24:] = True
    return [(bars, side, everywhere), (bars.astype(np.uint16), side, partial), (ball, process, np.ones(ball.shape, bool))]


_GRAPH_OPTIONS = dict(teasar_params=dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4, soma_detection_threshold=6,
                                         soma_acceptance_threshold=8, soma_invalidation_scale=1., soma_invalidation_const=0),
                      dust_threshold=10, fix_borders=True, progress=False)


def _same_skeletons(result, reference):
    if list(result) != list(reference):
        return False
    for label in reference:
        for name in ["vertices", "edges", "radii"]:
            a, b = getattr(result[label], name), getattr(reference[label], name)
            if a.shape != b.shape or not np.array_equal(a, b):
                return False
    return True


def _assert_batch(batch, expected):
    assert len(batch) == len(expected)
    for i, reference in enumerate(expected):
        result = batch[i].to_skeletons()
        assert list(result) == list(reference), i
        for label in reference:
            for name in ["vertices", "edges", "radii"]:
                np.testing.assert_array_equal(getattr(result[label], name), getattr(reference[label], name))


def _cut_tube(phases=(0.538, 1.488, 5.035, 3.658, 0.591, 2.721)):
    """One label: a wavy tube along x that the voxel graph cuts in two at z = 21, a wavy arm along y
    on each side (both parts span the volume's full x and y extent), an enclosed cavity with a loose
    blob inside the lower part, and a cavity straddling the cut that only the two parts together
    enclose. Returns (labels, side of the cut)."""
    p = phases
    x, y, z = np.indices((40, 36, 44), dtype=np.float64)
    labels = np.zeros(x.shape, np.uint32)
    yc, zc = 17.3 + 2.9 * np.sin(0.17 * x + p[0]), 20.6 + 1.3 * np.cos(0.21 * x + p[1])
    labels[((y - yc) / 6.6) ** 2 + ((z - zc) / 9.5) ** 2 <= 1] = 5
    xa, za = 21.6 + 2.3 * np.sin(0.19 * y + p[2]), 13.2 + 1.4 * np.sin(0.27 * y + p[3])
    labels[((x - xa) / 3.7) ** 2 + ((z - za) / 3.3) ** 2 <= 1] = 5
    xb, zb = 16.4 + 2.1 * np.sin(0.16 * y + p[4]), 29.1 + 1.6 * np.cos(0.23 * y + p[5])
    labels[((x - xb) / 3.4) ** 2 + ((z - zb) / 3.9) ** 2 <= 1] = 5
    cx, cy = 8.6, 17.3 + 2.9 * np.sin(0.17 * 8.6 + p[0])
    cavity = ((x - cx) / 4.1) ** 2 + ((y - cy) / 3.4) ** 2 + ((z - 16.4) / 2.8) ** 2 <= 1
    blob = ((x - cx) / 2.0) ** 2 + ((y - cy) / 1.6) ** 2 + ((z - 16.4) / 1.1) ** 2 <= 1
    sx, sy = 30.2, 17.3 + 2.9 * np.sin(0.17 * 30.2 + p[0])
    labels[cavity & ~blob] = 0
    labels[((x - sx) / 3.1) ** 2 + ((y - sy) / 2.6) ** 2 + ((z - 20.7) / 2.5) ** 2 <= 1] = 0
    return np.asfortranarray(labels), z >= 21


def _open_graph(side):
    """Every step permitted (background and the steps out of the volume included) except across the
    cut: a filled voxel is joined to its neighbours."""
    graph = np.full(side.shape, (1 << 26) - 1, np.uint32)
    for (dx, dy, dz), bit in _GRAPH_BITS.items():
        src = tuple(slice(max(0, -d), side.shape[a] - max(0, d)) for a, d in enumerate((dx, dy, dz)))
        dst = tuple(slice(max(0, d), side.shape[a] - max(0, -d)) for a, d in enumerate((dx, dy, dz)))
        graph[src] &= ~((side[src] != side[dst]).astype(np.uint32) << np.uint32(bit))
    return graph


def _label_graph(labels, side):
    """cc3d.voxel_connectivity_graph(labels, 26) (a step between equal labels, background included,
    the steps out of the volume set) without the steps across the cut: a filled voxel keeps the
    bits of the background it was."""
    graph = np.zeros(labels.shape, np.uint32)
    for (dx, dy, dz), bit in _GRAPH_BITS.items():
        src = tuple(slice(max(0, -d), labels.shape[a] - max(0, d)) for a, d in enumerate((dx, dy, dz)))
        dst = tuple(slice(max(0, d), labels.shape[a] - max(0, -d)) for a, d in enumerate((dx, dy, dz)))
        inside = np.zeros(labels.shape, bool)
        inside[src] = True
        graph[~inside] |= np.uint32(1 << bit)
        graph[src] |= ((labels[src] == labels[dst]) & (side[src] == side[dst])).astype(np.uint32) << np.uint32(bit)
    return graph


def _digest(skeleton):
    """An order-independent exact digest of a skeleton: its (vertex, radius) rows and its edges."""
    rows = np.column_stack([skeleton.vertices, skeleton.radii]).astype(np.float32)
    order = np.lexsort(rows.T[::-1])
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    edges = np.sort(rank[np.asarray(skeleton.edges, np.int64)], axis=1).reshape(-1, 2)
    edges = edges[np.lexsort(edges.T[::-1])]
    return hashlib.sha256(rows[order].tobytes() + edges.tobytes()).hexdigest()[:16]


def _summary(skeletons):
    """{label: (vertices, edges, largest radius, _digest)}; vertices - edges counts the trees."""
    return {int(label): (len(s.vertices), len(s.edges), float(np.max(s.radii)), _digest(s)) for label, s in skeletons.items()}


def _avocado(jitter=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
    """A soma (label 3) clipped by the four side faces, so that it spans the volume's full x and y
    extent, a nucleus of label 8 poking out below it (first in Fortran order: the avocado stage
    merges it into the soma and refreshes the distance field) and a process leaving the soma upwards."""
    j = jitter
    x, y, z = np.indices((30, 28, 44), dtype=np.float64)
    labels = np.zeros(x.shape, np.uint32)
    labels[((x - 15.2 - j[0]) / 16.1) ** 2 + ((y - 13.9 - j[1]) / 15.3) ** 2 + ((z - 22.4 - j[2]) / 12.7) ** 2 <= 1] = 3
    labels[((x - 14.3 - j[3]) / 6.2) ** 2 + ((y - 13.6 - j[4]) / 5.8) ** 2 + ((z - 10.5 - j[5]) / 5.1) ** 2 <= 1] = 8
    px, py = 12.1 + j[6] + 1.7 * np.sin(0.21 * z), 15.4 + j[7] + 1.3 * np.cos(0.17 * z)
    labels[(((x - px) / 2.6) ** 2 + ((y - py) / 2.3) ** 2 <= 1) & (z >= 30)] = 3
    return np.asfortranarray(labels)


def _avocado_order():
    """A soma (label 3) clipped by the four side faces around a nucleus (label 8), a void inside the soma
    that the voxel graph joins to the soma, and five dust specks, first in Fortran order, that take
    graph ids 2-6 (the background is 1). Kimimaro's avocado stage visits the nucleus (graph id 8)
    before the soma (7), merges it and recomputes the field; in compact-id order the soma comes first
    and its fill absorbs the nucleus without a merge. Returns (labels, graph)."""
    x, y, z = np.indices((40, 36, 48), dtype=np.float64)
    labels = np.zeros(x.shape, np.uint32)
    for i in range(5):
        labels[3 + 7 * i:5 + 7 * i, 2:4, 1] = 20 + i
    labels[((x - 19.6) / 22) ** 2 + ((y - 17.8) / 20) ** 2 + ((z - 27.2) / 13.1) ** 2 <= 1] = 3
    labels[((x - 19.3) / 6.1) ** 2 + ((y - 17.6) / 5.7) ** 2 + ((z - 26.8) / 5.4) ** 2 <= 1] = 8
    graph = _label_graph(labels, np.zeros(labels.shape, bool))
    labels[((x - 7.8) / 2.6) ** 2 + ((y - 17.9) / 2.4) ** 2 + ((z - 27.3) / 2.2) ** 2 <= 1] = 0
    return np.asfortranarray(labels), graph


def _soma_voids():
    """A soma-sized component clipped by the four side faces with an enclosed cavity, and the labels'
    26-connectivity graph, under which the filled cavity keeps its boundary. Returns (labels, graph)."""
    x, y, z = np.indices((26, 24, 30), dtype=np.float64)
    ball = ((x - 12.6) / 14.2) ** 2 + ((y - 11.7) / 13.1) ** 2 + ((z - 14.8) / 11.3) ** 2 <= 1
    cavity = ((x - 12.3) / 4.2) ** 2 + ((y - 11.4) / 3.6) ** 2 + ((z - 14.6) / 3.1) ** 2 <= 1
    labels = np.asfortranarray((ball & ~cavity).astype(np.uint32))
    return labels, _label_graph(labels, np.zeros(labels.shape, bool))


def _painted_root():
    """Label 1 has its lowest voxels in label 2's cavity and leaves it through a diagonal step in the
    wall, so filling label 2's holes paints label 1's first voxel; a dust speck (label 3) lies between
    label 1's first voxel before and after the fill."""
    labels = np.zeros((22, 22, 26), np.uint32)
    labels[3:17, 3:17, 3:17] = 2
    labels[5:15, 5:15, 5:15] = 0
    labels[6:13, 6:13, 6:12] = 1
    labels[9, 9, 12:16] = 1
    labels[10, 10, 16:18] = 1
    labels[6:16, 6:16, 18:24] = 1
    labels[19, 19, 13] = 3
    return np.asfortranarray(labels)


def _cut_tube_graphs():
    """The cut tube and its three graphs: open (background joined), cut (the labels' own graph with
    the cut) and labels (cc3d's graph of the labels, background included, with the cut)."""
    labels, side = _cut_tube()
    return labels, {"open": np.asfortranarray(_open_graph(side)),
                    "cut": np.asfortranarray(_voxel_graph(labels, side, np.ones(side.shape, bool))),
                    "labels": np.asfortranarray(_label_graph(labels, side))}


def _spanning_labels():
    """Labels 5 and 9, which the all-steps graph joins into one component: its last run of voxels in
    Fortran order starts in label 5, while its first and last voxels and most of its voxels are label 9.
    Returns (labels, graph)."""
    labels = np.zeros((12, 6, 10), np.uint32)
    labels[1:6, 1:5, 5:9] = 5
    labels[6:11, 1:5, 1:9] = 9
    return np.asfortranarray(labels), np.asfortranarray(_open_graph(np.zeros(labels.shape, bool)))


# The Kimimaro 5.8.1 comparisons below use these teasar_params (with per-geometry soma thresholds).
_HOLE_PARAMS = dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4, soma_detection_threshold=100,
                    soma_acceptance_threshold=120, soma_invalidation_scale=1., soma_invalidation_const=0)
_AVOCADO_JITTER = (0.15, 0.48, 0.33, -0.33, -0.24, 0.45, -0.59, 0.39)


@pytest.mark.gpu
@pytest.mark.parametrize("fix_branching", [True, False])
def test_gpu_voxel_graph_lockstep_identity(gpu, monkeypatch, fix_branching):
    # A voxel graph traces in lockstep (the default) with the result of the per-component tracer
    # (BROOK_LOCKSTEP=0), on cut contacts, a partial cut and a soma with a cut process.
    from brook.device import context

    options = dict(_GRAPH_OPTIONS, fix_branching=fix_branching)
    for labels, side, region in _graph_cases():
        graph = _voxel_graph(labels, side, region)
        labels = np.asfortranarray(labels)
        with_graph = brook.skeletonize(labels, voxel_graph=graph, streaming=False, **options)
        stats = context().stats
        assert stats["lockstep_labels"] >= 1 and stats.get("ejected", 0) == 0, "the graph traced in lockstep"
        assert not _same_skeletons(with_graph, brook.skeletonize(labels, streaming=False, **options)), "the cut changes the skeleton"
        monkeypatch.setenv("BROOK_LOCKSTEP", "0")
        reference = brook.skeletonize(labels, voxel_graph=graph, streaming=False, **options)
        monkeypatch.delenv("BROOK_LOCKSTEP")
        assert "lockstep_labels" not in context().stats
        assert _same_skeletons(with_graph, reference)


@pytest.mark.gpu
def test_gpu_voxel_graph_device_input(gpu):
    # A device-resident volume accepts a voxel graph (device-resident, borrowed, or a host array), also
    # with fill_holes and fix_avocados, and leaves the graph's bytes as given.
    from brook.device import context

    labels, side, region = _graph_cases()[0]
    graph = _voxel_graph(labels, side, region)
    expected = brook.skeletonize(np.asfortranarray(labels), voxel_graph=graph, streaming=False, **_GRAPH_OPTIONS)
    ctx = context()
    device_labels, device_graph = ctx.upload(np.asfortranarray(labels)), ctx.upload(np.asfortranarray(graph))
    assert _same_skeletons(brook.skeletonize(device_labels, voxel_graph=device_graph, **_GRAPH_OPTIONS), expected)
    assert _same_skeletons(brook.skeletonize(device_labels, voxel_graph=graph, **_GRAPH_OPTIONS), expected)
    packed = brook.skeletonize(device_labels, voxel_graph=device_graph, output="packed_device", **_GRAPH_OPTIONS)
    assert packed.on_device and len(packed) == len(expected)
    with pytest.raises(TypeError, match="needs device labels"):                  # host labels, device graph
        brook.skeletonize(np.asfortranarray(labels), voxel_graph=device_graph, **_GRAPH_OPTIONS)
    np.testing.assert_array_equal(device_graph.get(), np.asfortranarray(graph))
    options = dict(teasar_params=dict(_HOLE_PARAMS, soma_detection_threshold=10, soma_acceptance_threshold=11),
                   dust_threshold=10, fill_holes=True, fix_avocados=True, progress=False)
    avocado = _avocado(_AVOCADO_JITTER)
    tube, graphs = _cut_tube_graphs()
    for labels, graph in ((avocado, np.asfortranarray(_label_graph(avocado, np.zeros(avocado.shape, bool)))), (tube, graphs["labels"])):
        expected = brook.skeletonize(labels, voxel_graph=graph, streaming=False, **options)
        assert expected
        device_labels, device_graph = ctx.upload(labels), ctx.upload(graph)
        assert _same_skeletons(brook.skeletonize(device_labels, voxel_graph=device_graph, **options), expected)
        assert _same_skeletons(brook.skeletonize(device_labels, voxel_graph=graph, **options), expected)
        np.testing.assert_array_equal(device_graph.get(), graph)


@pytest.mark.gpu
@pytest.mark.parametrize("fix_branching", [True, False])
def test_gpu_batch_voxel_graph(gpu, monkeypatch, fix_branching):
    # Per-sample voxel graphs in a batch of mixed shapes (the generic path): host and device graphs,
    # a sample without a graph among graph samples, and forced sub-batching all match the per-sample calls.
    from brook.device import context

    options = dict(_GRAPH_OPTIONS, fix_branching=fix_branching)
    cases = _graph_cases()
    samples = [case[0] for case in cases]
    graphs = [_voxel_graph(*case) for case in cases]
    expected = [brook.skeletonize(np.asfortranarray(s), voxel_graph=g, streaming=False, **options) for s, g in zip(samples, graphs)]
    batch = brook.skeletonize_batch(samples, voxel_graph=graphs, **options)
    assert context().stats["batch_sub_batches"] == 1 and context().stats.get("batch_uniform", 0) == 0
    _assert_batch(batch, expected)
    ctx = context()
    device_samples = [ctx.upload(np.asfortranarray(s)) for s in samples]
    device_graphs = [ctx.upload(np.asfortranarray(g)) for g in graphs]
    _assert_batch(brook.skeletonize_batch(device_samples, voxel_graph=device_graphs, **options), expected)
    plain = brook.skeletonize(np.asfortranarray(samples[1]), streaming=False, **options)
    _assert_batch(brook.skeletonize_batch(samples, voxel_graph=[graphs[0], None, graphs[2]], **options), [expected[0], plain, expected[2]])
    with pytest.raises(ValueError):
        brook.skeletonize_batch(samples, voxel_graph=graphs[:2], **options)
    monkeypatch.setenv("BROOK_BATCH_BUDGET", "1")
    split = brook.skeletonize_batch(samples, voxel_graph=graphs, **options)
    assert context().stats["batch_sub_batches"] == len(samples)
    _assert_batch(split, expected)


@pytest.mark.gpu
def test_gpu_batch_uniform_voxel_graph(gpu, monkeypatch):
    # Samples of one shape with voxel graphs take the stacked path: the graphs stack with the samples
    # (seam bits masked), one graph CCL, one segmented graph EDT, one trace. Same results as the
    # per-sample calls, the generic path and forced sub-batching; dense (B, Z, Y, X) graphs too.
    from brook.device import context

    canvas = (44, 42, 46)
    samples, graphs = [], []
    for labels, side, region in _graph_cases():
        sample, sample_side, sample_region = np.zeros(canvas, np.int32), np.zeros(canvas, bool), np.ones(canvas, bool)
        x, y, z = labels.shape
        sample[2:2 + x, 1:1 + y, 3:3 + z] = labels
        sample_side[2:2 + x, 1:1 + y, 3:3 + z] = side
        sample_region[2:2 + x, 1:1 + y, 3:3 + z] = region
        sample[40:42, 36:40, 40:44] = 7                     # below the dust threshold
        samples.append(sample)
        graphs.append(_voxel_graph(sample, sample_side, sample_region))
    samples.append(samples[0].copy())                       # identical labels in two samples: ids must not collide
    graphs.append(graphs[0].copy())
    # every step permitted, the steps out of the volume included: unmasked, the bits across a seam
    # would join consecutive samples, and the interface behind a sample's last slice must stay open
    samples.append(samples[1].copy())
    graphs.append(np.where(samples[1] != 0, np.uint32((1 << 26) - 1), np.uint32(0)))
    options = dict(_GRAPH_OPTIONS, dust_threshold=100)
    expected = [brook.skeletonize(np.asfortranarray(s), voxel_graph=g, streaming=False, **options) for s, g in zip(samples, graphs)]
    assert all(expected) and not _same_skeletons(expected[0], brook.skeletonize(np.asfortranarray(samples[0]), streaming=False, **options))
    batch = brook.skeletonize_batch(samples, voxel_graph=graphs, **options)
    assert context().stats["batch_uniform"] == 1
    _assert_batch(batch, expected)
    dense = brook.skeletonize_batch(np.stack(samples), voxel_graph=np.stack(graphs), output="packed", **options)
    assert context().stats["batch_uniform"] == 1
    host = batch.to_host()
    assert dense.labels.tolist() == host.labels.tolist()
    for name in ["vertices", "edges", "radii", "v_off", "e_off", "s_off"]:
        np.testing.assert_array_equal(getattr(dense, name), getattr(host, name))
    ctx = context()
    device = brook.skeletonize_batch([ctx.upload(np.asfortranarray(s)) for s in samples],
                                     voxel_graph=[ctx.upload(np.asfortranarray(g)) for g in graphs], **options)
    assert context().stats["batch_uniform"] == 1
    _assert_batch(device, expected)
    monkeypatch.setenv("BROOK_BATCH_BUDGET", "1")           # one stacked sub-batch per sample
    _assert_batch(brook.skeletonize_batch(samples, voxel_graph=graphs, **options), expected)
    assert context().stats["batch_uniform"] == 1 and context().stats["batch_sub_batches"] == len(samples)
    monkeypatch.delenv("BROOK_BATCH_BUDGET")
    monkeypatch.setenv("BROOK_BATCH_UNIFORM", "0")           # the generic path gives the same batch
    generic = brook.skeletonize_batch(samples, voxel_graph=graphs, output="packed", **options)
    assert context().stats.get("batch_uniform", 0) == 0
    assert generic.labels.tolist() == host.labels.tolist()
    for name in ["vertices", "edges", "radii", "v_off", "e_off", "s_off"]:
        np.testing.assert_array_equal(getattr(generic, name), getattr(host, name))


@pytest.mark.gpu
@pytest.mark.parametrize("fix_branching", [False, True])
def test_gpu_voxel_graph_fill_holes(gpu, monkeypatch, fix_branching):
    # fill_holes with a voxel graph fills each graph component's holes and uses the graph as given, as
    # Kimimaro 5.8.1 does. Values from Kimimaro on the Fortran-ordered graphs (both parts of the tube span
    # the full x and y extent): digests where no tie of the path search selects other vertices, else the
    # two trees and the largest radius. Under the cut and label graphs no rail reaches the lower part's
    # filled cavity: the lockstep ejects that component and its trace stops at the cavity.
    from brook.device import context

    ctx = context()
    labels, graphs = _cut_tube_graphs()
    filled = {"open": 146, "cut": 163, "labels": 163}
    radius = {"open": (5.220153331756592, 6.164999961853027), "cut": (4.924428939819336, 6.112677097320557),
              "labels": (4.924428939819336, 6.112677097320557)}
    digests = {("open", (1., 1., 1.)): (177, 175, 5.220153331756592, "79d57c10fa68bdb2"),
               ("open", (1., 1.13, 1.37)): (219, 217, 6.164999961853027, "7282c1e0c830083c"),
               ("cut", (1., 1.13, 1.37)): (376, 374, 6.112677097320557, "efa10b05120eddc4"),
               ("labels", (1., 1.13, 1.37)): (315, 313, 6.112677097320557, "f09248b04e476d32")}
    for name, graph in graphs.items():
        unreachable = int(fix_branching and name != "open")
        for i, aniso in enumerate([(1., 1., 1.), (1., 1.13, 1.37)]):
            options = dict(teasar_params=_HOLE_PARAMS, anisotropy=aniso, dust_threshold=10, fix_branching=fix_branching,
                           progress=False)
            result = brook.skeletonize(labels, voxel_graph=graph, fill_holes=True, streaming=False, **options)
            stats = context().stats
            assert stats["hole_filled_voxels"] == filled[name]
            assert stats["ejected"] == unreachable and stats.get("component_unreachable_stops", 0) == unreachable
            summary = _summary(result)
            assert list(summary) == [5] and summary[5][0] - summary[5][1] == 2 and summary[5][2] == radius[name][i]
            if not fix_branching and (name, aniso) in digests:
                assert summary[5] == digests[name, aniso]
            assert not _same_skeletons(result, brook.skeletonize(labels, voxel_graph=graph, streaming=False, **options))
            assert _same_skeletons(brook.skeletonize(ctx.upload(labels), voxel_graph=ctx.upload(graph), fill_holes=True, **options), result)
            monkeypatch.setenv("BROOK_LOCKSTEP", "0")
            assert _same_skeletons(brook.skeletonize(labels, voxel_graph=graph, fill_holes=True, streaming=False, **options), result)
            assert context().stats.get("component_unreachable_stops", 0) == unreachable
            monkeypatch.delenv("BROOK_LOCKSTEP")
    cc, _, _ = ctx.connected_components(ctx.upload(labels))
    assert ctx._fill_all_holes(cc)[1] == 247                           # without a graph


@pytest.mark.gpu
def test_gpu_voxel_graph_avocados(gpu, monkeypatch):
    # With a voxel graph the avocado stage recomputes the graph EDT after a merge, as Kimimaro 5.8.1 does
    # (values from Kimimaro; the soma spans the full x and y extent). The label graph cuts the merged
    # nucleus off from the soma, so with branching correction no rail reaches it.
    from brook.device import context

    ctx = context()
    labels = _avocado(_AVOCADO_JITTER)
    graph = np.asfortranarray(_label_graph(labels, np.zeros(labels.shape, bool)))
    options = dict(teasar_params=dict(_HOLE_PARAMS, soma_detection_threshold=10, soma_acceptance_threshold=11),
                   dust_threshold=10, progress=False)
    soma = (316, 315, 9.5, "922c9a29f61e3a8e")
    plain = brook.skeletonize(labels, voxel_graph=graph, fix_branching=False, streaming=False, **options)
    assert _summary(plain) == {8: (9, 8, 4.609772205352783, "6048950761de63d6"), 3: soma}
    device_graph = ctx.upload(graph)
    for fill_holes in (False, True):
        options.update(fill_holes=fill_holes, fix_avocados=True)
        corrected = brook.skeletonize(labels, voxel_graph=graph, fix_branching=False, streaming=False, **options)
        assert _summary(corrected) == {3: soma} and context().stats["avocado_passes"] == 2
        branched = brook.skeletonize(labels, voxel_graph=graph, streaming=False, **options)
        stats = context().stats
        assert stats["ejected"] == 1 and stats["component_unreachable_stops"] == 1
        summary = _summary(branched)
        assert list(summary) == [3] and summary[3][0] - summary[3][1] == 1 and summary[3][2] == 9.5
        monkeypatch.setenv("BROOK_LOCKSTEP", "0")
        assert _same_skeletons(brook.skeletonize(labels, voxel_graph=graph, streaming=False, **options), branched)
        monkeypatch.delenv("BROOK_LOCKSTEP")
        assert _same_skeletons(brook.skeletonize(ctx.upload(labels), voxel_graph=device_graph, **options), branched)
    np.testing.assert_array_equal(device_graph.get(), graph)


@pytest.mark.gpu
def test_gpu_voxel_graph_avocado_order(gpu, monkeypatch):
    # Kimimaro keeps cc3d.color_connectivity_graph's ids through its avocado stage and visits candidates in
    # the order of a Python set of those ids: here the nucleus (graph id 8) before the soma (7), so the
    # nucleus merges and the field is recomputed; in compact-id order the soma's fill absorbs the nucleus
    # without a merge (580 vertices). Values from Kimimaro 5.8.1 (the soma spans the full x and y extent,
    # and neither argmax ties).
    from brook.device import context

    ctx = context()
    labels, graph = _avocado_order()
    graph, aniso = np.asfortranarray(graph), (1., 1.13, 1.37)
    options = dict(teasar_params=dict(_HOLE_PARAMS, soma_detection_threshold=10, soma_acceptance_threshold=1000),
                   anisotropy=aniso, dust_threshold=10, progress=False, streaming=False)
    plain = _summary(brook.skeletonize(labels, voxel_graph=graph, fix_branching=False, **options))
    assert plain[8] == (11, 10, 5.464176177978516, "9693fc7b4f82a49c") and plain[3][:3] == (580, 579, 9.179778099060059)
    corrected = brook.skeletonize(labels, voxel_graph=graph, fix_avocados=True, fix_branching=False, **options)
    assert _summary(corrected) == {3: (573, 572, 9.179778099060059, "74eed02464c34728")}
    assert context().stats["avocado_passes"] == 2
    branched = brook.skeletonize(labels, voxel_graph=graph, fix_avocados=True, **options)
    stats = context().stats
    assert stats["ejected"] == 1 and stats["component_unreachable_stops"] == 1       # the merged nucleus
    summary = _summary(branched)
    assert list(summary) == [3] and summary[3][0] - summary[3][1] == 1 and summary[3][2] == 8.973200798034668
    monkeypatch.setenv("BROOK_LOCKSTEP", "0")
    assert _same_skeletons(brook.skeletonize(labels, voxel_graph=graph, fix_avocados=True, **options), branched)
    monkeypatch.delenv("BROOK_LOCKSTEP")
    # The stage: graph ids 2-6 are the specks (the background is 1), 7 the soma and 8 the nucleus.
    device_graph = ctx.upload(graph)
    keys = ctx.graph_components(ctx.upload(labels), device_graph, return_keys=True)[3]
    assert keys.tolist() == [0, 2, 3, 4, 5, 6, 7, 8]
    for given, passes in ((keys, 2), ([], 1)):
        cc, mapping, roots = ctx.graph_components(ctx.upload(labels), device_graph)
        dbf = ctx.graph_edt(cc, device_graph, aniso, False)
        ctx._fix_avocados(cc, dbf, mapping, 10., aniso, False, roots, voxel_graph=device_graph, keys=given)
        assert ctx.stats["avocado_passes"] == passes


@pytest.mark.gpu
@pytest.mark.parametrize("fix_branching", [False, True])
def test_gpu_voxel_graph_soma_voids(gpu, monkeypatch, fix_branching):
    # The tracer's soma fill recomputes the distance field with the voxel graph, as Kimimaro 5.8.1 does
    # (filled voxels keep their graph bits), so the soma root lies on voxels the paths reach. Kimimaro gives
    # one tree with these largest radii without branching correction; with it Brook keeps one tree around
    # the soma root (its soma rule), and no rail reaches the filled cavity.
    from brook.device import context

    labels, graph = _soma_voids()
    graph = np.asfortranarray(graph)
    params = dict(_HOLE_PARAMS, soma_detection_threshold=3, soma_acceptance_threshold=4)
    for aniso, radius in (((1., 1., 1.), 5.7662811279296875), ((1., 1.13, 1.37), 6.881685733795166)):
        options = dict(teasar_params=params, anisotropy=aniso, dust_threshold=10, fix_branching=fix_branching,
                       progress=False, streaming=False)
        result = brook.skeletonize(labels, voxel_graph=graph, **options)
        stats = context().stats
        summary = _summary(result)
        assert list(summary) == [1] and summary[1][0] - summary[1][1] == 1 and summary[1][2] == radius
        assert stats["lockstep_labels"] == 1 and stats["ejected"] == fix_branching
        assert stats.get("component_unreachable_stops", 0) == fix_branching
        monkeypatch.setenv("BROOK_LOCKSTEP", "0")
        assert _same_skeletons(brook.skeletonize(labels, voxel_graph=graph, **options), result)
        monkeypatch.delenv("BROOK_LOCKSTEP")


@pytest.mark.gpu
@pytest.mark.parametrize("fix_branching", [False, True])
def test_gpu_public_trace_voxel_graph(gpu, fix_branching):
    # brook.trace.trace given the soma-void component as Kimimaro's skeletonize passes it to
    # kimimaro.trace.trace (the crop's mask, graph EDT and graph) recomputes the graph EDT after the soma
    # fill and, with branching correction, stops at the unreachable target: it returns skeletonize's
    # skeleton (with fix_borders=False, as trace adds no border targets).
    from brook.device import context
    from brook.trace import trace

    ctx = context()
    labels, graph = _soma_voids()
    graph = np.asfortranarray(graph)
    params = dict(_HOLE_PARAMS, soma_detection_threshold=3, soma_acceptance_threshold=4)
    points = np.argwhere(labels)
    low = points.min(0)
    crop = tuple(slice(a, b + 1) for a, b in zip(low, points.max(0)))
    mask = labels[crop] != 0
    for aniso in ((1., 1., 1.), (1., 1.13, 1.37)):
        expected = brook.skeletonize(labels, teasar_params=params, anisotropy=aniso, dust_threshold=10, fix_borders=False,
                                     fix_branching=fix_branching, voxel_graph=graph, progress=False, streaming=False)[1]
        cc, _, _ = ctx.graph_components(ctx.upload(labels), ctx.upload(graph))
        dbf = np.asfortranarray(ctx.graph_edt(cc, ctx.upload(graph), aniso, False).get()[crop] * mask)
        result = trace(mask, dbf, anisotropy=aniso, fix_branching=fix_branching, voxel_graph=np.asfortranarray(graph[crop]),
                       **params)
        np.testing.assert_array_equal(np.multiply(result.vertices + low, aniso, dtype=np.float32), expected.vertices)
        np.testing.assert_array_equal(result.edges, expected.edges)
        np.testing.assert_array_equal(result.radii, expected.radii)


@pytest.mark.gpu
@pytest.mark.parametrize("fix_branching", [True, False])
def test_gpu_voxel_graph_component_spanning_labels(gpu, monkeypatch, fix_branching):
    # A graph component that spans two labels takes the label at the start of its last run in Fortran
    # order, as Kimimaro's get_mapping does, on host and device input of either layout, both tracing paths
    # and both batch paths. Digest from Kimimaro 5.8.1, with and without fill_holes and fix_avocados.
    from brook.device import context

    ctx = context()
    labels, graph = _spanning_labels()
    expected = {5: (12, 11, 2.0, "e4fa75811ac3b077")}
    for lockstep in ("1", "0"):
        monkeypatch.setenv("BROOK_LOCKSTEP", lockstep)
        for extra in (dict(), dict(fill_holes=True, fix_avocados=True)):
            options = dict(teasar_params=_HOLE_PARAMS, dust_threshold=10, fix_branching=fix_branching, progress=False,
                           streaming=False, **extra)
            for volume, g in ((labels, graph), (np.ascontiguousarray(labels), np.ascontiguousarray(graph)),
                              (ctx.upload(labels), ctx.upload(graph)),
                              (ctx.upload_tensor(np.ascontiguousarray(labels)), graph)):
                assert _summary(brook.skeletonize(volume, voxel_graph=g, **options)) == expected
        options = dict(teasar_params=_HOLE_PARAMS, dust_threshold=10, fix_branching=fix_branching, fill_holes=True)
        for uniform in ("1", "0"):
            monkeypatch.setenv("BROOK_BATCH_UNIFORM", uniform)
            batch = brook.skeletonize_batch([labels, labels], voxel_graph=[graph, graph], output="skeletons", **options)
            assert [_summary(result) for result in batch] == [expected, expected]
            assert ctx.stats.get("batch_uniform", 0) == (uniform == lockstep == "1")
    monkeypatch.delenv("BROOK_LOCKSTEP")
    monkeypatch.delenv("BROOK_BATCH_UNIFORM")


@pytest.mark.gpu
@pytest.mark.parametrize("fix_branching", [False, True])
def test_gpu_fill_holes_painted_root(gpu, monkeypatch, fix_branching):
    # Filling label 2's cavity paints label 1's first voxel: label 1 is traced from its first remaining voxel
    # (Kimimaro's first_label) on both tracing paths, with and without a voxel graph, and a batch of such
    # samples takes the stacked path although the refreshed roots are no longer sorted (the speck, label 3,
    # starts between label 1's old and new first voxels). Values from Kimimaro 5.8.1; label 2's route under
    # branching correction and the graph case are compared by trees and radii (the tie rule; see
    # test_gpu_voxel_graph_fill_holes for the graph crop).
    from brook.device import context

    labels = _painted_root()
    graph = np.asfortranarray(_label_graph(labels, np.zeros(labels.shape, bool)))
    options = dict(teasar_params=_HOLE_PARAMS, dust_threshold=10, fix_borders=False, fix_branching=fix_branching,
                   fill_holes=True, progress=False)
    expected = ({2: (38, 37, 7.0, "8850fde826c023e3"), 1: (24, 23, 3.1622776985168457, "bd25239c8e51a899")}
                if not fix_branching else {2: (35, 34, 7.0), 1: (20, 19, 3.1622776985168457, "13904653524930cd")})
    for lockstep in ("1", "0"):
        monkeypatch.setenv("BROOK_LOCKSTEP", lockstep)
        for fix_avocados in (False, True):                             # no avocado candidate exceeds 100 / 2.5
            summary = _summary(brook.skeletonize(labels, fix_avocados=fix_avocados, streaming=False, **options))
            assert context().stats["hole_filled_voxels"] == 1001
            assert list(summary) == [2, 1] and all(summary[label][:len(v)] == v for label, v in expected.items())
            summary = _summary(brook.skeletonize(labels, voxel_graph=graph, fix_avocados=fix_avocados, streaming=False, **options))
            assert context().stats["hole_filled_voxels"] == 1001
            assert list(summary) == [2, 1] and all(v[0] - v[1] == 1 for v in summary.values())
            assert {label: v[2] for label, v in summary.items()} == {2: 1.5, 1: 2.5495097637176514}
    monkeypatch.delenv("BROOK_LOCKSTEP")
    # Kimimaro 5.8.1 files extra targets under component ids before its avocado stage renumbers the
    # components by first voxel. The fill moved label 1's first voxel past the speck's, so the two swap ids,
    # and with fix_avocados Kimimaro looks label 1's target up under the speck and drops it (with branching
    # correction label 1 has 20 vertices instead of 23). Brook files the target under label 1 after the stage.
    target = [(11, 15, 18)]
    with_target = brook.skeletonize(labels, extra_targets_after=target, streaming=False, **options)
    assert (with_target[1].vertices == np.float32(target)).all(1).any()
    assert _same_skeletons(brook.skeletonize(labels, extra_targets_after=target, fix_avocados=True, streaming=False,
                                             **options), with_target)
    single = brook.skeletonize(labels, fix_avocados=True, streaming=False, **options)
    batch = brook.skeletonize_batch([labels, labels], fix_avocados=True, output="skeletons", **options)
    assert context().stats["batch_uniform"] == 1
    for i, result in enumerate(batch):
        _assert_same_skeletons(result, single, f"sample {i}")


@pytest.mark.gpu
def test_gpu_avocado_stage_voxel_graph(gpu):
    # The avocado stage with a voxel graph and Kimimaro's graph ids (test_gpu_voxel_graph_avocado_order):
    # a merge recomputes the graph EDT; a stack takes neither. On a stack whose fill moved a root past
    # another component's, the ids of each sample stay together: the stage equals the per-sample stage.
    from brook.device import context

    ctx = context()
    labels, aniso = _avocado(_AVOCADO_JITTER), (1., 1.13, 1.37)
    graph = ctx.upload(np.asfortranarray(_label_graph(labels, np.zeros(labels.shape, bool))))
    cc, mapping, roots, keys = ctx.graph_components(ctx.upload(labels), graph, return_keys=True)
    assert (mapping.tolist(), roots.tolist(), keys.tolist()) == ([0, 8, 3], [5414, 9590], [0, 2, 3])
    dbf = ctx.graph_edt(cc, graph, aniso, False)
    with pytest.raises(ValueError, match="keys"):
        ctx._fix_avocados(cc, dbf, mapping, 10., aniso, False, roots, voxel_graph=graph, keys=[0, 3, 2])
    corrected, distance, mapping, roots = ctx._fix_avocados(cc, dbf, mapping, 10., aniso, False, roots, voxel_graph=graph, keys=keys)
    assert ctx.stats["avocado_passes"] == 2 and (mapping, roots) == ([0, 3], [5414])
    distance = distance.get()
    np.testing.assert_array_equal(distance, ctx.graph_edt(corrected, graph, aniso, False).get())
    assert not np.array_equal(distance, ctx.edt(corrected, aniso, False).get())
    depth = labels.shape[2]
    cc, mapping, roots = ctx.connected_components(ctx.upload(np.asfortranarray(np.concatenate([labels, labels], axis=2))))
    dbf = ctx.edt(cc, aniso, False, 3, depth)
    with pytest.raises(ValueError, match="one sample"):
        ctx._fix_avocados(cc, dbf, mapping, 10., aniso, False, roots, depth, voxel_graph=graph)
    with pytest.raises(ValueError, match="no voxel graph"):
        ctx._fix_avocados(cc, dbf, mapping, 10., aniso, False, roots, depth, keys=list(range(len(mapping))))
    samples = [_painted_root(), np.where(_painted_root() > 0, _painted_root() + 3, 0).astype(np.uint32)]
    depth, voxels = samples[0].shape[2], samples[0].size
    per_sample = []
    for s in samples:
        cc, mapping, roots = ctx.connected_components(ctx.upload(np.asfortranarray(s)))
        filled, _ = ctx._fill_all_holes(cc)
        per_sample.append(ctx._fix_avocados(filled, ctx.edt(filled, aniso, False), mapping, 6., aniso, False))
    cc, mapping, roots = ctx.connected_components(ctx.upload(np.asfortranarray(np.concatenate(samples, axis=2))))
    filled, _ = ctx._fill_all_holes(cc)
    flat = filled.get().ravel(order="F")
    roots = [int(np.flatnonzero(flat == i)[0]) for i in range(1, len(mapping))]    # as refresh_roots leaves them
    assert roots == [1521, 7974, 6729, 1521 + voxels, 7974 + voxels, 6729 + voxels]
    labels, distance, mapping, roots = ctx._fix_avocados(filled, ctx.edt(filled, aniso, False, 3, depth), mapping, 6., aniso,
                                                         False, roots, depth)
    labels, distance = labels.get(), distance.get()
    first = 0
    for i, (local, field, m, r) in enumerate(per_sample):
        local, field, count = local.get(), field.get(), len(r)
        z = slice(i * depth, (i + 1) * depth)
        np.testing.assert_array_equal(labels[:, :, z], np.where(local > 0, local + first, 0))
        np.testing.assert_array_equal(distance[:, :, z], field)
        assert mapping[first + 1:first + count + 1] == m[1:]
        assert roots[first:first + count] == [v + i * voxels for v in r]
        first += count
    assert first == len(roots) == len(mapping) - 1


def _graph_batch_samples():
    """Samples of one shape with voxel graphs: the cut tube under its three graphs, the avocado and the
    painted root under their labels' graphs, two blocks under an open graph cut along x (the right
    block's component is numbered second but starts at a lower z, so the roots are not sorted), and a
    duplicate of the first sample."""
    tube, graphs = _cut_tube_graphs()
    samples, sample_graphs = [tube] * 3, [graphs["open"], graphs["cut"], graphs["labels"]]
    none = np.zeros(tube.shape, bool)
    for part, origin in ((_avocado(_AVOCADO_JITTER), (4, 3, 0)), (_painted_root(), (9, 7, 10))):
        sample = np.zeros(tube.shape, np.uint32)
        sample[tuple(slice(o, o + n) for o, n in zip(origin, part.shape))] = part
        samples.append(np.asfortranarray(sample))
        sample_graphs.append(np.asfortranarray(_label_graph(sample, none)))
    blocks, side = np.zeros(tube.shape, np.uint32), np.zeros(tube.shape, bool)
    blocks[4:11, 5:11, 25:31] = 7
    blocks[20:28, 5:11, 3:9] = 4
    side[16:] = True
    samples.append(np.asfortranarray(blocks))
    sample_graphs.append(np.asfortranarray(_open_graph(side)))
    return samples + [tube], sample_graphs + [graphs["open"]]


@pytest.mark.gpu
@pytest.mark.parametrize("fix_branching", [True, False])
@pytest.mark.parametrize("fill_holes,fix_avocados", [(True, False), (False, True), (True, True)])
def test_gpu_batch_voxel_graph_holes_and_avocados(gpu, monkeypatch, fill_holes, fix_avocados, fix_branching):
    # Batches with voxel graphs and fill_holes take the stacked path (holes filled on the stacked graph
    # components); with fix_avocados they take the generic path, because Kimimaro orders avocado candidates
    # by each volume's own graph ids. Every path gives the per-sample results.
    from brook.device import context

    samples, graphs = _graph_batch_samples()
    options = dict(teasar_params=dict(_HOLE_PARAMS, soma_detection_threshold=10, soma_acceptance_threshold=11),
                   dust_threshold=10, fix_branching=fix_branching, fill_holes=fill_holes, fix_avocados=fix_avocados,
                   progress=False)
    expected, filled = [], 0
    for sample, graph in zip(samples, graphs):
        expected.append(brook.skeletonize(sample, voxel_graph=graph, streaming=False, **options))
        filled += context().stats.get("hole_filled_voxels", 0)
    assert all(expected)
    stacked = int(not fix_avocados)
    batch = brook.skeletonize_batch(samples, voxel_graph=graphs, **options)
    stats = context().stats
    assert stats.get("batch_uniform", 0) == stacked
    if stacked:
        assert stats["hole_filled_voxels"] == filled                  # one stacked call: the fill of every sample
    _assert_batch(batch, expected)
    ctx = context()
    _assert_batch(brook.skeletonize_batch([ctx.upload(s) for s in samples], voxel_graph=[ctx.upload(g) for g in graphs],
                                          **options), expected)
    assert context().stats.get("batch_uniform", 0) == stacked
    _assert_batch(brook.skeletonize_batch(np.stack(samples), voxel_graph=np.stack(graphs), **options), expected)
    assert context().stats.get("batch_uniform", 0) == stacked
    monkeypatch.setenv("BROOK_BATCH_BUDGET", "1")                     # one sample per sub-batch
    _assert_batch(brook.skeletonize_batch(samples, voxel_graph=graphs, **options), expected)
    assert context().stats.get("batch_uniform", 0) == stacked and context().stats["batch_sub_batches"] == len(samples)
    monkeypatch.delenv("BROOK_BATCH_BUDGET")
    monkeypatch.setenv("BROOK_BATCH_UNIFORM", "0")                    # the generic path
    _assert_batch(brook.skeletonize_batch(samples, voxel_graph=graphs, **options), expected)
    assert context().stats.get("batch_uniform", 0) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("fix_branching", [True, False])
def test_gpu_batch_stacked_border_order(gpu, fix_branching):
    # The stacked path collects a component's border targets in its own sample's frame: their order (a
    # Python set's, which depends on the coordinates), and so the root and the order of the targets, are
    # those of a separate call for every sample. The cut tube's label touches four faces (six targets).
    from brook.device import context

    tube, _ = _cut_tube()
    options = dict(teasar_params=_HOLE_PARAMS, dust_threshold=10, fix_branching=fix_branching, progress=False)
    expected = brook.skeletonize(tube, streaming=False, **options)
    batch = brook.skeletonize_batch([tube] * 3, output="skeletons", **options)
    assert context().stats["batch_uniform"] == 1
    for i, result in enumerate(batch):
        _assert_same_skeletons(result, expected, f"sample {i}")


def _thin_slabs(shape, count, seed):
    """Label volumes one voxel thick along y or x, like the vertical slabs a surface-tracking
    pipeline cuts from a scroll volume: wavy strips across z and along the long axis, overlapping,
    so that most components touch the two faces of the thin axis and a z face or a long-axis end,
    and collect several border targets."""
    rng = np.random.default_rng(seed)
    nz, n = shape[0], max(shape[1], shape[2])
    z, s = np.arange(nz)[:, None], np.arange(n)[None, :]
    samples = []
    for _ in range(count):
        plane = np.zeros((nz, n), np.int32)
        for label in range(1, 41):
            amplitude, period, phase, width = rng.uniform(2, 25), rng.uniform(30, 250), rng.uniform(0, 2 * np.pi), rng.uniform(1.5, 4)
            if rng.random() < 0.6:                            # across z, wavy along the long axis
                plane[np.abs(s - (rng.uniform(0, n) + amplitude * np.sin(2 * np.pi * z / period + phase))) < width] = label
            else:                                             # along the long axis over part of the slab
                a, b = sorted(rng.integers(0, n, 2))
                along = np.abs(z - (rng.uniform(0, nz) + amplitude * np.sin(2 * np.pi * s / period + phase))) < width
                plane[along & (s >= a) & (s <= b)] = label
        samples.append(np.asfortranarray(plane.reshape(shape)))
    return samples


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(100, 1, 2043), (100, 2043, 1)])
def test_gpu_batch_uniform_thin_slab_borders(gpu, shape):
    # Stacked samples must yield the border targets, in the order, of a call on each sample alone:
    # the order decides the trace root. Slabs one voxel thick give most components several targets.
    from brook.device import context

    ctx = context()
    options = dict(teasar_params=dict(scale=1., const=2.), anisotropy=(4, 4, 4), dust_threshold=40,
                   fix_branching=True, fix_borders=True, fill_holes=False, progress=False)
    samples = _thin_slabs(shape, 6, seed=sum(shape))
    cc, mapping, roots = ctx.connected_components(ctx.upload(samples[1]))
    assert sum(len(b) > 2 for b in ctx.border_targets(cc, len(roots), (4., 4., 4.))) > 10
    expected = [brook.skeletonize(s, **options) for s in samples]
    batch = brook.skeletonize_batch(samples, **options)
    assert context().stats["batch_uniform"] == 1
    for i, reference in enumerate(expected):
        result = batch[i].to_skeletons()
        assert list(result) == list(reference), i
        for label in reference:
            for name in ["vertices", "edges", "radii"]:
                np.testing.assert_array_equal(getattr(result[label], name), getattr(reference[label], name), err_msg=f"{i} {label}")


def _sine_band(amplitude, period, width, phase=0.0, axis=0):
    """A band |y - (32 + amplitude sin(2 pi x / period + phase))| < width of a 64 x 160 plane, one voxel
    thick along `axis`."""
    y, x = np.mgrid[:64, :160]
    band = np.abs(y - (32 + amplitude * np.sin(2 * np.pi * x / period + phase))) < width
    return np.moveaxis(band[None].astype(np.uint32), 0, axis)


def _sine_bands():
    """Three bands of labels 1 to 3 in one 64 x 160 plane, each painted where the earlier ones left 0."""
    y, x = np.mgrid[:64, :160]
    plane = np.zeros((64, 160), np.uint32)
    for label, (centre, amplitude, period, width) in enumerate([(12, 4, 50, 1.7), (30, 6, 70, 2.2), (48, 5, 45, 1.5)], 1):
        plane[(np.abs(y - (centre + amplitude * np.sin(2 * np.pi * x / period))) < width) & (plane == 0)] = label
    return plane[None]


def _spiral_band():
    """A one-voxel-thick spiral band in a 96 x 96 plane (turns 11 voxels apart)."""
    y, x = np.mgrid[:96, :96]
    r, angle = np.hypot(y - 47.3, x - 48.6), np.mod(np.arctan2(y - 47.3, x - 48.6), 2 * np.pi)
    turn = (r - 6 - 5.5 * angle / np.pi) / 11.0
    return ((np.abs(turn - np.round(turn)) * 11.0 < 1.8) & (r > 5) & (r < 44))[None].astype(np.uint32)


def _helix_tube(radius, turn_radius, pitch, length):
    """A tube of the given radius around a helix along x (its joins take corner steps)."""
    n = int(2 * turn_radius + 2 * radius + 6)
    x, y, z = np.arange(length, dtype=np.float64)[:, None, None], np.arange(n, dtype=np.float64)[None, :, None], np.arange(n, dtype=np.float64)[None, None, :]
    cy, cz = n / 2 + turn_radius * np.cos(2 * np.pi * x / pitch), n / 2 + turn_radius * np.sin(2 * np.pi * x / pitch)
    return ((y - cy) ** 2 + (z - cz) ** 2 <= radius * radius).astype(np.uint32)


_JOIN_OPTIONS = dict(teasar_params=dict(scale=1., const=2.), anisotropy=(4, 4, 4), dust_threshold=0,
                     fix_branching=True, fix_borders=True, fill_holes=False, progress=False)
_JOIN_MODES = [{}, {"BROOK_LOCKSTEP": "0"}, {"BROOK_BODY": "seq"}, {"BROOK_BODY": "pipe"},
               {"BROOK_BODY": "k", "BROOK_KDRAFT": "eager"}, {"BROOK_GRAPHS": "0"}]


def _assert_joins_follow_kimimaro(monkeypatch, labels, expected, **options):
    # Every tracing path: the component tracer, the lockstep bodies (sequential, pipelined, drafts), the
    # host loops, C and Fortran order, device input and batches (stacked and generic). Each mode's
    # variables are scoped, so a run-wide setting (BROOK_GRAPHS=0 for racecheck) holds between them.
    from brook.device import context

    options = dict(_JOIN_OPTIONS, **options)
    for mode in _JOIN_MODES:
        with monkeypatch.context() as scoped:
            for name, value in mode.items():
                scoped.setenv(name, value)
            for layout in (np.ascontiguousarray, np.asfortranarray):
                assert _summary(brook.skeletonize(layout(labels), **options)) == expected, (mode, layout.__name__)
    assert _summary(brook.skeletonize(context().upload(np.asfortranarray(labels)), **options)) == expected, "device"
    for uniform in ("1", "0"):
        with monkeypatch.context() as scoped:
            scoped.setenv("BROOK_BATCH_UNIFORM", uniform)
            batch = brook.skeletonize_batch([np.asfortranarray(labels)] * 3, output="skeletons", **options)
        assert [_summary(result) for result in batch] == [expected] * 3, f"batch uniform={uniform}"


# Expected values: Kimimaro 5.8.1 with dijkstra3d 1.15.2 (script settings, _JOIN_OPTIONS), as _summary.
@pytest.mark.gpu
@pytest.mark.parametrize("axis,digest", [(0, "200ef3501335cf16"), (1, "0534898a155b5511"), (2, "d56a71fc8dfcab68")])
def test_gpu_thin_slab_single_line(gpu, monkeypatch, axis, digest):
    # A thin band traces as one line (two endpoints). Its last path meets the first one's tip, where the
    # tip and its diagonal neighbour are both rails: joining the tip extends the line, while joining the
    # lowest flat index of the two would leave a one-voxel spur (one more endpoint and a branch point).
    _assert_joins_follow_kimimaro(monkeypatch, _sine_band(6.0, 60.0, 1.6, axis=axis), {1: (162, 161, 8.0, digest)})


@pytest.mark.gpu
@pytest.mark.parametrize("case", ["bands", "wide", "spiral"])
def test_gpu_thin_bands_join_follow_kimimaro(gpu, monkeypatch, case):
    # Joins of several labels at once, of a wider band, and of a spiral at unit anisotropy.
    labels, expected, options = {
        "bands": (_sine_bands(), {1: (162, 161, 8.0, "7e6a08ac7387f4ce"), 2: (184, 183, 12.0, "286cc820df17f2bf"),
                                  3: (160, 159, 8.0, "c33379d9635b67e3")}, {}),
        "wide": (_sine_band(9.0, 80.0, 2.1, 0.7), {1: (172, 171, 11.313708305358887, "9e3341073cdd01b3")}, {}),
        "spiral": (_spiral_band(), {1: (507, 505, 2.2360680103302, "d2f05a7237a1d4e7")}, {"anisotropy": (1, 1, 1)}),
    }[case]
    _assert_joins_follow_kimimaro(monkeypatch, labels, expected, **options)


@pytest.mark.gpu
def test_gpu_thin_tube_corner_join_order(gpu, monkeypatch):
    # A thin helical tube: its joins take corner steps, so dijkstra3d's corner order decides them
    # (Brook's own neighbour table orders the corners differently).
    _assert_joins_follow_kimimaro(monkeypatch, _helix_tube(1.9, 8.0, 28.0, 28), {1: (90, 89, 5.656854152679443, "317bd004853af27a")})


# One window size, a fixed reserve and the host loops: which drafts run and which are accepted then
# depend neither on timing nor on the GPU's memory.
_PINNED_DRAFTS = dict(BROOK_GRAPHS="0", BROOK_BODY="k", BROOK_KDRAFT="eager", BROOK_DRAFTS="8", BROOK_HORIZON="0",
                      BROOK_KPOOL="1", BROOK_KWINDOW="16", BROOK_DRAFT_RESERVE="8192")


@pytest.mark.gpu
def test_gpu_draft_join_repick(gpu, monkeypatch):
    # A window draft joins a rail of the batch's start; when a path accepted before it in the same batch
    # passes its last road voxel earlier in dijkstra3d's order, the sequential run joins that path, and
    # acceptance re-picks the join. This slab has one such draft in the pinned configuration.
    from brook.device import context

    labels = _thin_slabs((60, 1, 300), 1, seed=113)[0]
    reference = brook.skeletonize(labels, **_JOIN_OPTIONS)
    for mode in ({"BROOK_BODY": "seq"}, {"BROOK_LOCKSTEP": "0"}, _PINNED_DRAFTS):
        with monkeypatch.context() as scoped:
            for name, value in mode.items():
                scoped.setenv(name, value)
            _assert_same_skeletons(brook.skeletonize(labels, **_JOIN_OPTIONS), reference, str(mode))
    stats = context().stats
    assert stats["draft_repicked"] >= 1 and stats["accepted"] > stats["draft_repicked"], stats
    assert [stats[f"draft_window_{axis}"] for axis in "xyz"] == [8, 8, 16] and stats["draft_reserve"] == 8192, stats


@pytest.mark.gpu
def test_gpu_draft_join_fuzz(gpu, monkeypatch):
    # The pinned draft configuration against the sequential body: seeded thin slabs and a longer helical
    # tube (drafts with corner joins).
    from brook.device import context

    samples = [_thin_slabs((60, 1, 300), 1, seed)[0] for seed in range(3)] + [_helix_tube(2.2, 12.0, 36.0, 72)]
    for i, labels in enumerate(samples):
        with monkeypatch.context() as scoped:
            scoped.setenv("BROOK_BODY", "seq")
            reference = brook.skeletonize(labels, **_JOIN_OPTIONS)
        with monkeypatch.context() as scoped:
            for name, value in _PINNED_DRAFTS.items():
                scoped.setenv(name, value)
            _assert_same_skeletons(brook.skeletonize(labels, **_JOIN_OPTIONS), reference, f"sample {i}")
        assert context().stats["accepted"] > 0, i


@pytest.mark.gpu
@pytest.mark.parametrize("anisotropy", [(1.0, 1.0, 1.7), (1.1, 1.3, 2.9)])
def test_gpu_batch_stacked_z_spacing(gpu, monkeypatch, anisotropy):
    # A stacked sample's vertices are its own voxel coordinates times the anisotropy, rounded once, as
    # in a separate call, also where the z spacing's products are inexact in float32. Covered: the
    # lockstep's device and host assembly, the component tracer (sample 1's `after` target takes its
    # component out of the lockstep trace), a soma-sized body, host and device input, sub-batches of
    # two samples, the generic path and lockstep off, and a border-target root (the cut tube, stacked).
    from brook.device import context

    ctx = context()
    samples, before, after = _targeted_samples()
    options = dict(teasar_params=_CAVITY_PARAMS, anisotropy=anisotropy, dust_threshold=10, fix_borders=True,
                   fix_branching=True, progress=False)
    devices = [ctx.upload(np.asfortranarray(s)) for s in samples]
    spacing = np.float32(anisotropy[2])

    def separate():
        return [brook.skeletonize(np.asfortranarray(s), streaming=False, extra_targets_before=b,
                                  extra_targets_after=a, **options) for s, b, a in zip(samples, before, after)]

    def stack_frame_differs(skeletons, k, depth):
        # Scaled in the stack's frame and moved back by the scaled offset, sample k would get other z
        # coordinates: the comparisons below can tell the two computations apart.
        z, depth = np.concatenate([skeleton.vertices[:, 2] for skeleton in skeletons.values()]), np.float32(depth)
        voxel = np.rint(z / spacing).astype(np.float32)
        in_stack = (voxel + np.float32(k) * depth) * spacing + (np.float32(-k) * depth) * spacing
        return np.array_equal(voxel * spacing, z) and (in_stack != z).any()

    expected = separate()
    for k in range(1, len(samples)):
        assert stack_frame_differs(expected[k], k, samples[0].shape[2]), k

    def check(where, uniform, inputs=samples, sub_batches=None):
        batch = brook.skeletonize_batch(inputs, extra_targets_before=before, extra_targets_after=after, **options)
        stats = dict(context().stats)
        assert stats.get("batch_uniform", 0) == uniform, where
        assert sub_batches is None or stats["batch_sub_batches"] == sub_batches, where
        for i, reference in enumerate(expected):
            _assert_same_skeletons(batch[i].to_skeletons(), reference, f"{where}: sample {i}")

    check("stacked", 1)
    check("stacked, device input", 1, devices)
    # The samples above touch no face. The cut tube's label touches four; below the soma thresholds of
    # _HOLE_PARAMS its root is a border target.
    tube, _ = _cut_tube()
    cc, _, roots = ctx.connected_components(ctx.upload(tube))
    assert sum(len(b) for b in ctx.border_targets(cc, len(roots), anisotropy)) >= 4
    tube_options = dict(options, teasar_params=_HOLE_PARAMS)
    alone = brook.skeletonize(tube, streaming=False, **tube_options)
    assert all(stack_frame_differs(alone, k, tube.shape[2]) for k in (1, 2))
    batch = brook.skeletonize_batch([tube] * 3, output="skeletons", **tube_options)
    assert context().stats["batch_uniform"] == 1
    for i, result in enumerate(batch):
        _assert_same_skeletons(result, alone, f"stacked, border targets: sample {i}")
    monkeypatch.setenv("BROOK_BATCH_BUDGET", str(int(2.5 * samples[0].size * 96)))   # 96 bytes per voxel
    check("stacked, sub-batches of two", 1, sub_batches=3)
    monkeypatch.delenv("BROOK_BATCH_BUDGET")
    monkeypatch.setenv("BROOK_BATCH_UNIFORM", "0")
    check("generic", 0)
    monkeypatch.delenv("BROOK_BATCH_UNIFORM")
    monkeypatch.setenv("BROOK_ASSEMBLY", "cpu")                  # the lockstep assembles on the host
    expected = separate()
    check("stacked, host assembly", 1)
    monkeypatch.delenv("BROOK_ASSEMBLY")
    monkeypatch.setenv("BROOK_LOCKSTEP", "0")                    # no stacking: every sample traces alone
    expected = separate()
    check("lockstep off", 0)
    check("lockstep off, device input", 0, devices)


class _DLPackOnly:
    """An array that exposes only DLPack; `device` overrides the device it reports."""

    def __init__(self, array, device=None):
        self._array, self._device = array, device

    def __dlpack__(self, **kwargs):
        return self._array.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self._device or self._array.__dlpack_device__()


class _ArrayLike(_DLPackOnly):
    """A CPU tensor as PyTorch or JAX present one: DLPack and NumPy's `__array__`, no CUDA array interface."""

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._array, dtype=dtype)


class _HostReadableDevice:
    """A GPU array as JAX presents one: the CUDA array interface and an `__array__` that copies to the host."""

    def __init__(self, device):
        self._device = device

    @property
    def __cuda_array_interface__(self):
        return self._device.__cuda_array_interface__

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._device.get(), dtype=dtype)


def _dlpack_volume():
    labels = np.zeros((24, 20, 16), np.uint32)
    labels[2:22, 6:12, 5:11] = 7
    labels[4:8, 2:18, 12:15] = 9
    labels[10:14, 14:18, 2:4] = 3
    return np.asfortranarray(labels)


# Kimimaro 5.8.1 on _dlpack_volume() with _DLPACK_OPTIONS: each skeleton is a list of paths, each
# path its vertices and radii in vertex order (edges join consecutive vertices of a path).
# _DLPACK_CUT: the same with the cut graph of the test (Fortran order), which splits labels 3 and 7.
_DLPACK_PLAIN = {
    3: [([[10, 14, 2], [11, 15, 3], [12, 16, 3], [13, 17, 3]], [1] * 4)],
    7: [([[2, 6, 5], [3, 7, 6], [4, 8, 7]] + [[z, 9, 8] for z in range(5, 20)] + [[20, 10, 9], [21, 11, 10]],
         [1, 2] + [3] * 16 + [2, 1])],
    9: [([[4, 2, 12], [5, 3, 13]] + [[6, y, 13] for y in range(4, 17)] + [[7, 17, 14]], [1] + [2] * 14 + [1])],
}
_DLPACK_CUT = {
    3: [([[10, 14, 2], [11, 15, 3], [11, 16, 3], [11, 17, 3]], [.5] * 4),
        ([[12, 14, 2], [13, 15, 3], [13, 16, 3], [13, 17, 3]], [.5] * 4)],
    7: [([[2, 6, 5], [3, 7, 6], [4, 8, 7]] + [[z, 9, 8] for z in range(5, 10)] + [[10, 10, 9], [11, 11, 10]],
         [.5, 1.5] + [2.5] * 6 + [1.5, .5]),
        ([[12, 6, 5], [13, 7, 6], [14, 8, 7]] + [[z, 9, 8] for z in range(15, 20)] + [[20, 10, 9], [21, 11, 10]],
         [.5, 1.5] + [2.5] * 6 + [1.5, .5])],
    9: [([[4, 2, 12], [5, 3, 13]] + [[6, y, 13] for y in range(4, 17)] + [[7, 17, 14]], [.5] + [1.5] * 14 + [.5])],
}
_DLPACK_OPTIONS = dict(teasar_params=dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4,
                                          soma_detection_threshold=750, soma_acceptance_threshold=3500,
                                          soma_invalidation_scale=2, soma_invalidation_const=300),
                       dust_threshold=0, progress=False)


def _assert_paths(result, paths):
    assert list(result) == list(paths)
    for label, pieces in paths.items():
        vertices, edges, radii = [], [], []
        for piece, piece_radii in pieces:
            edges += [[len(vertices) + i, len(vertices) + i + 1] for i in range(len(piece) - 1)]
            vertices += piece
            radii += piece_radii
        np.testing.assert_array_equal(result[label].vertices, vertices)
        np.testing.assert_array_equal(result[label].edges, edges)
        np.testing.assert_array_equal(result[label].radii, radii)


@pytest.mark.gpu
@pytest.mark.parametrize("lockstep", ["0", "1"])
def test_gpu_dlpack_input_routing(gpu, monkeypatch, lockstep):
    # Device input is recognized by its CUDA array interface. A CPU tensor that NumPy reads through
    # __array__ is host input, as Kimimaro reads its labels; an array that exposes only DLPack is host
    # input when it is in host memory (CPU or pinned) and raises a TypeError on a GPU or in managed
    # memory. Kimimaro reads labels through NumPy's array protocols only, so the DLPack-only inputs
    # are checked against its skeletons for the same NumPy array. With host labels, a device graph
    # that NumPy reads (a JAX GPU array's __array__ copies it to the host) is read through NumPy.
    from brook.device import context

    monkeypatch.setenv("BROOK_LOCKSTEP", lockstep)
    labels = _dlpack_volume()
    side = np.zeros(labels.shape, bool)
    side[12:] = True
    # cc3d's convention (background steps permitted too), cut between slices 11 and 12
    graph = np.asfortranarray(_voxel_graph(labels + 1, side, np.ones(labels.shape, bool)))
    options = _DLPACK_OPTIONS
    for host in (_ArrayLike(labels), _DLPackOnly(labels), _DLPackOnly(np.ascontiguousarray(labels)),
                 _DLPackOnly(labels, device=(3, 0))):
        _assert_paths(brook.skeletonize(host, **options), _DLPACK_PLAIN)
        assert ("lockstep_labels" in context().stats) == (lockstep == "1")
    _assert_paths(brook.skeletonize(_DLPackOnly(labels), streaming=True, **options), _DLPACK_PLAIN)
    source = labels.copy(order="F")
    _assert_paths(brook.skeletonize(_DLPackOnly(source), object_ids=[7, 9], in_place=True, **options),
                  {label: _DLPACK_PLAIN[label] for label in (7, 9)})
    np.testing.assert_array_equal(source, labels)          # a DLPack import (maybe read-only) is not masked in place
    _assert_paths(brook.skeletonize(_DLPackOnly(labels), voxel_graph=_DLPackOnly(graph), **options), _DLPACK_CUT)
    ctx = context()
    device_labels, device_graph = ctx.upload(labels), ctx.upload(graph)
    _assert_paths(brook.skeletonize(device_labels, voxel_graph=_DLPackOnly(graph), **options), _DLPACK_CUT)
    _assert_paths(brook.skeletonize(device_labels, voxel_graph=_ArrayLike(graph), **options), _DLPACK_CUT)
    for volume in (labels, device_labels):
        _assert_paths(brook.skeletonize(volume, voxel_graph=_HostReadableDevice(device_graph), **options), _DLPACK_CUT)

    # Batches: two shapes (the generic path), a dense DLPack or __array__ array (the stacked path; the
    # stand-in is not iterable), per-sample and dense graphs. Appending empty slices along the last axis
    # keeps every skeleton.
    wide = np.pad(labels, ((0, 0), (0, 0), (0, 4)))
    for result in brook.skeletonize_batch([_DLPackOnly(labels), _ArrayLike(wide)], output="skeletons", **options):
        _assert_paths(result, _DLPACK_PLAIN)
    assert context().stats.get("batch_uniform", 0) == 0
    uniform = int(lockstep == "1")                          # the stacked path traces in lockstep
    for wrap in (_DLPackOnly, _ArrayLike):
        for result in brook.skeletonize_batch(wrap(np.stack([labels, labels])), output="skeletons", **options):
            _assert_paths(result, _DLPACK_PLAIN)
        assert context().stats.get("batch_uniform", 0) == uniform
        stacked = brook.skeletonize_batch([labels, labels], voxel_graph=wrap(np.stack([graph, graph])),
                                          output="skeletons", **options)
        assert context().stats.get("batch_uniform", 0) == uniform
        for result in stacked:
            _assert_paths(result, _DLPACK_CUT)
    mixed = brook.skeletonize_batch([device_labels, wide], voxel_graph=[_DLPackOnly(graph), None], output="skeletons", **options)
    _assert_paths(mixed[0], _DLPACK_CUT)
    _assert_paths(mixed[1], _DLPACK_PLAIN)

    for call in (lambda: brook.skeletonize(_DLPackOnly(device_labels), **options),
                 lambda: brook.skeletonize(_DLPackOnly(labels, device=(13, 0)), **options),     # managed memory
                 lambda: brook.skeletonize(labels, voxel_graph=_DLPackOnly(device_graph), **options),
                 lambda: brook.skeletonize(device_labels, voxel_graph=_DLPackOnly(device_graph), **options),
                 lambda: brook.skeletonize_batch([labels, _DLPackOnly(device_labels)], **options),
                 lambda: brook.skeletonize_batch(_DLPackOnly(device_labels), **options),
                 lambda: brook.skeletonize_batch([labels], voxel_graph=[_DLPackOnly(device_graph)], **options),
                 lambda: brook.skeletonize_batch([labels], voxel_graph=_DLPackOnly(device_graph), **options)):
        with pytest.raises(TypeError, match="CUDA array interface"):
            call()
    np.testing.assert_array_equal(device_labels.get(), labels)


def _arch_number(name):
    return int("".join(c for c in name.split("_", 1)[1] if c.isdigit()))


def test_cpu_build_info():
    # The build names its GPU code; the minimum is compute capability 8.0.
    from brook.device import build_info

    info = build_info()
    assert info["sass"] or info["ptx"], info
    assert all(_arch_number(name) >= 80 for name in info["sass"] + info["ptx"]), info
    assert info["cuda_toolkit"] and info["cuda_runtime"], info


@pytest.mark.gpu
def test_gpu_startup_check(gpu):
    # require_gpu() and every Context run the same check; the build has code this GPU can load.
    from brook import _core
    from brook.device import build_info, device_count, device_properties, require_gpu

    require_gpu()
    require_gpu(0)
    _core.check_device(0)
    with pytest.raises(ValueError):
        _core.check_device(device_count())
    props, info = device_properties(0), build_info()
    capability = props["major"] * 10 + props["minor"]
    assert capability >= 80
    sass = [_arch_number(name) for name in info["sass"]]
    ptx = [_arch_number(name) for name in info["ptx"]]
    assert any(a // 10 == capability // 10 and a <= capability for a in sass) or any(a <= capability for a in ptx), info
    assert info["cuda_driver"] is not None


def test_cpu_pool_startup_check_without_warmup():
    # Every pool worker runs the startup check when it starts, warm or not: a worker that sees no
    # usable device fails the pool at start with Brook's message.
    with pytest.raises(RuntimeError, match="worker on GPU 99 failed to start") as failure:
        brook.GpuPool(devices=["99"], warmup=False)
    assert "Brook found no CUDA device" in str(failure.value), failure.value


def test_cpu_profile_sync_keeps_the_block_error(monkeypatch):
    # BROOK_PROFILE=sync synchronizes after each range that finished, so a CUDA error surfaces
    # there; it never replaces an exception the profiled block raised.
    from brook import profile

    calls = []

    def failing_sync():
        calls.append("sync")
        raise RuntimeError("device error")

    monkeypatch.setattr(profile, "ENABLED", True)
    monkeypatch.setattr(profile, "SYNC", True)
    monkeypatch.setattr(profile, "_sync", failing_sync)
    prof = profile.Profiler()
    with pytest.raises(ValueError, match="block error"):
        with prof.range("block"):
            raise ValueError("block error")
    assert calls == []
    with pytest.raises(RuntimeError, match="device error"):
        with prof.range("block"):
            pass
    assert calls == ["sync"] and prof.stats["block"].count == 2


def _comb_volume():
    # Six comb-shaped labels: many branches each, traced in lockstep.
    labels = np.zeros((64, 64, 64), np.uint16)
    for i, z in enumerate(range(6, 60, 9)):
        labels[4:60, 30:33, z:z + 3] = i + 1
        for x in range(8, 60, 10):
            labels[x:x + 2, 8:56, z:z + 3] = i + 1
    return np.asfortranarray(labels)


_COMB_OPTIONS = dict(teasar_params=dict(scale=1.5, const=2, pdrf_scale=100000, pdrf_exponent=4), dust_threshold=0,
                     fix_borders=False, fix_branching=True, progress=False, streaming=False)


def _body_iterations(stats):
    # One lockstep trace: its iterations ran in the sequential body, the pipelined body, or the
    # draft lane alone.
    return sum(stats.get(f"iterations_{body}", 0) for body in ("seq", "pipe", "k"))


def _assert_host_loop_stats(stats):
    # The host loops (BROOK_GRAPHS=0) run the sequential lockstep body only.
    assert stats["graphs"] == 0 and stats["chunks_seq"] >= 1 and stats["iterations"] == _body_iterations(stats), stats
    assert not any(key.startswith(("grid_pipe_", "grid_crit_")) for key in stats), stats


@pytest.mark.gpu
def test_gpu_grid_and_scheduler_stats(gpu, monkeypatch):
    # The stats name the barrier-probe grid, the grid of each cooperative kernel family and the
    # lockstep bodies the scheduler ran. Forced bodies, a grid override and the host loops
    # (BROOK_GRAPHS=0) give the same skeletons.
    from brook.device import context

    labels = _comb_volume()
    reference = brook.skeletonize(labels, **_COMB_OPTIONS)
    stats = context().stats
    if not stats["graphs"]:  # BROOK_GRAPHS=0 for the whole run, as for racecheck and synccheck
        _assert_host_loop_stats(stats)
        return
    probe = stats["grid_probe"]
    assert stats["graphs"] == 1 and stats["lockstep_labels"] == 6 and probe >= 1
    wide = {family: stats[f"grid_lockstep_{family}"] for family in ("target", "invalidation", "railroad")}
    assert all(1 <= grid <= probe for grid in wide.values()), stats
    assert stats.get("chunks_seq", 0) + stats.get("chunks_pipe", 0) >= 1
    assert stats["iterations"] == _body_iterations(stats) >= 1, stats

    monkeypatch.setenv("BROOK_BODY", "pipe")
    _assert_same_skeletons(brook.skeletonize(labels, **_COMB_OPTIONS), reference, "pipelined body")
    stats = context().stats
    assert stats["pipelined"] == 1 and stats["chunks_pipe"] >= 1 and stats["iterations_pipe"] >= 1, stats
    assert stats["pipe_rejected"] >= 0, stats
    assert stats["iterations"] == _body_iterations(stats), stats
    for family in wide:  # the two branches share the grid of the sequential body (no drafts: same arenas)
        assert 1 <= stats[f"grid_pipe_{family}"] <= max(1, wide[family] // 2), stats

    monkeypatch.setenv("BROOK_BODY", "k")
    _assert_same_skeletons(brook.skeletonize(labels, **_COMB_OPTIONS), reference, "draft lane")
    stats = context().stats
    share = stats["grid_crit_share"]
    assert stats["chunks_k"] >= 1 and share == 4 and stats["draft_repicked"] >= 0, stats
    assert stats["iterations"] == _body_iterations(stats), stats
    for family in ("target", "draft", "railroad", "invalidation"):  # a share of grids the probe caps
        assert 1 <= stats[f"grid_crit_{family}"] <= max(1, probe // share), stats
    monkeypatch.delenv("BROOK_BODY")

    # Each override variable has its own key; with every family overridden the probe is not used.
    monkeypatch.setenv("BROOK_LOCKSTEP_GRID", "64")
    monkeypatch.setenv("BROOK_GRID", "48")
    _assert_same_skeletons(brook.skeletonize(labels, **_COMB_OPTIONS), reference, "grid override")
    stats = context().stats
    assert stats["grid_override_lockstep"] == 64 and stats["grid_override"] == 48 and "grid_probe" not in stats, stats
    assert all(1 <= stats[f"grid_lockstep_{family}"] <= 64 for family in wide), stats
    assert 1 <= stats["grid_batch_flood"] <= 48, stats
    monkeypatch.delenv("BROOK_LOCKSTEP_GRID")
    monkeypatch.delenv("BROOK_GRID")

    monkeypatch.setenv("BROOK_GRAPHS", "0")
    _assert_same_skeletons(brook.skeletonize(labels, **_COMB_OPTIONS), reference, "host loops")
    _assert_host_loop_stats(context().stats)
