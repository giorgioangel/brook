# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""PyTorch interop: label tensors in, skeleton tensors out, on the caller's CUDA stream.

`skeletonize` and `skeletonize_batch` accept CUDA tensors (CPU tensors are uploaded), default
`stream` to the current CUDA stream of the tensor's device, and return `vertices`, `edges` and
`radii` as torch tensors through DLPack without a copy; the tensors stay valid after the Brook
result is freed. Offsets, labels and sample indices are CPU int64 tensors. There is no gradient:
`Skeletonize.forward` runs under `torch.no_grad()`. PyTorch is imported on first use, so
`import brook.contrib.torch` works without it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("brook.contrib.torch requires PyTorch, which is not installed; "
                          "`pip install torch` (it is not a Brook dependency).") from error
    return torch


def _resolve_device(torch, explicit, *tensors):
    """The explicit device, else the first CUDA tensor's device, else the current CUDA device."""
    if explicit is not None:
        return torch.device(explicit)
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            return t.device
    return torch.device("cuda", torch.cuda.current_device())


def _current_stream_handle(torch, device):
    return torch.cuda.current_stream(device).cuda_stream


def _fortranize(x, torch):
    """Rearrange the trailing three dims into a Fortran-contiguous block (any leading
    batch dimension keeps its own, larger, stride), so Brook borrows the tensor in place instead
    of copying and transposing it. This permute/contiguous/permute round trip is itself a copy;
    it pays off because every later call against this tensor is then zero-copy."""
    n = x.dim()
    if n < 3:
        raise ValueError(f"brook.contrib.torch: layout='fortran' needs at least 3 dimensions, got {tuple(x.shape)}")
    perm = list(range(n - 3)) + [n - 1, n - 2, n - 3]   # its own inverse: reverses only the last 3 axes
    return x.permute(*perm).contiguous().permute(*perm)


def _prepare(tensor, torch, layout, device):
    """Validate one label tensor, upload it if it is on the CPU, and apply `layout`."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"brook.contrib.torch expects torch.Tensor input, got {type(tensor)!r}")
    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise TypeError(f"brook.contrib.torch: labels must be an integer or boolean tensor, got dtype={tensor.dtype}")
    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.uint8)
    if tensor.device.type == "cpu":
        tensor = tensor.to(device)
    elif tensor.device != device:
        raise ValueError(f"brook.contrib.torch: tensor is on {tensor.device}, expected {device} "
                         "(pass device=... explicitly, or move the tensors onto one device first)")
    if layout == "fortran":
        tensor = _fortranize(tensor, torch)
    elif layout is not None:
        raise ValueError("brook.contrib.torch: layout must be 'fortran' or None")
    return tensor


def _to_tensor(array, torch):
    """Zero-copy torch tensor for a Brook device array (DLPack); the capsule keeps the underlying
    allocation alive independently of `array`."""
    return array if isinstance(array, torch.Tensor) else torch.from_dlpack(array)


def _offsets(values, torch):
    """A CPU int64 tensor sharing memory with a freshly downloaded offsets/labels/sample array."""
    return values if isinstance(values, torch.Tensor) else torch.from_numpy(values)


@dataclass
class PackedSkeletonsT:
    """Torch view of a `brook.PackedSkeletons`. `vertices`/`edges`/`radii` are tensors on
    `device` (zero-copy via DLPack from Brook's own device arrays); `labels`/`v_off`/`e_off` are
    CPU int64 tensors. `edges` keeps Brook's native `uint32` dtype (torch's arithmetic on it is
    limited; call `.long()` for a copy with full operator support)."""
    labels: torch.Tensor
    v_off: torch.Tensor
    e_off: torch.Tensor
    vertices: torch.Tensor
    edges: torch.Tensor
    radii: torch.Tensor
    anisotropy: tuple
    device: torch.device

    def __len__(self):
        return int(self.labels.numel())


@dataclass
class BatchedSkeletonsT:
    """Torch view of a `brook.BatchedSkeletons`: every sample's skeletons, packed in sample
    order. `vertices`/`edges`/`radii` are tensors on `device`; `labels`/`sample`/`s_off`/`v_off`/
    `e_off` are CPU int64 tensors. `sample[k]`/the range `s_off[i]:s_off[i+1]` are as in
    `brook.BatchedSkeletons`."""
    labels: torch.Tensor
    sample: torch.Tensor
    s_off: torch.Tensor
    v_off: torch.Tensor
    e_off: torch.Tensor
    vertices: torch.Tensor
    edges: torch.Tensor
    radii: torch.Tensor
    anisotropy: tuple
    device: torch.device

    def __len__(self):
        return int(self.s_off.numel()) - 1

    def __getitem__(self, i):
        """Sample i as a `PackedSkeletonsT` view sharing the batch's storage (no copy)."""
        i = int(i)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        lo, hi = int(self.s_off[i]), int(self.s_off[i + 1])
        v0, v1 = int(self.v_off[lo]), int(self.v_off[hi])
        e0, e1 = int(self.e_off[lo]), int(self.e_off[hi])
        return PackedSkeletonsT(labels=self.labels[lo:hi], v_off=self.v_off[lo:hi + 1] - self.v_off[lo],
                                e_off=self.e_off[lo:hi + 1] - self.e_off[lo], vertices=self.vertices[v0:v1],
                                edges=self.edges[e0:e1], radii=self.radii[v0:v1],
                                anisotropy=self.anisotropy, device=self.device)


def _wrap_packed(result, torch, device):
    return PackedSkeletonsT(labels=_offsets(result.labels, torch), v_off=_offsets(result.v_off, torch),
                            e_off=_offsets(result.e_off, torch), vertices=_to_tensor(result.vertices, torch),
                            edges=_to_tensor(result.edges, torch), radii=_to_tensor(result.radii, torch),
                            anisotropy=tuple(result.anisotropy), device=device)


def _wrap_batched(result, torch, device):
    p = result.packed
    return BatchedSkeletonsT(labels=_offsets(result.labels, torch), sample=_offsets(result.sample, torch),
                             s_off=_offsets(result.s_off, torch), v_off=_offsets(p.v_off, torch),
                             e_off=_offsets(p.e_off, torch), vertices=_to_tensor(p.vertices, torch),
                             edges=_to_tensor(p.edges, torch), radii=_to_tensor(p.radii, torch),
                             anisotropy=tuple(p.anisotropy), device=device)


def skeletonize(labels, *, layout=None, stream=None, device=None, **options):
    """Skeletonize one label volume resident on a CUDA device; wraps `brook.skeletonize`.

    `labels`: a 3-D integer or boolean tensor on a CUDA device or the CPU (it is uploaded first);
    `anisotropy` and points follow its axis order, as in `brook.skeletonize`. By default the tensor
    is copied once and transposed into Brook's Fortran layout on the GPU; pass `layout="fortran"` to
    rearrange it into a Fortran-contiguous view first, so Brook borrows it in place instead (zero
    copy; the input is never written and stays alive for the call). `stream` defaults to the calling
    thread's current CUDA stream on the tensor's device
    (`torch.cuda.current_stream(...).cuda_stream`); Brook's own work waits on an event recorded
    there before it reads the tensor, and every result is complete, on that stream, by the time this
    call returns. Every other keyword is a `brook.skeletonize` option; `output` defaults to
    `"packed_device"` here, so the result's arrays stay torch tensors on `device`
    (`PackedSkeletonsT`) -- pass `output="skeletons"` or `"packed"` for `brook`'s own host-resident
    result types instead.

    Skeletonization is a discrete graph search: it has no gradient. Do not call this on a tensor
    you expect to backpropagate through.
    """
    torch = _torch()
    import brook

    dev = _resolve_device(torch, device, labels)
    labels = _prepare(labels, torch, layout, dev)
    options.setdefault("output", "packed_device")
    options.setdefault("progress", False)
    if stream is None:
        stream = _current_stream_handle(torch, dev)
    result = brook.skeletonize(labels, stream=stream, **options)
    return _wrap_packed(result, torch, dev) if options["output"] == "packed_device" else result


def skeletonize_batch(samples, *, layout=None, stream=None, device=None, **options):
    """Skeletonize independent samples in one call; wraps `brook.skeletonize_batch`.

    `samples`: a 4-D tensor with the samples along its first axis, or a sequence of 3-D tensors
    (shapes may differ); integer or boolean dtype, on a CUDA device or the CPU (uploaded first).
    `layout="fortran"` and `stream` behave as in `skeletonize`. `output` defaults to
    `"packed_device"`, giving a `BatchedSkeletonsT` whose arrays are torch tensors on `device`.
    """
    torch = _torch()
    import brook

    if isinstance(samples, torch.Tensor):
        dev = _resolve_device(torch, device, samples)
        samples_arg = _prepare(samples, torch, layout, dev)
    else:
        tensors = list(samples)
        if not tensors:
            raise ValueError("brook.contrib.torch.skeletonize_batch: samples is empty")
        dev = _resolve_device(torch, device, *tensors)
        samples_arg = [_prepare(t, torch, layout, dev) for t in tensors]
    options.setdefault("output", "packed_device")
    options.setdefault("progress", False)
    if stream is None:
        stream = _current_stream_handle(torch, dev)
    result = brook.skeletonize_batch(samples_arg, stream=stream, **options)
    return _wrap_batched(result, torch, dev) if options["output"] == "packed_device" else result


def padded(result, max_skeletons=None, max_vertices=None):
    """Dense CUDA tensors for a batched result: `vertices [B, S, V, 3]`, `radii [B, S, V]`,
    `labels [B, S]` (`-1` where padded), `mask [B, S, V]`. `S`/`V` default to the batch's own
    maximum skeleton/vertex counts (each needs one device-to-host sync to size the output;
    passing both `max_skeletons` and `max_vertices` avoids both, so the whole call stays on the
    device). Skeletons beyond `max_skeletons` in a sample, or vertices beyond `max_vertices` in a
    skeleton, are dropped. Equivalent to `BatchedSkeletons.padded()`, but computed with torch
    ops on `result`'s own device instead of a host round trip."""
    torch = _torch()
    device = result.vertices.device
    K = int(result.labels.numel())
    B = int(result.s_off.numel()) - 1
    s_off = result.s_off.to(device)
    v_off = result.v_off.to(device)
    sample = result.sample.to(device)
    labels = result.labels.to(device)

    counts = s_off[1:] - s_off[:-1]
    nv = v_off[1:] - v_off[:-1]
    S = int(max_skeletons) if max_skeletons is not None else (int(counts.max()) if B else 0)
    V = int(max_vertices) if max_vertices is not None else (int(nv.max()) if K else 0)

    vertices = torch.zeros(B, S, V, 3, dtype=result.vertices.dtype, device=device)
    radii = torch.zeros(B, S, V, dtype=result.radii.dtype, device=device)
    mask = torch.zeros(B, S, V, dtype=torch.bool, device=device)
    out_labels = torch.full((B, S), -1, dtype=torch.int64, device=device)

    if K and S:
        s_idx = torch.arange(K, device=device) - s_off[sample]        # each skeleton's index within its sample
        valid_skel = s_idx < S
        if bool(valid_skel.any()):
            out_labels[sample[valid_skel], s_idx[valid_skel]] = labels[valid_skel]
        if V:
            local_v = torch.arange(V, device=device)
            valid_v = (local_v.unsqueeze(0) < nv.clamp(max=V).unsqueeze(1)) & valid_skel.unsqueeze(1)   # [K, V]
            if bool(valid_v.any()):
                src = v_off[:-1].unsqueeze(1) + local_v.unsqueeze(0)                     # global vertex index, [K, V]
                b_grid = sample.unsqueeze(1).expand(K, V)[valid_v]
                s_grid = s_idx.unsqueeze(1).expand(K, V)[valid_v]
                v_grid = local_v.unsqueeze(0).expand(K, V)[valid_v]
                src = src[valid_v]
                vertices[b_grid, s_grid, v_grid] = result.vertices[src]
                radii[b_grid, s_grid, v_grid] = result.radii[src]
                mask[b_grid, s_grid, v_grid] = True

    return dict(vertices=vertices, radii=radii, labels=out_labels, mask=mask)


def _build_skeletonize_module():
    torch = _torch()

    class Skeletonize(torch.nn.Module):
        """Forward-only Brook skeletonization as an `nn.Module`.

        Skeletonization is a discrete TEASAR graph search over the label volume: it has no
        gradient, and this module never records one (`forward` runs under `torch.no_grad()`).
        `forward` calls `skeletonize_batch` for a 4-D tensor (samples along the first axis) or a
        sequence of 3-D tensors, and `skeletonize` for a single 3-D tensor; every keyword passed to
        `__init__` (`layout`, `stream`, `anisotropy`, `teasar_params`, ...) is forwarded. Do not
        feed this a tensor you expect to backpropagate through -- detach it first
        (`Skeletonize()(labels.detach())`); a tensor that still requires grad raises.
        """

        def __init__(self, **options):
            super().__init__()
            self.options = options

        @torch.no_grad()
        def forward(self, labels):
            tensors = [labels] if isinstance(labels, torch.Tensor) else list(labels)
            if any(t.requires_grad for t in tensors if isinstance(t, torch.Tensor)):
                raise RuntimeError(
                    "brook.contrib.torch.Skeletonize: the input requires grad, but skeletonization is a "
                    "discrete graph search with no gradient -- none will flow back through its output. "
                    "Detach the input first, e.g. `Skeletonize()(labels.detach())`.")
            if isinstance(labels, torch.Tensor):
                if labels.dim() == 4:
                    return skeletonize_batch(labels, **self.options)
                if labels.dim() == 3:
                    return skeletonize(labels, **self.options)
                raise ValueError("brook.contrib.torch.Skeletonize expects a 3-D or 4-D (samples along the first axis) "
                                 f"label tensor, or a sequence of 3-D tensors, got shape {tuple(labels.shape)}")
            return skeletonize_batch(labels, **self.options)

    return Skeletonize


_LAZY = {}


def __getattr__(name):
    if name == "Skeletonize":
        if "Skeletonize" not in _LAZY:
            _LAZY["Skeletonize"] = _build_skeletonize_module()
        return _LAZY["Skeletonize"]
    raise AttributeError(f"module 'brook.contrib.torch' has no attribute {name!r}")


# Skeletonize is created on first access because it subclasses torch.nn.Module; it stays out of
# __all__ so that a star import works without torch.
__all__ = ["skeletonize", "skeletonize_batch", "padded", "PackedSkeletonsT", "BatchedSkeletonsT"]
