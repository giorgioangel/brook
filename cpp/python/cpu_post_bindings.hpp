// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include "cpu_post.hpp"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace brook::python {
namespace py = pybind11;
inline CpuGeometry cpu_input(py::handle vertices, py::handle edges, py::handle radii) {
  auto va = py::array::ensure(vertices), ra = py::array::ensure(radii);
  if (!va || !ra)
    throw py::value_error("vertices and radii must be arrays");
  auto isdouble = [](const py::array &a) {
    if (a.dtype().equal(py::dtype::of<double>()))
      return true;
    if (a.dtype().equal(py::dtype::of<float>()))
      return false;
    throw py::type_error("vertices/radii require float32 or float64 arrays");
  };
  CpuGeometry s;
  s.vertex64 = isdouble(va);
  s.radius64 = isdouble(ra);
  auto v = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(va);
  auto e = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>::ensure(edges);
  auto r = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(ra);
  if (v.ndim() != 2 || v.shape(1) != 3 || !e || (e.size() && (e.ndim() != 2 || e.shape(1) != 2)) ||
      r.ndim() != 1 || r.size() != v.shape(0))
    throw py::value_error("expected vertices (N,3), edges (M,2), radii (N)");
  s.vertices.resize(v.shape(0));
  s.edges.resize(e.size() / 2);
  s.radii.resize(r.size());
  if (v.size())
    std::memcpy(s.vertices.data(), v.data(), v.nbytes());
  if (e.size())
    std::memcpy(s.edges.data(), e.data(), e.nbytes());
  if (r.size())
    std::memcpy(s.radii.data(), r.data(), r.nbytes());
  return s;
}
inline CpuMetadata cpu_metadata(py::object transform, const std::string &space,
                                int64_t owner) {
  CpuMetadata metadata;
  metadata.owner = owner;
  if (space != "voxel" && space != "physical")
    throw py::value_error("space must be voxel or physical");
  metadata.physical = space == "physical";
  if (!transform.is_none()) {
    auto t = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(transform);
    if (!t || t.size() != 12)
      throw py::value_error("transform must contain 12 values");
    std::memcpy(metadata.transform.data(), t.data(), 48);
  }
  return metadata;
}
inline py::dict cpu_output(const CpuSkeleton &g) {
  py::array v(g.skeleton.vertex64 ? py::dtype::of<double>() : py::dtype::of<float>(),
              {py::ssize_t(g.sources.size()), py::ssize_t(3)});
  py::array r(g.skeleton.radius64 ? py::dtype::of<double>() : py::dtype::of<float>(),
              std::vector<py::ssize_t>{py::ssize_t(g.sources.size())});
  py::array_t<uint32_t> e({py::ssize_t(g.skeleton.edges.size()), py::ssize_t(2)});
  py::array_t<uint64_t> source(g.sources.size());
  if (g.skeleton.vertex64) {
    if (v.size())
      std::memcpy(v.mutable_data(), g.skeleton.vertices.data(), v.nbytes());
  } else {
    auto *p = static_cast<float *>(v.mutable_data());
    for (auto point : g.skeleton.vertices)
      for (auto x : point)
        *p++ = float(x);
  }
  if (g.skeleton.radius64) {
    if (r.size())
      std::memcpy(r.mutable_data(), g.skeleton.radii.data(), r.nbytes());
  } else {
    auto *p = static_cast<float *>(r.mutable_data());
    for (auto x : g.skeleton.radii)
      *p++ = float(x);
  }
  if (e.size())
    std::memcpy(e.mutable_data(), g.skeleton.edges.data(), e.nbytes());
  if (source.size())
    std::memcpy(source.mutable_data(), g.sources.data(), source.nbytes());
  py::dict out;
  out["vertices"] = v;
  out["edges"] = e;
  out["radii"] = r;
  out["sources"] = source;
  out["metadata_owner"] = g.metadata.owner;
  out["space"] = g.metadata.physical ? "physical" : "voxel";
  py::array_t<float> t({py::ssize_t(3), py::ssize_t(4)});
  std::memcpy(t.mutable_data(), g.metadata.transform.data(), 48);
  out["transform"] = t;
  return out;
}
inline bool cpu_single_scalar(py::handle value) {
  return py::hasattr(value, "dtype") &&
         py::cast<std::string>(value.attr("dtype").attr("kind")) == "f" &&
         py::cast<int>(value.attr("dtype").attr("itemsize")) <= 4;
}
inline void bind_cpu_post(py::module_ &m) {
  py::register_exception<CpuQueryDtypeError>(m, "CpuQueryDtypeError", PyExc_TypeError);
  m.def(
      "cpu_postprocess",
      [](py::handle v, py::handle e, py::handle r, py::object dust, py::object tick,
         py::object transform, std::string space) {
        auto metadata = cpu_metadata(transform, space, 0);
        auto s = cpu_input(v, e, r);
        double d = py::cast<double>(dust);
        // The C API compares thresholds in double precision. A Python int or float is a weak
        // scalar in NumPy, so Kimimaro compares it in float32 against float32 vertices.
        if (!s.vertex64 && (PyFloat_CheckExact(dust.ptr()) || PyLong_CheckExact(dust.ptr())))
          d = double(float(d));
        double t = py::cast<double>(tick);
        bool single_tick = cpu_single_scalar(tick);
        CpuSkeleton out;
        {
          py::gil_scoped_release release;
          out = cpu_postprocess(std::move(s), d, t, metadata, single_tick);
        }
        return cpu_output(out);
      },
      py::arg("vertices"), py::arg("edges"), py::arg("radii"), py::arg("dust_threshold") = 1500,
      py::arg("tick_threshold") = 3500, py::arg("transform") = py::none(),
      py::arg("space") = "voxel");
  m.def(
      "cpu_join",
      [](py::iterable inputs, py::object radius, bool restrict) {
        double r =
            radius.is_none() ? std::numeric_limits<double>::infinity() : py::cast<double>(radius);
        bool single_radius = cpu_single_scalar(radius),
             strong_radius = py::hasattr(radius, "dtype") && !single_radius;
        std::vector<CpuGeometry> ss;
        std::vector<CpuMetadata> metadata;
        for (auto item : inputs) {
          auto t = py::cast<py::tuple>(item);
          if (t.size() != 3 && t.size() != 5)
            throw py::value_error(
                "join inputs must be (vertices,edges,radii[,transform,space]) tuples");
          metadata.push_back(cpu_metadata(
              t.size() == 5 ? py::reinterpret_borrow<py::object>(t[3]) : py::object(py::none()),
              t.size() == 5 ? py::cast<std::string>(t[4]) : "voxel", ss.size()));
          ss.push_back(cpu_input(t[0], t[1], t[2]));
        }
        CpuSkeleton out;
        CpuJoinTrace trace;
        {
          py::gil_scoped_release release;
          out = cpu_join_precise(std::move(ss), r, restrict, metadata, single_radius,
                                 strong_radius, &trace);
        }
        auto result = cpu_output(out);
        result["pieces"] =
            py::array_t<int64_t>(py::ssize_t(trace.pieces.size()), trace.pieces.data());
        result["joined"] = trace.joined;
        return result;
      },
      py::arg("inputs"), py::arg("radius") = std::numeric_limits<double>::infinity(),
      py::arg("restrict_by_radius") = false);
}
} // namespace brook::python
