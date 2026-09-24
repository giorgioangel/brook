// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
namespace brook {
using CpuTransform = std::array<float, 12>;
inline constexpr CpuTransform cpu_identity = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
// A 3-by-3 partial-pivot LU solve in double precision, then float32 output,
// matching NumPy's float32 linalg.inv promotion. Translation follows osteoid 0.7.3's
// inverse convention (-t), including its non-affine inverse convention.
inline CpuTransform cpu_inverse(const CpuTransform &t) {
  double a[3][3], b[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      a[i][j] = t[4 * i + j];
  // OpenBLAS 0.3.31 GETF2 uses column-wise Crout updates for this size.
  // In particular the final two products are summed before subtraction;
  // row-wise rank-one updates change near-singular float32 inverse results.
  // Source/notice: cpp/licenses/OpenBLAS-getf2.txt.
  int pivots[3];
  for (int column = 0; column < 3; ++column) {
    for (int row = 0; row < column; ++row)
      std::swap(a[row][column], a[pivots[row]][column]);
    for (int row = 1; row < column; ++row) {
      double sum = 0;
      for (int k = 0; k < row; ++k)
        sum += a[row][k] * a[k][column];
      a[row][column] -= sum;
    }
    for (int row = column; row < 3; ++row) {
      double sum = 0;
      for (int k = 0; k < column; ++k)
        sum += a[row][k] * a[k][column];
      a[row][column] -= sum;
    }
    int pivot = column;
    for (int row = column + 1; row < 3; ++row)
      if (std::abs(a[row][column]) > std::abs(a[pivot][column]))
        pivot = row;
    pivots[column] = pivot;
    if (a[pivot][column] == 0)
      throw std::invalid_argument("singular skeleton transform");
    double inverse = 1 / a[pivot][column];
    if (pivot != column)
      for (int k = 0; k <= column; ++k)
        std::swap(a[column][k], a[pivot][k]);
    for (int row = column + 1; row < 3; ++row)
      a[row][column] *= inverse;
  }
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      std::swap(b[row][column], b[pivots[row]][column]);
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < i; ++k)
      for (int j = 0; j < 3; ++j)
        b[i][j] = std::fma(-a[i][k], b[k][j], b[i][j]);
  for (int i = 2; i >= 0; --i) {
    for (int k = i + 1; k < 3; ++k)
      for (int j = 0; j < 3; ++j)
        b[i][j] = std::fma(-a[i][k], b[k][j], b[i][j]);
    for (int j = 0; j < 3; ++j)
      b[i][j] /= a[i][i];
  }
  CpuTransform out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j)
      out[4 * i + j] = float(b[i][j]);
    out[4 * i + 3] = -t[4 * i + 3];
  }
  return out;
}
inline std::array<float, 3> cpu_transform_point(std::array<float, 3> v,
                                                const CpuTransform &t) {
  std::array<float, 3> o;
  for (int i = 0; i < 3; ++i) {
    float x = t[4 * i] * v[0];
    x = std::fma(t[4 * i + 1], v[1], x);
    x = std::fma(t[4 * i + 2], v[2], x);
    o[i] = std::fma(t[4 * i + 3], 1.f, x);
  }
  return o;
}
inline std::array<double, 3> cpu_transform_point(std::array<double, 3> v,
                                                 const CpuTransform &t, bool wide) {
  if (!wide) {
    auto o = cpu_transform_point(std::array<float, 3>{float(v[0]), float(v[1]), float(v[2])}, t);
    return {o[0], o[1], o[2]};
  }
  std::array<double, 3> o;
  for (int i = 0; i < 3; ++i) {
    double x = double(t[4 * i]) * v[0];
    x = std::fma(double(t[4 * i + 1]), v[1], x);
    x = std::fma(double(t[4 * i + 2]), v[2], x);
    o[i] = std::fma(double(t[4 * i + 3]), 1.0, x);
  }
  return o;
}
} // namespace brook
