// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Giorgio Angelotti
/* Pinned PyKDtree 1.4.3 core: LGPL-3.0-or-later, cpp/licenses/PyKDTree.txt.
 * Kept private and serial: parallelism belongs at the independent graph level. */
#include "third_party/pykdtree_core.c"
#include <math.h>
void *brook_cpu_tree_create(float *v, uint32_t n) {
  return construct_tree_float_int32_t(v, 3, n, 16);
}
void brook_cpu_tree_delete(void *p) { delete_tree_float_int32_t(p); }
void brook_cpu_tree_query(void *p, float *v, float *q, uint32_t n, float bound, uint32_t *idx,
                          float *dist) {
  search_tree_float_int32_t(p, v, q, n, 1, bound, 0, NULL, idx, dist);
  for (uint32_t i = 0; i < n; ++i)
    dist[i] = dist[i] >= bound ? INFINITY : sqrtf(dist[i]);
}
void *brook_cpu_tree64_create(double *v, uint32_t n) {
  return construct_tree_double_int32_t(v, 3, n, 16);
}
void brook_cpu_tree64_delete(void *p) { delete_tree_double_int32_t(p); }
void brook_cpu_tree64_query(void *p, double *v, double *q, uint32_t n, double bound,
                            uint32_t *idx, double *dist) {
  search_tree_double_int32_t(p, v, q, n, 1, bound, 0, NULL, idx, dist);
  for (uint32_t i = 0; i < n; ++i)
    dist[i] = dist[i] >= bound ? INFINITY : sqrt(dist[i]);
}
