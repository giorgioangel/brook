#include <brook/brook.hpp>
#include <stdexcept>
#define CHECK(x)                                                                                   \
  do {                                                                                             \
    if (!(x))                                                                                      \
      throw std::runtime_error(#x);                                                                \
  } while (false)
#include <cmath>
#include <vector>
int main() {
  float vertices[] = {0, 0, 0, 1, 0, 0, 5, 0, 0, 11, 0, 0};
  uint32_t edges[] = {0, 1, 2, 3};
  float radii[] = {100, 100, 100, 100};
  int64_t labels[] = {1337}, vo[] = {0, 4}, eo[] = {0, 2};
  brook_skeletons_view input{1, 4, 2, labels, vo, eo, vertices, edges, radii, {1, 1, 1}};
  auto joined = brook::api::join_cpu(input);
  auto v = joined.view();
  CHECK(v.skeleton_count == 1 && v.vertex_count == 4 && v.edge_count == 3 && v.labels[0] == 1337);
  const uint32_t expected[] = {0, 1, 1, 2, 2, 3};
  for (int i = 0; i < 6; ++i)
    CHECK(v.edges[i] == expected[i]);
  auto sources = joined.sources();
  CHECK(sources.second == 4);
  for (int i = 0; i < 4; ++i)
    CHECK(sources.first[i] == uint64_t(i));
  auto result = brook::api::postprocess_cpu(input, 0, 0);
  v = result.view();
  CHECK(v.edge_count == 3);
  auto gone = brook::api::postprocess_cpu(input, 1000, 0);
  v = gone.view();
  CHECK(v.skeleton_count == 1 && v.vertex_count == 0 && v.labels[0] == 1337);
  brook_skeletons *bad = nullptr;
  CHECK(brook_join_cpu(&input, 0, 0, &bad) == BROOK_INVALID_ARGUMENT && bad == nullptr);
  edges[3] = 4;
  CHECK(brook_join_cpu(&input, INFINITY, 0, &bad) == BROOK_INVALID_ARGUMENT && bad == nullptr);
}
