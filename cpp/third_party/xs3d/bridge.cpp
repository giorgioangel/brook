/* Brook C bridge to xs3d 1.13.0, LGPL-3.0-or-later. */
#include <tuple>
#include "xs3d.hpp"
#include "bridge.h"
#define EXPORT __attribute__((visibility("default")))
extern "C" {
EXPORT void brook_xs3d_set_shape(uint64_t sx,uint64_t sy,uint64_t sz) { xs3d::set_shape(sx,sy,sz); }
EXPORT void brook_xs3d_clear_shape(void) { xs3d::clear_shape(); }
EXPORT float brook_xs3d_area(const bool *mask,const uint64_t s[3],const float p[3],
    const float n[3],const float w[3],uint8_t *contact) {
  auto value=xs3d::cross_sectional_area(mask,true,s[0],s[1],s[2],p[0],p[1],p[2],n[0],n[1],n[2],w[0],w[1],w[2],true);
  *contact=std::get<1>(value);return std::get<0>(value);
}
EXPORT float brook_xs3d_area_u8(const bool *mask,const uint64_t s[3],const float p[3],
    const float n[3],const float w[3],uint8_t *contact) {
  static_assert(sizeof(bool)==sizeof(uint8_t));
  // Reading object representation through unsigned char is explicitly permitted.
  auto value=xs3d::cross_sectional_area(reinterpret_cast<const uint8_t*>(mask),uint8_t(1),s[0],s[1],s[2],p[0],p[1],p[2],n[0],n[1],n[2],w[0],w[1],w[2],true);
  *contact=std::get<1>(value);return std::get<0>(value);
}
EXPORT void brook_xs3d_section(const bool *mask,const uint64_t s[3],const float p[3],
    const float n[3],const float w[3],float *out) {
  xs3d::cross_section(reinterpret_cast<const uint8_t*>(mask),s[0],s[1],s[2],p[0],p[1],p[2],n[0],n[1],n[2],w[0],w[1],w[2],out);
}
}
