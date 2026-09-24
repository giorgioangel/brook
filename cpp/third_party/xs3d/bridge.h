/* Brook C bridge to xs3d 1.13.0, LGPL-3.0-or-later. */
#pragma once
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
void brook_xs3d_set_shape(uint64_t sx,uint64_t sy,uint64_t sz);
void brook_xs3d_clear_shape(void);
float brook_xs3d_area(const bool *mask,const uint64_t shape[3],const float position[3],
    const float normal[3],const float spacing[3],uint8_t *contact);
float brook_xs3d_area_u8(const bool *mask,const uint64_t shape[3],const float position[3],
    const float normal[3],const float spacing[3],uint8_t *contact);
void brook_xs3d_section(const bool *mask,const uint64_t shape[3],const float position[3],
    const float normal[3],const float spacing[3],float *output);
#ifdef __cplusplus
}
#endif
