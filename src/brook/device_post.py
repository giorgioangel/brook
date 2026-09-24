# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Giorgio Angelotti
"""GPU postprocessing of `PackedSkeletons`: dust, loops, joins and ticks, with Brook's tie rules."""
from ._device_post import (
    components,
    join_close_components,
    postprocess,
    remove_dust,
    remove_loops,
    remove_ticks,
)

__all__ = ["components", "join_close_components", "postprocess", "remove_dust", "remove_loops", "remove_ticks"]
