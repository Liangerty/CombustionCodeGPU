#pragma once
#include "Define.h"
//#include "gxl_lib/Array.hpp"

namespace cfd{
struct DZone;
class Block;
class InviscidScheme;

__global__ void store_last_step(DZone* zone);

void compute_inviscid_flux(const Block& block, DZone*zone, InviscidScheme** inviscid_scheme);

__global__ void
inviscid_flux_1d(cfd::DZone *zone, InviscidScheme **inviscid_scheme, integer direction, integer max_extent);
}
