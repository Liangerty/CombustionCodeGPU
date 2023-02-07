#pragma once
#include "Define.h"
#include "DParameter.h"
//#include "gxl_lib/Array.hpp"

namespace cfd{
struct DZone;
class Block;
class InviscidScheme;

__global__ void store_last_step(DZone* zone);

void compute_inviscid_flux(const Block &block, cfd::DZone *zone, InviscidScheme **inviscid_scheme, DParameter *param,
                           const integer n_var);

__global__ void
inviscid_flux_1d(cfd::DZone *zone, InviscidScheme **inviscid_scheme, integer direction, integer max_extent,
                 cfd::DParameter *param);
}
