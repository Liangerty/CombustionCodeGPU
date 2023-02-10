#pragma once
#include "Define.h"
#include "DParameter.h"
//#include "gxl_lib/Array.hpp"

namespace cfd{
struct DZone;
class Block;
class InviscidScheme;
class ViscousScheme;
struct TemporalScheme;

__global__ void store_last_step(DZone* zone);

void compute_inviscid_flux(const Block &block, cfd::DZone *zone, InviscidScheme **inviscid_scheme, DParameter *param,
                           integer n_var);

__global__ void
inviscid_flux_1d(cfd::DZone *zone, InviscidScheme **inviscid_scheme, integer direction, integer max_extent,
                 cfd::DParameter *param);

void compute_viscous_flux(const Block &block, cfd::DZone *zone, ViscousScheme **viscous_scheme, DParameter *param,
                          integer n_var);

__global__ void
viscous_flux_fv(cfd::DZone *zone, cfd::ViscousScheme **viscous_scheme, integer max_extent, cfd::DParameter *param);
__global__ void
viscous_flux_gv(cfd::DZone *zone, cfd::ViscousScheme **viscous_scheme, integer max_extent, cfd::DParameter *param);
__global__ void
viscous_flux_hv(cfd::DZone *zone, cfd::ViscousScheme **viscous_scheme, integer max_extent, cfd::DParameter *param);

__global__ void local_time_step(cfd::DZone *zone, DParameter *param, TemporalScheme **temporal_scheme);

__global__ void update_cv_and_bv(cfd::DZone *zone, DParameter *param);
}
