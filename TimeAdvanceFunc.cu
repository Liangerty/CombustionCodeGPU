#include "TimeAdvanceFunc.cuh"
#include "Field.h"
#include "Mesh.h"
#include "InviscidScheme.cuh"

__global__ void cfd::store_last_step(cfd::DZone *zone) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  integer j = blockDim.y * blockIdx.y + threadIdx.y;
  integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  zone->bv_last(i,j,k,0)=zone->bv(i,j,k,0);
  zone->bv_last(i,j,k,1)=zone->vel(i,j,k);
  zone->bv_last(i,j,k,2)=zone->bv(i,j,k,4);
  zone->bv_last(i,j,k,3)=zone->bv(i,j,k,5);
}

void cfd::compute_inviscid_flux(const Block& block, cfd::DZone *zone, InviscidScheme** inviscid_scheme) {
  const integer extent[3]{block.mx, block.my, block.mz};
  const integer ngg{block.ngg};
  const integer dim{extent[2]==1?2:3};

  for (auto dir=0;dir<dim;++dir){
    integer tpb[3]{1,1,1};
    tpb[dir]=64;
    // uint bpg[3]{(extent[0]-1)/tpb[0]+1,(extent[1]-1)/tpb[1]+1,(extent[2]-1)/tpb[2]+1};
    integer bpg[3]{extent[0],extent[1],extent[2]};
    bpg[dir]=extent[dir]/tpb[dir]+1;

    dim3 TPB(tpb[0],tpb[1],tpb[2]);
    dim3 BPG(bpg[0],bpg[1],bpg[2]);
    inviscid_flux_1d<<<BPG, TPB>>>(zone, inviscid_scheme, dir, extent[dir]);
  }
}

__global__ void
cfd::inviscid_flux_1d(cfd::DZone *zone, InviscidScheme **inviscid_scheme, integer direction, integer max_extent) {
  integer idx[3];
  idx[0]=blockDim.x*blockIdx.x+threadIdx.x;
  idx[1]=blockDim.y*blockIdx.y+threadIdx.y;
  idx[2]=blockDim.z*blockIdx.z+threadIdx.z;
  idx[direction]-=1;
  if (idx[direction]>=max_extent) return;

  const auto ngg{zone->ngg};
}
