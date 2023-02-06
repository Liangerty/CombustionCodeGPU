#include "InviscidScheme.cuh"
#include "DParameter.h"
#include "Field.h"

namespace cfd {
__device__ InviscidScheme::InviscidScheme(DParameter *param) {
  integer reconstruct_tag = param->reconstruction;
  switch (reconstruct_tag) {
    case 2:
      reconstruction = new MUSCL(param);
      break;
    default:
      reconstruction = new Reconstruction(param);
      break;
  }
}

__device__ AUSMP::AUSMP(DParameter *param) : InviscidScheme(param) {

}

__device__ void AUSMP::compute_inviscid_flux(DZone *zone) {
  auto n_reconstruction=6; // rho,u,v,w,p,E
#if MULTISPECIES==1
  n_reconstruction+=zone->n_spec+1; // gamma, Y_{1...Ns}
#endif
  real* bv_l=new real[n_reconstruction];
  real* bv_r=new real[n_reconstruction];

  delete[]bv_l;
  delete[]bv_r;
}
} // cfd