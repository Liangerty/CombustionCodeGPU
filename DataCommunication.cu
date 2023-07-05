#include "DataCommunication.cuh"
#include "Field.h"
#include "Constants.h"

//template<MixtureModel mix_model, TurbMethod turb_method>
//__global__ void cfd::inner_communication(DZone *zone, DZone *tar_zone, integer i_face) {
//  const auto &f = zone->innerface[i_face];
//  uint n[3];
//  n[0] = blockIdx.x * blockDim.x + threadIdx.x;
//  n[1] = blockDim.y * blockIdx.y + threadIdx.y;
//  n[2] = blockIdx.z * blockDim.z + threadIdx.z;
//  if (n[0] >= f.n_point[0] || n[1] >= f.n_point[1] || n[2] >= f.n_point[2]) return;
//
//  integer idx[3], idx_tar[3], d_idx[3];
//  for (integer i = 0; i < 3; ++i) {
//    d_idx[i] = f.loop_dir[i] * (integer) (n[i]);
//    idx[i] = f.range_start[i] + d_idx[i];
//  }
//  for (integer i = 0; i < 3; ++i) {
//    idx_tar[i] = f.target_start[i] + f.target_loop_dir[i] * d_idx[f.src_tar[i]];
//  }
//
//  // The face direction: which of i(0)/j(1)/k(2) is the coincided face.
//  const auto face_dir{f.direction > 0 ? f.range_start[f.face] : f.range_end[f.face]};
//
//  if (idx[f.face] == face_dir) {
//    // If this is the corresponding face, then average the values from both blocks
////    for (integer l = 0; l < 6; ++l) {
////      const real ave =
////          0.5 * (tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], l) + zone->bv(idx[0], idx[1], idx[2], l));
////      zone->bv(idx[0], idx[1], idx[2], l) = ave;
////      tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave;
////    }
//    for (int l = 0; l < zone->n_var; ++l) {
//      const real ave =
//          0.5 * (tar_zone->cv(idx_tar[0], idx_tar[1], idx_tar[2], l) + zone->cv(idx[0], idx[1], idx[2], l));
//      zone->cv(idx[0], idx[1], idx[2], l) = ave;
//      tar_zone->cv(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave;
//    }
//    update_bv_1_point(zone,param,idx[0], idx[1], idx[2]);
////    for (int l = 0; l < zone->n_scal; ++l) {
////      // Be Careful! The flamelet case is different from here, should be pay extra attention!
////      real ave = 0.5 * (tar_zone->sv(idx_tar[0], idx_tar[1], idx_tar[2], l) + zone->sv(idx[0], idx[1], idx[2], l));
////      zone->sv(idx[0], idx[1], idx[2], l) = ave;
////      tar_zone->sv(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave;
////    }
//  } else {
//    // Else, get the inner value for this block's ghost grid
//    for (int l = 0; l < 6; ++l) {
//      zone->bv(idx[0], idx[1], idx[2], l) = tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], l);
//    }
//    for (int l = 0; l < zone->n_scal; ++l) {
//      // Be Careful! The flamelet case is different from here, should be pay extra attention!
//      zone->sv(idx[0], idx[1], idx[2], l) = tar_zone->sv(idx_tar[0], idx_tar[1], idx_tar[2], l);
//      zone->cv(idx[0], idx[1], idx[2], l + 5) = tar_zone->cv(idx_tar[0], idx_tar[1], idx_tar[2], l + 5);
//    }
//  }
//}

__global__ void cfd::setup_data_to_be_sent(cfd::DZone *zone, integer i_face, real *data) {
  const auto &f = zone->parface[i_face];
  integer n[3];
  n[0] = blockIdx.x * blockDim.x + threadIdx.x;
  n[1] = blockDim.y * blockIdx.y + threadIdx.y;
  n[2] = blockIdx.z * blockDim.z + threadIdx.z;
  if (n[0] >= f.n_point[0] || n[1] >= f.n_point[1] || n[2] >= f.n_point[2]) return;

  integer idx[3];
  for (int ijk: f.loop_order) {
    idx[ijk] = f.range_start[ijk] + n[ijk] * f.loop_dir[ijk];
  }

  const integer n_var{zone->n_var}, ngg{zone->ngg};
  integer bias = n_var * (ngg + 1) * (n[f.loop_order[1]] * f.n_point[f.loop_order[2]] + n[f.loop_order[2]]);

  const auto &cv = zone->cv;
  for (integer l = 0; l < n_var; ++l) {
    data[bias + l] = cv(idx[0], idx[1], idx[2], l);
  }

  for (integer ig = 1; ig <= ngg; ++ig) {
    idx[f.face] -= f.direction;
    bias += n_var;
    for (integer l = 0; l < n_var; ++l) {
      data[bias + l] = cv(idx[0], idx[1], idx[2], l);
    }
  }
}

