#include "Driver.h"
#include "Define.h"
#include "DParameter.h"
#include "Initialize.h"
#include "InviscidScheme.cuh"
#include "ViscousScheme.cuh"
#include "Thermo.cuh"
#include "fmt/core.h"
#include "TimeAdvanceFunc.cuh"
#include "TemporalScheme.cuh"

#if MULTISPECIES == 1
#else
#include "Constants.h"
#endif

cfd::Driver::Driver(Parameter &parameter, Mesh &mesh_
#if MULTISPECIES == 1
    , ChemData &chem_data
#endif
) : myid(parameter.get_int("myid")), mesh(mesh_), parameter(parameter), bound_cond() {
  // Allocate the memory for every block
  for (integer blk = 0; blk < mesh.n_block; ++blk) {
    field.emplace_back(parameter, mesh[blk]);
  }

  initialize_basic_variables(parameter, mesh, field
#if MULTISPECIES == 1
      , chem_data
#endif
  );

  // The following code is used for GPU memory allocation
#ifdef GPU
  DParameter d_param(parameter
#if MULTISPECIES == 1
      , chem_data
#endif
  );
  cudaMalloc(&param, sizeof(DParameter));
  cudaMemcpy(param, &d_param, sizeof(DParameter), cudaMemcpyHostToDevice);
  for (integer blk = 0; blk < mesh.n_block; ++blk) {
    field[blk].setup_device_memory(parameter, mesh[blk]);
  }
  bound_cond.initialize_bc_on_GPU(mesh_, field
#if MULTISPECIES == 1
      , chem_data.spec
#endif
  );
  cudaMalloc(&inviscid_scheme, sizeof(InviscidScheme *));
  cudaMalloc(&viscous_scheme, sizeof(ViscousScheme *));
  cudaMalloc(&temporal_scheme, sizeof(TemporalScheme *));

  setup_schemes<<<1, 1>>>(inviscid_scheme, viscous_scheme, temporal_scheme, param);
#endif
}

void cfd::Driver::initialize_computation() {
  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  const auto ng_1 = 2 * mesh[0].ngg - 1;

  // First, compute the conservative variables from basic variables
  for (auto i = 0; i < mesh.n_block; ++i) {
    integer mx{mesh[i].mx}, my{mesh[i].my}, mz{mesh[i].mz};
    dim3 bpg{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    compute_cv_from_bv<<<bpg, tpb>>>(field[i].d_ptr, param);
  }

  // Second, apply boundary conditions to all boundaries, including face communication between faces
  bound_cond.apply_boundary_conditions(mesh, field, param);
  cudaDeviceSynchronize();
  if (myid == 0) {
    fmt::print("Boundary conditions are applied successfully for initialization\n");
  }

  // Third, communicate values between processes
  data_communication();
  // Currently not implemented, thus the current program can only be used on a single GPU

  if (myid == 0) {
    fmt::print("Finish data transfer.\n");
  }

  for (auto b = 0; b < mesh.n_block; ++b) {
    integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    update_physical_properties<<<bpg, tpb>>>(field[b].d_ptr, param);
  }
  cudaDeviceSynchronize();
  if (myid == 0) {
    fmt::print("The flowfield is completely initialized on GPU.\n");
  }
}

void cfd::Driver::simulate() {
  const auto steady{parameter.get_bool("steady")};
  if (steady) {
    steady_simulation();
  } else {
    const auto temporal_tag{parameter.get_int("temporal_scheme")};
    switch (temporal_tag) {
      case 11:
      case 12:
      default:fmt::print("Not implemented");
    }
  }
}

void cfd::Driver::steady_simulation() {
  fmt::print("Steady flow simulation.\n");
  bool converged{false};
  integer step{0};
  integer total_step{parameter.get_int("total_step")};
  const integer n_block{mesh.n_block};
  const integer n_var{parameter.get_int("n_var")};
  const integer ngg{mesh[0].ngg};
  const integer ng_1=2*ngg-1;

  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  dim3* bpg=new dim3 [n_block];
  for (integer b=0;b<n_block;++b){
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    bpg[b]={(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
  }

  //  const integer file_step{parameter.get_int("output_file")};
  while (!converged) {
    ++step;
    /*[[unlikely]]*/if (step > total_step) {
      break;
    }

    // Start a single iteration

    // First, store the value of last step
    for (auto b = 0; b < n_block; ++b) {
      store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
    }

    // Second, for each block, compute the residual dq
    for (auto b = 0; b < n_block; ++b) {
      compute_inviscid_flux(mesh[b], field[b].d_ptr, inviscid_scheme, param, n_var);
      compute_viscous_flux(mesh[b],field[b].d_ptr,viscous_scheme,param,n_var);

      // compute local time step
      local_time_step<<<bpg[b],tpb>>>(field[b].d_ptr,param,temporal_scheme);
      // implicit treatment if needed

      // update conservative and basic variables
      update_cv_and_bv<<<bpg[b],tpb>>>(field[b].d_ptr,param);

      // apply boundary conditions
      bound_cond.apply_boundary_conditions(mesh,field,param);
    }
    // Third, transfer data between and within processes
    data_communication();

    if (mesh.dimension==2){
      for (auto b = 0; b < n_block; ++b) {
        const auto mx{mesh[b].mx}, my{mesh[b].my};
        dim3 BPG{(mx+ng_1)/tpb.x+1,(my+ng_1)/tpb.y+1,1};
        eliminate_k_gradient<<<BPG, tpb>>>(field[b].d_ptr);
      }
    }

    // update physical properties such as Mach number, transport coefficients et, al.
    for (auto b = 0; b < mesh.n_block; ++b) {
      integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
      dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
      update_physical_properties<<<BPG, tpb>>>(field[b].d_ptr, param);
    }

    // Finally, test if the simulation reaches convergence state

    cudaDeviceSynchronize();
    fmt::print("Step {}\n",step);
  }
  delete[] bpg;
}

void cfd::Driver::data_communication() {
  // -1 - inner faces
  for (auto blk=0;blk<mesh.n_block;++blk){
    auto& inF=mesh[blk].inner_face;
    const auto n_innFace=inF.size();
    auto v=field[blk].d_ptr;
    const auto ngg = mesh[blk].ngg;
    for (auto l=0;l<n_innFace;++l){
      // reference to the current face
      const auto& fc = mesh[blk].inner_face[l];
      uint tpb[3], bpg[3], n_point[3];
      for (size_t j=0;j<3;++j){
        n_point[j]= abs(fc.range_start[j]-fc.range_end[j])+1;
        tpb[j]=n_point[j]<=(2*ngg+1)?1:16;
        bpg[j]=(n_point[j]-1)/tpb[j]+1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};

      // variables of the neighbor block
      auto nv = field[fc.target_block].d_ptr;
      inner_communication<<<BPG, TPB>>>(v, nv, n_point,l);
    }
  }
}

__global__ void cfd::setup_schemes(cfd::InviscidScheme **inviscid_scheme, cfd::ViscousScheme **viscous_scheme,
                                   cfd::TemporalScheme **temporal_scheme, cfd::DParameter *param) {
  const integer inviscid_tag{param->inviscid_scheme};
  switch (inviscid_tag) {
    case 3:*inviscid_scheme = new AUSMP(param);
      printf("Inviscid scheme: AUSM+\n");
      break;
    default:*inviscid_scheme = new AUSMP(param);
      printf("No such scheme. Set to AUSM+ scheme\n");
  }

  const integer viscous_tag{param->viscous_scheme};
  switch (viscous_tag) {
    case 2:*viscous_scheme = new SecOrdViscScheme;
      printf("Viscous scheme: 2nd order central difference\n");
      break;
    default:*viscous_scheme = new ViscousScheme;
      printf("Inviscid computaion\n");
  }

  const integer temporal_tag{param->temporal_scheme};
  switch (temporal_tag) {
    case 0:
      *temporal_scheme=new SteadyTemporalScheme;
      printf("Temporal scheme: 1st order explicit Euler\n");
      break;
    case 1:
      *temporal_scheme=new LUSGS;
      printf("Temporal scheme: Implicit LUSGS\n");
      break;
    default:
      *temporal_scheme=new SteadyTemporalScheme;
      printf("Temporal scheme: 1st order explicit Euler\n");
  }
}
