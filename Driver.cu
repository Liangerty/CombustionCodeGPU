#include "Driver.h"
#include "Define.h"
#include "DParameter.h"
#include "Initialize.h"
#include "InviscidScheme.cuh"
#include "ViscousScheme.cuh"
#include "fmt/core.h"
#include "TimeAdvanceFunc.cuh"
#include "TemporalScheme.cuh"
#include <fstream>
#if MULTISPECIES == 1
#include "Thermo.cuh"
#else
#include "Constants.h"
#endif

cfd::Driver::Driver(Parameter &parameter, Mesh &mesh_
#if MULTISPECIES == 1
    , ChemData &chem_data
#endif
) : myid(parameter.get_int("myid")), time(), mesh(mesh_), parameter(parameter), bound_cond() {
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
//  cudaMalloc(&d_res,4*sizeof (real));
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
  const integer ng_1 = 2 * ngg - 1;
  const integer output_screen = parameter.get_int("output_screen");

  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  dim3 *bpg = new dim3[n_block];
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    bpg[b] = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
  }

  //  const integer file_step{parameter.get_int("output_file")};
  while (!converged) {
    ++step;
    /*[[unlikely]]*/if (step > total_step) {
      break;
    }

    // Start a single iteration
    // First, store the value of last step
    if (step % output_screen == 0)
      for (auto b = 0; b < n_block; ++b)
        store_last_step <<<bpg[b], tpb >>>(field[b].d_ptr);

    for (auto b = 0; b < n_block; ++b) {
      set_dq_to_0 <<<bpg[b], tpb >>>(field[b].d_ptr);

      // Second, for each block, compute the residual dq
      compute_inviscid_flux(mesh[b], field[b].d_ptr, inviscid_scheme, param, n_var);
      compute_viscous_flux(mesh[b], field[b].d_ptr, viscous_scheme, param, n_var);

      // compute local time step
      local_time_step<<<bpg[b], tpb>>>(field[b].d_ptr, param, temporal_scheme);
      // implicit treatment if needed

      // update conservative and basic variables
      update_cv_and_bv<<<bpg[b], tpb>>>(field[b].d_ptr, param);

      // apply boundary conditions
      bound_cond.apply_boundary_conditions(mesh, field, param);
    }
    // Third, transfer data between and within processes
    data_communication();

    if (mesh.dimension == 2) {
      for (auto b = 0; b < n_block; ++b) {
        const auto mx{mesh[b].mx}, my{mesh[b].my};
        dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
        eliminate_k_gradient<<<BPG, tpb>>>(field[b].d_ptr);
      }
    }

    // update physical properties such as Mach number, transport coefficients et, al.
    for (auto b = 0; b < n_block; ++b) {
      integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
      dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
      update_physical_properties<<<BPG, tpb>>>(field[b].d_ptr, param);
    }

    // Finally, test if the simulation reaches convergence state
    if (step % output_screen == 0) {
      real err_max = compute_residual(step);
      converged=err_max<parameter.get_real("convergence_criteria")? true: false;
      if (myid == 0)
        steady_screen_output(step, err_max);
    }

    cudaDeviceSynchronize();
  }
  delete[] bpg;
}

void cfd::Driver::data_communication() {
  // -1 - inner faces
  for (auto blk = 0; blk < mesh.n_block; ++blk) {
    auto &inF = mesh[blk].inner_face;
    const auto n_innFace = inF.size();
    auto v = field[blk].d_ptr;
    const auto ngg = mesh[blk].ngg;
    for (auto l = 0; l < n_innFace; ++l) {
      // reference to the current face
      const auto &fc = mesh[blk].inner_face[l];
      uint tpb[3], bpg[3], n_point[3];
      for (size_t j = 0; j < 3; ++j) {
        n_point[j] = abs(fc.range_start[j] - fc.range_end[j]) + 1;
        tpb[j] = n_point[j] <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point[j] - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};

      // variables of the neighbor block
      auto nv = field[fc.target_block].d_ptr;
      inner_communication<<<BPG, TPB>>>(v, nv, n_point, l);
    }
  }
}

real cfd::Driver::compute_residual(integer step) {
  const integer n_block{mesh.n_block};
  for (auto &e: res)
    e = 0;

  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
    // compute the square of the difference of the basic variables
    compute_square_of_dbv<<<bpg, tpb>>>(field[b].d_ptr);
  }

  constexpr integer TPB{128};
  constexpr integer n_res_var{4};
  real res_block[n_res_var];
  int num_sms, num_blocks_per_sm;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_of_dv_squared<n_res_var>, TPB,
                                                TPB * sizeof(real) * n_res_var);
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    const integer size = mx * my * mz;
    int n_blocks = std::min(num_blocks_per_sm * num_sms, (size + TPB - 1) / TPB);
    reduction_of_dv_squared<n_res_var> <<<n_blocks, TPB, TPB * sizeof(real) * n_res_var >>>(
        field[b].h_ptr->bv_last.data(), size);
    reduction_of_dv_squared<n_res_var> <<<1, TPB, TPB * sizeof(real) * n_res_var >>>(field[b].h_ptr->bv_last.data(),
                                                                                     n_blocks);
    cudaMemcpy(res_block, field[b].h_ptr->bv_last.data(), n_res_var * sizeof(real), cudaMemcpyDeviceToHost);
    for (integer l = 0; l < n_res_var; ++l)
      res[l] += res_block[l];
  }

  if (parameter.get_bool("parallel")) {
    // Parallel reduction
  }
  for (auto &e: res)
    e = std::sqrt(e / mesh.n_grid_total);

  if (step == parameter.get_int("output_screen")) {
    for (integer i = 0; i < n_res_var; ++i) {
      res_scale[i] = res[i];
      if (res_scale[i] < 1e-20)
        res_scale[i] = 1e-20;
    }
  }

  for (integer i = 0; i < 4; ++i)
    res[i] /= res_scale[i];

  // Find the maximum error of the 4 errors
  real err_max = res[0];
  for (integer i = 1; i < 4; ++i) {
    if (res[i] > err_max)
      err_max = res[i];
  }
  return err_max;
}

void cfd::Driver::steady_screen_output(integer step, real err_max) {
  time.get_elapsed_time();
  std::ofstream history("history.dat", std::ios::app);
  history << fmt::format("{}\t{}\n", step, err_max);
  history.close();

  fmt::print("\n{:>38}    converged to: {:>11.4e}\n", "rho", res[0]);
  fmt::print("  n={:>8},                       V     converged to: {:>11.4e}   \n", step, res[1]);
  fmt::print("  n={:>8},                       p     converged to: {:>11.4e}   \n", step, res[2]);
  fmt::print("{:>38}    converged to: {:>11.4e}\n", "T ", res[3]);
  fmt::print("CPU time for this step is {:>16.8f}s\n", time.step_time);
  fmt::print("Total elapsed CPU time is {:>16.8f}s\n", time.elapsed_time);
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
    case 0:*temporal_scheme = new SteadyTemporalScheme;
      printf("Temporal scheme: 1st order explicit Euler\n");
      break;
    case 1:*temporal_scheme = new LUSGS;
      printf("Temporal scheme: Implicit LUSGS\n");
      break;
    default:*temporal_scheme = new SteadyTemporalScheme;
      printf("Temporal scheme: 1st order explicit Euler\n");
  }
}

template<integer N>
__global__ void cfd::reduction_of_dv_squared(real *arr, integer size) {
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  const integer t = threadIdx.x;
  extern __shared__ real s[];
  memset(&s[t * N], 0, N * sizeof(real));
  if (i >= size)
    return;
  real inp[N];
  memset(inp, 0, N * sizeof(real));
  for (integer idx = i; idx < size; idx += blockDim.x * gridDim.x) {
    inp[0] += arr[idx];
    inp[1] += arr[idx + size];
    inp[2] += arr[idx + size * 2];
    inp[3] += arr[idx + size * 3];
  }
  for (integer l = 0; l < N; ++l)
    s[t * N + l] = inp[l];
  __syncthreads();

  for (int stride = blockDim.x / 2, lst = blockDim.x & 1; stride >= 1; lst = stride & 1, stride >>= 1) {
    stride += lst;
    __syncthreads();
    if (t <stride) {//when t+stride is larger than #elements, there's no meaning of comparison. So when it happens, just keep the current value for parMax[t]. This always happens when an odd number of t satisfying the condition.
      if (t + stride < size) {
        for (integer l = 0; l < N; ++l)
          s[t * N + l] += s[(t + stride) * N + l];
      }
    }
    __syncthreads();
  }

  if (t == 0) {
    arr[blockIdx.x] = s[0];
    arr[blockIdx.x + gridDim.x] = s[1];
    arr[blockIdx.x + gridDim.x * 2] = s[2];
    arr[blockIdx.x + gridDim.x * 3] = s[3];
  }
}
