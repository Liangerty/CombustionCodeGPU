#include "BoundCond.h"
#include "gxl_lib/MyString.h"
#include "Transport.cuh"
#include "BoundCond.cuh"
#include <map>
#include "fmt/format.h"
#include "Field.h"
#include "Thermo.cuh"
#include "DParameter.h"

#ifdef GPU
namespace cfd {
cfd::Inflow::Inflow(integer type_label, std::ifstream &file
#if MULTISPECIES == 1
    , Species &spec
#endif
) : label{type_label} {
  std::map<std::string, real> par;
  std::map<std::string, real> spec_inflow;
  std::string input{}, key{}, name{};
  real val{};
  std::istringstream line(input);
  while (gxl::getline_to_stream(file, input, line, gxl::Case::lower)) {
    line >> key;
    if (key == "double") {
      line >> name >> key >> val;
      par.emplace(std::make_pair(name, val));
      continue;
    }
    if (key == "species") {
      line >> name >> key >> val;
      spec_inflow.emplace(std::make_pair(gxl::to_upper(name), val));
      continue;
    }
    if (key == "label" || key == "end") break;
  }
  // In default, the mach number, pressure and temperature should be given.
  // If other combinations are given, then implement it later.
  // Currently, 2 combinations are achieved. One is to give (mach, pressure, temperature)
  // The other is to give (density, velocity, pressure)
  if (par.find("mach") != par.cend()) mach = par["mach"];
  if (par.find("pressure") != par.cend()) pressure = par["pressure"];
  if (par.find("temperature") != par.cend()) temperature = par["temperature"];
  if (par.find("velocity") != par.cend()) velocity = par["velocity"];
  if (par.find("density") != par.cend()) density = par["density"];
  if (par.find("u") != par.cend()) u = par["u"];
  if (par.find("v") != par.cend()) v = par["v"];
  if (par.find("w") != par.cend()) w = par["w"];
#if MULTISPECIES == 1
  const integer n_spec{spec.n_spec};
  cudaMalloc(&yk, n_spec * sizeof(real));
  std::vector<real> hY(n_spec, 0);
  //Assign the species mass fraction to the corresponding position.
  //Should be done after knowing the order of species.
  for (auto &[sp, y]: spec_inflow) {
    if (spec.spec_list.find(sp) != spec.spec_list.cend()) {
      const int idx = spec.spec_list.at(sp);
      hY[idx] = y;
    }
  }
  mw = 0;
  for (int i = 0; i < n_spec; ++i) {
    mw += hY[i] / spec.mw[i];
  }
  mw = 1 / mw;
  cudaMemcpy(yk, hY.data(), n_spec * sizeof(real), cudaMemcpyHostToDevice);
#endif // MULTISPECIES==1
  if (temperature < 0) {
    // The temperature is not given, thus the density and pressure are given
    temperature = pressure * mw / (density * cfd::R_u);
  }
  viscosity = cfd::Sutherland(temperature);

  real gamma{gamma_air}; // specific heat ratio
  real cp{0}, cv{0};
#if MULTISPECIES == 1
  std::vector<real> cpi(n_spec, 0);
  spec.compute_cp(temperature, cpi.data());
  for (size_t i = 0; i < n_spec; ++i) {
    cp += hY[i] * cpi[i];
    cv += hY[i] * (cpi[i] - R_u / spec.mw[i]);
  }
  gamma = cp / cv;
#endif // MULTISPECIES==1
  const real c{std::sqrt(gamma * cfd::R_u / mw * temperature)}; //speed of sound

  if (mach < 0) {
    // The mach number is not given. The velocity magnitude should be given
    mach = velocity / c;
  } else {
    // The velocity magnitude is not given. The mach number should be given
    velocity = mach * c;
  }
  u *= velocity;
  v *= velocity;
  w *= velocity;
  if (density < 0) {
    // The density is not given, compute it from equation of state
    density = pressure * mw / (cfd::R_u * temperature);
  }
#if MULTISPECIES == 1
  viscosity = compute_viscosity(temperature, mw, hY.data(), spec);
#endif // MULTISPECIES==1
//  reynolds = density * velocity * parameter.get_real("referenceLength") / viscosity;
}

template<typename BCType>
void register_bc(BCType *&bc, int n_bc, std::vector<integer> &indices, BCInfo *&bc_info
#if MULTISPECIES == 1
    , Species &species
#endif
) {
  if (!(n_bc > 0)) {
    return;
  }
  cudaMalloc(&bc, n_bc * sizeof(BCType));
  bc_info = new BCInfo[n_bc];
  integer counter = 0;
  while (counter < n_bc) {
    BCType bctemp(indices[counter]);
    bc_info[counter].label = indices[counter];
    cudaMemcpy(&(bc[counter]), &bctemp, sizeof(BCType), cudaMemcpyHostToDevice);
    ++counter;
  }
}

template<>
void register_bc<Wall>(Wall *&walls, integer n_bc, std::vector<integer> &indices, BCInfo *&bc_info
#if MULTISPECIES == 1
    , Species &species
#endif
) {
  if (!(n_bc > 0)) {
    return;
  }

  cudaMalloc(&walls, n_bc * sizeof(Wall));
  bc_info = new BCInfo[n_bc];
  std::ifstream bc_file("./input_files/setup/7_wall.txt");
  integer counter = 0;
  std::string input{}, type{};
  integer bc_label{};
  std::istringstream line(input);
  gxl::read_until(bc_file, input, "label", gxl::Case::lower);
  while (counter < n_bc) {
    if (input.rfind("end", 0) == 0) {//input.starts_with("end")
      break;
    }
    gxl::to_stringstream(input, line);
    line >> type >> bc_label;
    integer idx = 0;
    for (integer i = 0; i < n_bc; ++i) {
      if (indices[i] == bc_label) {
        idx = i;
        break;
      }
    }
    Wall wall(bc_label, bc_file);
    bc_info[idx].label = bc_label;
    cudaMemcpy(&(walls[idx]), &wall, sizeof(Wall), cudaMemcpyHostToDevice);
    ++counter;
  }
  bc_file.close();
}

template<>
void register_bc<Inflow>(Inflow *&inflows, integer n_bc, std::vector<integer> &indices, BCInfo *&bc_info
#if MULTISPECIES == 1
    , Species &species
#endif
) {
  if (!(n_bc > 0)) {
    return;
  }

  cudaMalloc(&inflows, n_bc * sizeof(Inflow));
  bc_info = new BCInfo[n_bc];
  std::ifstream bc_file("./input_files/setup/6_inflow.txt");
  int counter{0};
  std::string input{}, type{};
  integer bc_label{};
  std::istringstream line(input);
  gxl::read_until(bc_file, input, "label", gxl::Case::lower);
  while (counter < n_bc) {
    if (input.rfind("end", 0) == 0) {//input.starts_with("end")
      break;
    }
    gxl::to_stringstream(input, line);
    line >> type >> bc_label;
    integer idx = 0;
    for (integer i = 0; i < n_bc; ++i) {
      if (indices[i] == bc_label) {
        idx = i;
        break;
      }
    }
    bc_info[idx].label = bc_label;
    Inflow inflow(bc_label, bc_file
#if MULTISPECIES == 1
        , species
#endif
    );
    cudaMemcpy(&(inflows[idx]), &inflow, sizeof(Inflow), cudaMemcpyHostToDevice);
    ++counter;
  }
  bc_file.close();
}

void DBoundCond::link_bc_to_boundaries(Mesh &mesh, std::vector<Field>& field) const {
  const integer n_block{mesh.n_block};
  auto *i_wall = new integer[n_block];
  auto *i_inflow = new integer[n_block];
  auto *i_outflow = new integer[n_block];
  for (integer i = 0; i < n_block; i++) {
    i_wall[i] = 0;
    i_inflow[i] = 0;
    i_outflow[i] = 0;
  }

  // We first count how many faces corresponds to a given boundary condition
  for (integer i = 0; i < n_block; i++) {
    count_boundary_of_type_bc(mesh[i].boundary, n_wall, i_wall, i, n_block, wall_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_inflow, i_inflow, i, n_block, inflow_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_outflow, i_outflow, i, n_block, outflow_info);
  }
  const auto ngg{mesh[0].ngg};
  for (auto i = 0; i < n_block; i++) {
    link_boundary_and_condition(mesh[i].boundary, wall_info, n_wall, i_wall, i);
    for (size_t l = 0; l < n_wall; l++) {
      const auto nb = wall_info[l].n_boundary;
      for (size_t m = 0; m < nb; m++) {
        auto &b = mesh[i].boundary[wall_info[l].boundary[m].y];
        for (int q = 0; q < 3; ++q) {
          if (q == b.face) continue;
          b.range_start[q] += ngg;
          b.range_end[q] -= ngg;
        }
      }
    }
    cudaMemcpy(field[i].h_ptr->boundary, mesh[i].boundary.data(), mesh[i].boundary.size() * sizeof(Boundary),
               cudaMemcpyHostToDevice);
    link_boundary_and_condition(mesh[i].boundary, inflow_info, n_inflow, i_inflow, i);
    link_boundary_and_condition(mesh[i].boundary, outflow_info, n_outflow, i_outflow, i);
  }
  delete[]i_wall;
  delete[]i_inflow;
  delete[]i_outflow;
  fmt::print("Finish setting up boundary conditions.\n");
}

void DBoundCond::apply_boundary_conditions(const Mesh &mesh, std::vector<Field> &field, DParameter *param) const {
  // Boundary conditions are applied in the order of priority, which with higher priority is applied later.
  // Finally, the communication between faces will be carried out after these bc applied
  // Priority: (-1 - inner faces >) 2-wall > 3-symmetry > 5-inflow > 6-outflow > 4-farfield

  // 4-farfield

  // 6-outflow
  for (size_t l = 0; l < n_outflow; l++) {
    const auto nb = outflow_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = outflow_info[l].boundary[i];
      const auto &h_f = mesh[i_zone].boundary[i_face];
      const auto ngg = mesh[i_zone].ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_outflow<<<BPG, TPB>>>(field[i_zone].d_ptr, i_face);
    }
  }

  // 5-inflow
  for (size_t l = 0; l < n_inflow; l++) {
    const auto nb = inflow_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = inflow_info[l].boundary[i];
      const auto &hf = mesh[i_zone].boundary[i_face];
      const auto ngg = mesh[i_zone].ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_inflow<<<BPG, TPB>>>(field[i_zone].d_ptr, &inflow[l], param, i_face);
    }
  }

  // 3-symmetry

  // 2 - wall
  for (size_t l = 0; l < n_wall; l++) {
    const auto nb = wall_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = wall_info[l].boundary[i];
      const auto &hf = mesh[i_zone].boundary[i_face];
      const auto ngg = mesh[i_zone].ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_wall<<<BPG, TPB>>>(field[i_zone].d_ptr, &wall[l], param, i_face);
    }
  }
}

void DBoundCond::initialize_bc_on_GPU(Mesh &mesh, std::vector<Field> &field, Species &species) {
  std::vector<integer> bc_labels;
  // Count the number of distinct boundary conditions
  for (auto i = 0; i < mesh.n_block; i++) {
    for (auto &b: mesh[i].boundary) {
      auto lab = b.type_label;
      bool has_this_bc = false;
      for (auto l: bc_labels) {
        if (l == lab) {
          has_this_bc = true;
          break;
        }
      }
      if (!has_this_bc) {
        bc_labels.push_back(lab);
      }
    }
  }
  // Open the file containing non-default bc info and read them
  std::ifstream bc_file("./input_files/setup/5_boundary_condition.txt");
  std::string input{}, type{}, bc_name{};
  integer bc_label{};
  std::istringstream line(input);
  std::vector<integer> wall_idx, inflow_idx, outflow_idx;
  while (gxl::getline_to_stream(bc_file, input, line, gxl::Case::lower)) {
    line >> type;
    if (input.rfind("//", 0) == 0) {//input.starts_with("//")
      continue;
    }
    if (type == "end") {
      break;
    }
    line >> bc_name >> bc_label;

    auto this_iter = bc_labels.end();
    for (auto iter = bc_labels.begin(); iter != bc_labels.end(); ++iter) {
      if (*iter == bc_label) {
        this_iter = iter;
        break;
      }
    }
    if (this_iter == bc_labels.end()) {
      printf("This boundary condition is not included in the grid boundary file. Ignore boundary condition {%s}\n",
             bc_name.c_str());
    } else {
      bc_labels.erase(this_iter);
      if (type == "wall") {
        wall_idx.push_back(bc_label);
        ++n_wall;
      } else if (type == "inflow") {
        inflow_idx.push_back(bc_label);
        ++n_inflow;
      } else if (type == "outflow") {
        outflow_idx.push_back(bc_label);
        ++n_outflow;
      }
    }
  }
  bc_file.close();
  for (int lab: bc_labels) {
    if (lab == 2) {
      wall_idx.push_back(lab);
      ++n_wall;
    } else if (lab == 5) {
      inflow_idx.push_back(lab);
      ++n_inflow;
    } else if (lab == 6) {
      outflow_idx.push_back(lab);
      ++n_outflow;
    }
  }

  // Read specific conditions
  register_bc<Wall>(wall, n_wall, wall_idx, wall_info
#if MULTISPECIES == 1
      , species
#endif
  );
  register_bc<Inflow>(inflow, n_inflow, inflow_idx, inflow_info
#if MULTISPECIES == 1
      , species
#endif
  );
  register_bc<Outflow>(outflow, n_outflow, outflow_idx, outflow_info
#if MULTISPECIES == 1
      , species
#endif
  );

  link_bc_to_boundaries(mesh,field);
}

void count_boundary_of_type_bc(const std::vector<Boundary> &boundary, integer n_bc, integer *sep, integer blk_idx,
                               integer n_block, BCInfo *bc_info) {
  const auto n_boundary{boundary.size()};
  if (!(n_bc > 0)) {
    return;
  }

  // Count how many faces correspond to the given bc
  integer n{0};
  for (size_t l = 0; l < n_bc; l++) {
    integer label = bc_info[l].label; // This means every bc should have a member "label"
    for (size_t i = 0; i < n_boundary; i++) {
      auto &b = boundary[i];
      if (b.type_label == label) {
        ++bc_info[l].n_boundary;
        ++n;
      }
    }
  }
  if (blk_idx < n_block - 1) {
    sep[blk_idx + 1] = n;
  }
}

void link_boundary_and_condition(const std::vector<Boundary> &boundary, BCInfo *bc, integer n_bc, const integer *sep,
                                 integer i_zone) {
  for (size_t l = 0; l < n_bc; l++) {
    bc[l].boundary = new int2[bc[l].n_boundary];
  }

  const auto n_boundary{boundary.size()};
  for (size_t l = 0; l < n_bc; l++) {
    integer label = bc[l].label;
    int has_read{sep[i_zone]};
    for (auto i = 0; i < n_boundary; i++) {
      auto &b = boundary[i];
      if (b.type_label == label) {
        bc[l].boundary[has_read] = make_int2(i_zone, i);
        ++has_read;
      }
    }
  }
}

__global__ void apply_outflow(DZone *zone, integer i_face) {
  const integer ngg = zone->ngg;
  integer dir[]{0, 0, 0};
  const auto& b=zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &cv = zone->cv;
#if MULTISPECIES == 1
  auto &yk = zone->yk;
#endif // MULTISPECIES==1

  for (integer g = 1; g <= ngg; ++g) {
    const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    for (integer l = 0; l < 6; ++l) {
      bv(gi, gj, gk, l) = bv(i, j, k, l);
    }
    for (integer l = 0; l < zone->n_var; ++l) {
      // All conservative variables including species and scalars are assigned with appropriate value
      cv(gi, gj, gk, l) = cv(i, j, k, l);
    }
#if MULTISPECIES == 1
    for (integer l = 0; l < zone->n_spec; ++l) {
      yk(gi, gj, gk, l) = yk(i, j, k, l);
    }
#endif // MULTISPECIES==1
  }
}

__global__ void apply_inflow(DZone *zone, Inflow *inflow, DParameter *param, integer i_face) {
  const integer ngg = zone->ngg;
  integer dir[]{0, 0, 0};
  const auto& b=zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &cv = zone->cv;
#if MULTISPECIES == 1
  auto &yk = zone->yk;
#endif // MULTISPECIES==1

  for (integer g = 1; g <= ngg; g++) {
    const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    real density = inflow->density;
    bv(gi, gj, gk, 0) = density;
    bv(gi, gj, gk, 1) = inflow->u;
    bv(gi, gj, gk, 2) = inflow->v;
    bv(gi, gj, gk, 3) = inflow->w;
    bv(gi, gj, gk, 4) = inflow->pressure;
    bv(gi, gj, gk, 5) = inflow->temperature;
#if MULTISPECIES == 1
    for (int l = 0; l < zone->n_spec; ++l) {
      yk(gi, gj, gk, l) = inflow->yk[l];
      cv(gi, gj, gk, 5 + l) = density * inflow->yk[l];
    }
#endif // MULTISPECIES==1
    cv(gi, gj, gk, 0) = density;
    cv(gi, gj, gk, 1) = density * inflow->u;
    cv(gi, gj, gk, 2) = density * inflow->v;
    cv(gi, gj, gk, 3) = density * inflow->w;
    compute_total_energy(gi, gj, gk, zone, param);
  }
}

__global__ void apply_wall(DZone *zone, Wall *wall, DParameter *param, integer i_face) {
  const integer ngg = zone->ngg;
  integer dir[]{0, 0, 0};
  const auto& b=zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &cv = zone->cv;
#if MULTISPECIES == 1
  auto &yk = zone->yk;
#endif // MULTISPECIES==1

  real t_wall{wall->temperature};

  const integer idx[]{i - dir[0], j - dir[1], k - dir[2]};
  if (wall->thermal_type == Wall::ThermalType::adiabatic) {
    t_wall = bv(idx[0], idx[1], idx[2], 5);
  }
  const real p{bv(idx[0], idx[1], idx[2], 4)};
#if MULTISPECIES == 1
  const auto mwk = param->mw;
  real mw{0};
  for (integer l = 0; l < zone->n_spec; ++l) {
    yk(i, j, k, l) = yk(idx[0], idx[1], idx[2], l);
    mw += yk(i, j, k, l) / mwk[l];
  }
  mw = 1 / mw;
#else
  constexpr real mw{cfd::mw_air};
#endif // MULTISPECIES==1

  bv(i, j, k, 0) = p * mw / (t_wall * cfd::R_u);
  bv(i, j, k, 1) = 0;
  bv(i, j, k, 2) = 0;
  bv(i, j, k, 3) = 0;
  bv(i, j, k, 4) = p;
  bv(i, j, k, 5) = t_wall;
  cv(i, j, k, 0) = bv(i, j, k, 0);
  cv(i, j, k, 1) = 0;
  cv(i, j, k, 2) = 0;
  cv(i, j, k, 3) = 0;
#if MULTISPECIES == 1
  for (integer l = 0; l < zone->n_spec; ++l) {
    cv(i, j, k, l + 5) = yk(i, j, k, l) * bv(i, j, k, 0);
  }
#endif // MULTISPECIES==1
  compute_total_energy(i, j, k, zone, param);

  for (int g = 1; g <= ngg; ++g) {
    const integer i_in[]{i - g * dir[0], j - g * dir[1], k - g * dir[2]};
    const integer i_gh[]{i + g * dir[0], j + g * dir[1], k + g * dir[2]};

#if MULTISPECIES == 1
    mw = 0;
    for (integer l = 0; l < zone->n_spec; ++l) {
      yk(i_gh[0], i_gh[1], i_gh[2], l) = yk(i_in[0], i_in[1], i_in[2], l);
      mw += yk(i_gh[0], i_gh[1], i_gh[2], l) / mwk[l];
    }
    mw = 1 / mw;
#endif // MULTISPECIES==1

    const double u_i{bv(i_in[0], i_in[1], i_in[2], 1)};
    const double v_i{bv(i_in[0], i_in[1], i_in[2], 2)};
    const double w_i{bv(i_in[0], i_in[1], i_in[2], 3)};
    const double p_i{bv(i_in[0], i_in[1], i_in[2], 4)};
    const double t_i{bv(i_in[0], i_in[1], i_in[2], 5)};

    double t_g{t_i};

    if (wall->thermal_type == Wall::ThermalType::isothermal) {
      t_g = 2 * t_wall - t_i;  // 0.5*(t_i+t_g)=t_w
      if (t_g <= 0.1 * t_wall) { // improve stability
        t_g = t_wall;
      }
    }

    const double rho_g{p_i * mw / (t_g * cfd::R_u)};
    bv(i_gh[0], i_gh[1], i_gh[2], 0) = rho_g;
    bv(i_gh[0], i_gh[1], i_gh[2], 1) = -u_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 2) = -v_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 3) = -w_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 4) = p_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 5) = t_g;
    cv(i_gh[0], i_gh[1], i_gh[2], 0) = rho_g;
    cv(i_gh[0], i_gh[1], i_gh[2], 1) = -rho_g * u_i;
    cv(i_gh[0], i_gh[1], i_gh[2], 2) = -rho_g * v_i;
    cv(i_gh[0], i_gh[1], i_gh[2], 3) = -rho_g * w_i;
#if MULTISPECIES == 1
    for (integer l = 0; l < zone->n_spec; ++l) {
      cv(i_gh[0], i_gh[1], i_gh[2], l + 5) = yk(i_gh[0], i_gh[1], i_gh[2], l) * rho_g;
    }
#endif // MULTISPECIES==1
    compute_total_energy(i_gh[0], i_gh[1], i_gh[2], zone, param);
  }

}
} // cfd
#endif
