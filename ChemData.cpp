#include "ChemData.h"
#include <fstream>
#include "fmt/core.h"
#include "gxl_lib/MyString.h"
#include "Constants.h"
#include "Element.h"
#include <cmath>

cfd::Species::Species(Parameter &parameter) {
  parameter.update_parameter("n_spec", 0);
  if (parameter.get_bool("species")) {
    std::ifstream file("./input_files/" + parameter.get_string("mechanism_file"));
    std::string input{};
    while (file >> input) {
      if (input[0] == '!') {
        // This line is comment
        std::getline(file, input);
        continue;
      }
      if (input == "ELEMENTS" || input == "ELEM") {
        // As the ANSYS_Chemkin-Pro_Input_Manual told, the element part must start with "ELEMENTS" or "ELEM",
        // which are all capitalized.
        break;
      }
    }
    // Read elements
    int n_elem{0};
    while (file >> input) {
      if (input[0] == '!') {
        // This line is comment
        std::getline(file, input);
        continue;
      }
      gxl::to_upper(input);
      if (input == "END") continue;// If this line is "END", there must be a "SPECIES" or "SPEC" followed.
      if (input == "SPECIES" || input == "SPEC") break;
      elem_list.emplace(input, n_elem++);
    }

    // Species
    int num_spec{0};
    bool has_therm{false};
    while (file >> input) {
      if (input[0] == '!') {
        // This line is comment
        std::getline(file, input);
        continue;
      }
      gxl::to_upper(input);
      if (input == "END") continue;// If this line is "END", there must be a "REACTIONS" or "THERMO" followed.
      if (input == "REACTIONS") break;
      if (input == "THERMO") {
        // The thermodynamic info is in this mechanism file
        has_therm = true;
        break;
      }
      spec_list.emplace(input, num_spec);
      ++num_spec;
    }
    set_nspec(num_spec, n_elem);
    parameter.update_parameter("n_spec", num_spec);
    parameter.update_parameter("n_var", parameter.get_int("n_var") + num_spec);
    parameter.update_parameter("n_scalar", parameter.get_int("n_scalar") + num_spec);

    if (!has_therm) {
      file.close();
      file.open("./input_files/" + parameter.get_string("therm_file"));
    }
    bool has_trans = read_therm(file, has_therm);

    if (!has_trans) {
      file.close();
      file.open("input_files/" + parameter.get_string("transport_file"));
    }
    read_tran(file);

    if (parameter.get_int("myid") == 0) {
      fmt::print("Mixture composed of {} species will be simulated.\n", n_spec);
      integer counter_spec{0};
      for (auto &[name, label]: spec_list) {
        fmt::print("{}\t", name);
        ++counter_spec;
        if (counter_spec % 10 == 0) {
          fmt::print("\n");
        }
      }
      fmt::print("\n");
    }
  }
}

void cfd::Species::compute_cp(real temp, real *cp) const &{
  const real t2{temp * temp}, t3{t2 * temp}, t4{t3 * temp};
  for (int i = 0; i < n_spec; ++i) {
    real tt{temp};
    if (temp < t_low[i]) {
      tt = t_low[i];
      const real tt2{tt * tt}, tt3{tt2 * tt}, tt4{tt3 * tt};
      auto &coeff = low_temp_coeff;
      cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 +
              coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
    } else {
      auto &coeff = tt < t_mid[i] ? low_temp_coeff : high_temp_coeff;
      cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * t2 +
              coeff(i, 3) * t3 + coeff(i, 4) * t4;
    }
    cp[i] *= R_u / mw[i];
  }
}

void cfd::Species::set_nspec(integer n_sp, integer n_elem) {
  n_spec = n_sp;
  elem_comp.resize(n_sp, n_elem);
  mw.resize(n_sp, 0);
  t_low.resize(n_sp, 300);
  t_mid.resize(n_sp, 1000);
  t_high.resize(n_sp, 5000);
  high_temp_coeff.resize(n_sp, 7);
  low_temp_coeff.resize(n_sp, 7);
  LJ_potent_inv.resize(n_sp, 0);
  vis_coeff.resize(n_sp, 0);
  WjDivWi_to_One4th.resize(n_sp, n_sp);
  sqrt_WiDivWjPl1Mul8.resize(n_sp, n_sp);
  x.resize(n_sp, 0);
  vis_spec.resize(n_sp, 0);
  lambda.resize(n_sp, 0);
  partition_fun.resize(n_sp, n_sp);
}

bool cfd::Species::read_therm(std::ifstream &therm_dat, bool read_from_comb_mech) {
  std::string input{};
  if (!read_from_comb_mech) {
    gxl::read_until(therm_dat, input, "THERMO", gxl::Case::upper);  // "THERMO"
  }
  while (std::getline(therm_dat, input)) {
    if (input[0] == '!' || input.empty()) {
      continue;
    }
    std::istringstream line(input);
    real T_low{300}, T_mid{1000}, T_high{5000};
    line >> T_low >> T_mid >> T_high;
    t_low.resize(n_spec, T_low);
    t_mid.resize(n_spec, T_mid);
    t_high.resize(n_spec, T_high);
    break;
  }

  std::string key{};
  int n_read{0};
  std::vector<int> have_read;
  std::istringstream line(input);
  bool has_trans{false};
  while (gxl::getline_to_stream(therm_dat, input, line, gxl::Case::upper)) {
    if (input[0] == '!' || input.empty()) {
      continue;
    }
    line >> key;
    // If the keyword is "END", just read the next line, if it's eof, then we won't come into this loop.
    // Else, a keyword "REACTIONS" or "TRANSPORT" may be encountered.
    if (key == "END") {
      if (n_read < n_spec) {
        fmt::print("The thermodynamic data aren't enough. We need {} species info but only {} are supplied.\n", n_spec,
                   n_read);
      }
      continue;
    }
    if (key == "REACTIONS") break;
    if (key == "TRANSPORT" || key == "TRAN") {
      has_trans = true;
      break;
    }
    if (n_read >= n_spec) continue;

    // Let us read the species.
    key.assign(input, 0, 18);
    gxl::to_stringstream(key, line);
    line >> key;
    if (!spec_list.contains(key)) {
      gxl::getline(therm_dat, input);
      gxl::getline(therm_dat, input);
      gxl::getline(therm_dat, input);
      gxl::getline_to_stream(therm_dat, input, line, gxl::Case::upper);
      line >> key;
      continue;
    }
    const int curr_sp = spec_list.at(key);
    // If the species info has been read, then the second set of parameters are ignored.
    bool read{false};
    for (auto ss: have_read) {
      if (ss == curr_sp) {
        read = true;
        break;
      }
    }
    if (read) {
      gxl::getline(therm_dat, input);
      gxl::getline(therm_dat, input);
      gxl::getline(therm_dat, input);
      gxl::getline_to_stream(therm_dat, input, line, gxl::Case::upper);
      line >> key;
      continue;
    }

    key.assign(input, 45, 10);  // T_low
    t_low[curr_sp] = std::stod(key);
    key.assign(input, 55, 10);  // T_high
    t_high[curr_sp] = std::stod(key);
    key.assign(input, 65, 10);  // Probably specify a different T_mid
    gxl::to_stringstream(key, line);
    line >> key;
    if (!key.empty()) t_mid[curr_sp] = std::stod(key);

    // Read element composition
    std::string comp_str{};
    for (int i = 0; i < 4; ++i) {
      comp_str.assign(input, 24 + i * 5, 5);
      gxl::trim_left(comp_str);
      if (comp_str.empty() || comp_str.starts_with('0')) break;
      gxl::to_stringstream(comp_str, line);
      line >> key;
      int stoi{0};
      line >> stoi;
      elem_comp(curr_sp, elem_list[key]) = stoi;
    }
    // Compute the relative molecular weight
    double mole_weight{0};
    for (const auto &[element, label]: elem_list) {
      mole_weight += Element{element}.get_atom_weight() *
                     elem_comp(curr_sp, label);
    }
    mw[curr_sp] = mole_weight;

    // Read the thermodynamic fitting coefficients
    std::getline(therm_dat, input);
    std::string cs1{}, cs2{}, cs3{}, cs4{}, cs5{};
    double c1, c2, c3, c4, c5;
    cs1.assign(input, 0, 15);
    cs2.assign(input, 15, 15);
    cs3.assign(input, 30, 15);
    cs4.assign(input, 45, 15);
    cs5.assign(input, 60, 15);
    c1 = std::stod(cs1);
    c2 = std::stod(cs2);
    c3 = std::stod(cs3);
    c4 = std::stod(cs4);
    c5 = std::stod(cs5);
    high_temp_coeff(curr_sp, 0) = c1;
    high_temp_coeff(curr_sp, 1) = c2;
    high_temp_coeff(curr_sp, 2) = c3;
    high_temp_coeff(curr_sp, 3) = c4;
    high_temp_coeff(curr_sp, 4) = c5;
    // second line
    std::getline(therm_dat, input);
    cs1.assign(input, 0, 15);
    cs2.assign(input, 15, 15);
    cs3.assign(input, 30, 15);
    cs4.assign(input, 45, 15);
    cs5.assign(input, 60, 15);
    c1 = std::stod(cs1);
    c2 = std::stod(cs2);
    c3 = std::stod(cs3);
    c4 = std::stod(cs4);
    c5 = std::stod(cs5);
    high_temp_coeff(curr_sp, 5) = c1;
    high_temp_coeff(curr_sp, 6) = c2;
    low_temp_coeff(curr_sp, 0) = c3;
    low_temp_coeff(curr_sp, 1) = c4;
    low_temp_coeff(curr_sp, 2) = c5;
    // third line
    std::getline(therm_dat, input);
    cs1.assign(input, 0, 15);
    cs2.assign(input, 15, 15);
    cs3.assign(input, 30, 15);
    cs4.assign(input, 45, 15);
    c1 = std::stod(cs1);
    c2 = std::stod(cs2);
    c3 = std::stod(cs3);
    c4 = std::stod(cs4);
    low_temp_coeff(curr_sp, 3) = c1;
    low_temp_coeff(curr_sp, 4) = c2;
    low_temp_coeff(curr_sp, 5) = c3;
    low_temp_coeff(curr_sp, 6) = c4;

    have_read.push_back(curr_sp);
    ++n_read;
  }
  return has_trans;
}

void cfd::Species::read_tran(std::ifstream &tran_dat) {
  std::string input{}, key{};
  std::istringstream line(input);
  integer n_read{0};
  std::vector<int> have_read;
  while (gxl::getline_to_stream(tran_dat, input, line, gxl::Case::upper)) {
    if (input[0] == '!' || input.empty()) {
      continue;
    }
    line >> key;
    if (key.starts_with("END") || key.starts_with("REACTIONS")) {
      if (n_read < n_spec) {
        fmt::print("The transport data aren't enough. We need {} species info but only {} are supplied.\n", n_spec,
                   n_read);
      }
      break;
    }
    if (!spec_list.contains(key)) {
      continue;
    }
    if (n_read >= n_spec) break;
    const int curr_sp = spec_list.at(key);
    // If the species info has been read, then the second set of parameters are ignored.
    bool read{false};
    for (auto ss: have_read) {
      if (ss == curr_sp) {
        read = true;
        break;
      }
    }
    if (read) {
      continue;
    }

    gxl::to_stringstream(input, line);
    real lj_potential{0}, collision_diameter{0}, pass{0};
    line >> key >> pass >> lj_potential >> collision_diameter;
    LJ_potent_inv[curr_sp] = 1.0 / lj_potential;
    vis_coeff[curr_sp] =
        2.6693e-6 * sqrt(mw[curr_sp]) / (collision_diameter * collision_diameter);

    have_read.push_back(curr_sp);
    ++n_read;
  }

  for (int i = 0; i < n_spec; ++i) {
    for (int j = 0; j < n_spec; ++j) {
      WjDivWi_to_One4th(i, j) = std::pow(mw[j] / mw[i], 0.25);
      sqrt_WiDivWjPl1Mul8(i, j) = 1.0 / std::sqrt(8 * (1 + mw[i] / mw[j]));
    }
  }
}

cfd::Reaction::Reaction(Parameter &parameter) {}

cfd::ChemData::ChemData(Parameter &parameter) : spec(parameter), reac(parameter) {}

