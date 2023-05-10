// Although this file is purely on CPU, but because one of the included file is "Matrix.hpp", the CUDA part would be in
// And the cuda_runtime.h is included, thus cu is needed instead of cpp

#include "Output.h"
#include "fmt/core.h"
#include "Field.h"
#include "ChemData.h"
#include <filesystem>

cfd::Output::Output(int _myid, const cfd::Mesh& _mesh, std::vector<Field>& _field, const cfd::Parameter& _parameter, const cfd::Species& spec)://
    myid{_myid},mesh{_mesh}, field(_field), parameter{_parameter}, species{spec} {
  const std::filesystem::path out_dir("output/field");
  if (!exists(out_dir))
    create_directories(out_dir);
}

namespace cfd {
void Output::print_field(int ngg) const {
  // Copy data from GPU to CPU
  for (auto& f:field) {
    f.copy_data_from_device();
  }

  const std::filesystem::path out_dir("output/field");
  FILE* fp = fopen((out_dir.string() + fmt::format("/flowfield{:>4}.plt", myid)).c_str(), "wb");

  // I. Header section

  // i. Magic number, Version number
  // V112 / V191. V112 was introduced in 2009 while V191 in 2019. They are different only in poly data, so no
  // difference is related to us. For common use, we use V112.
  constexpr auto magic_number{"#!TDV112"};
  fwrite(magic_number, 8, 1, fp);

  // ii. Integer value of 1
  constexpr int32_t byte_order{1};
  fwrite(&byte_order, 4, 1, fp);

  // iii. Title and variable names.
  // 1. FileType: 0=full, 1=grid, 2=solution
  constexpr int32_t file_type{0};
  fwrite(&file_type, 4, 1, fp);
  // 2. Title
  write_str("Solution file", fp);
  // 3. Number of variables in the datafile, for this file, n_var = 3(x,y,z)+7(density,u,v,w,p,t,Ma)+n_spec+n_scalar
  const int32_t n_var{3 + 7 + field[0].n_spec};
  fwrite(&n_var, 4, 1, fp);
  // 4. Variable names.
  std::vector<std::string> var_name{"x", "y", "z", "density", "u", "v", "w", "pressure", "temperature", "mach"};
  var_name.resize(n_var);
  auto& names = species.spec_list;
  for (auto& [name, ind] : names)
    var_name[ind + 10] = name;
  for (auto& name : var_name)
    write_str(name.c_str(), fp);

  // iv. Zones
  for (int i = 0; i < mesh.n_block; ++i) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    constexpr float zone_marker{299.0f};
    fwrite(&zone_marker, 4, 1, fp);
    // 2. Zone name.
    write_str(fmt::format("zone {}", i).c_str(), fp);
    // 3. Parent zone. No longer used
    constexpr int32_t parent_zone{-1};
    fwrite(&parent_zone, 4, 1, fp);
    // 4. Strand ID. -2 = pending strand ID for assignment by Tecplot; -1 = static strand ID; >= 0 valid strand ID
    constexpr int32_t strand_id{-2};
    fwrite(&strand_id, 4, 1, fp);
    // 5. Solution time. For steady, the value is set 0. For unsteady, please create a new class
    constexpr real solution_time{0};
    fwrite(&solution_time, 8, 1, fp);
    // 6. Default Zone Color. Seldom used. Set to -1.
    constexpr int32_t zone_color{-1};
    fwrite(&zone_color, 4, 1, fp);
    // 7. ZoneType 0=ORDERED
    constexpr int32_t zone_type{0};
    fwrite(&zone_type, 4, 1, fp);
    // 8. Specify Var Location. 0 = All data is located at the nodes
    constexpr int32_t var_location{0};
    fwrite(&var_location, 4, 1, fp);
    // 9. Are raw local 1-to-1 face neighbors supplied? ORDERED zones must specify 0 for this value because
    // raw face neighbors are not defined for these zone types.
    constexpr int32_t raw_face_neighbor{0};
    fwrite(&raw_face_neighbor, 4, 1, fp);
    // 10. Number of miscellaneous user-defined face neighbor connections (value >= 0)
    constexpr int32_t miscellaneous_face{0};
    fwrite(&miscellaneous_face, 4, 1, fp);
    // For ordered zone, specify IMax, JMax, KMax
    auto& b = mesh[i];
    const auto mx{b.mx + 2 * ngg}, my{b.my + 2 * ngg}, mz{b.mz + 2 * ngg};
    fwrite(&mx, 4, 1, fp);
    fwrite(&my, 4, 1, fp);
    fwrite(&mz, 4, 1, fp);
    // 11. For all zone types (repeat for each Auxiliary data name/value pair)
    // 1=Auxiliary name/value pair to follow; 0=No more Auxiliary name/value pairs.
    constexpr int32_t auxi_data{0};
    fwrite(&auxi_data, 4, 1, fp);
    // If the above is 1, then supply the following: name string, Auxiliary Value Format, Value string
  }

  // End of Header
  constexpr float EOHMARKER{357.0f};
  fwrite(&EOHMARKER, 4, 1, fp);

  // II. Data Section
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    constexpr float zone_marker{299.0f};
    fwrite(&zone_marker, 4, 1, fp);
    // 2. Variable data format, 1=Float, 2=Double, 3=LongInt, 4=ShortInt, 5=Byte, 6=Bit
    constexpr int32_t data_format{1};
    constexpr size_t data_size{4};
    for (int l = 0; l < n_var; ++l)
      fwrite(&data_format, 4, 1, fp);
    // 3. Has passive variables: 0 = no, 1 = yes.
    constexpr int32_t passive_var{0};
    fwrite(&passive_var, 4, 1, fp);
    // 4. Has variable sharing 0 = no, 1 = yes.
    constexpr int32_t shared_var{0};
    fwrite(&shared_var, 4, 1, fp);
    // 5. Zero based zone number to share connectivity list with (-1 = no sharing).
    constexpr int32_t shared_connect{-1};
    fwrite(&shared_connect, 4, 1, fp);
    // 6. Compressed list of min/max pairs for each non-shared and non-passive variable.
    // For each non-shared and non-passive variable (as specified above):
    auto& b{mesh[blk]};
    auto& v{field[blk].h_zone};
    const auto mx{b.mx}, my{b.my}, mz{b.mz};
    const std::vector<gxl::Array3D<double>>& vars{b.x, b.y, b.z};
    for (auto& var : vars) {
      double min_val{var(-ngg, -ngg, -ngg)}, max_val{var(-ngg, -ngg, -ngg)};
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, var(i, j, k));
            max_val = std::max(max_val, var(i, j, k));
          }
        }
      }
      fwrite(&min_val, 8, 1, fp);
      fwrite(&max_val, 8, 1, fp);
    }
    std::array min_val{
        v.bv(-ngg, -ngg, -ngg, 0), v.bv(-ngg, -ngg, -ngg, 1), v.bv(-ngg, -ngg, -ngg, 2),
        v.bv(-ngg, -ngg, -ngg, 3), v.bv(-ngg, -ngg, -ngg, 4), v.bv(-ngg, -ngg, -ngg, 5)
    };
    std::array max_val{
        v.bv(-ngg, -ngg, -ngg, 0), v.bv(-ngg, -ngg, -ngg, 1), v.bv(-ngg, -ngg, -ngg, 2),
        v.bv(-ngg, -ngg, -ngg, 3), v.bv(-ngg, -ngg, -ngg, 4), v.bv(-ngg, -ngg, -ngg, 5)
    };
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          for (int l = 0; l < 6; ++l) {
            min_val[l] = std::min(min_val[l], v.bv(i, j, k, l));
            max_val[l] = std::max(max_val[l], v.bv(i, j, k, l));
          }
        }
      }
    }
    for (int l = 0; l < 6; ++l) {
      fwrite(&min_val[l], 8, 1, fp);
      fwrite(&max_val[l], 8, 1, fp);
    }
    min_val[0] = v.mach(-ngg, -ngg, -ngg);
    max_val[0] = v.mach(-ngg, -ngg, -ngg);
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val[0] = std::min(min_val[0], v.mach(i, j, k));
          max_val[0] = std::max(max_val[0], v.mach(i, j, k));
        }
      }
    }
    fwrite(min_val.data(), 8, 1, fp);
    fwrite(max_val.data(), 8, 1, fp);
    const int n_spec{field[0].n_spec};
#if MULTISPECIES==1
    std::vector<double> y_min(n_spec,0),y_max(n_spec,0);
    for (int l = 0; l < n_spec; ++l) {
      y_min[l]=v.yk(-ngg, -ngg, -ngg,l);
      y_max[l]=v.yk(-ngg, -ngg, -ngg,l);
    }
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          for (int l = 0; l < n_spec; ++l) {
            y_min[l] = std::min(y_min[l], v.yk(i, j, k, l));
            y_max[l] = std::max(y_max[l], v.yk(i, j, k, l));
          }
        }
      }
    }
    for (int l = 0; l < n_spec; ++l) {
      fwrite(&y_min[l], 8, 1, fp);
      fwrite(&y_max[l], 8, 1, fp);
    }
#endif

    // 7. Zone Data.
    for (auto& var : vars) {
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            const auto value = static_cast<float>(var(i, j, k));
            fwrite(&value, data_size, 1, fp);
          }
        }
      }
    }
    for (int l = 0; l < 6; ++l) {
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            const auto value = static_cast<float>(v.bv(i, j, k, l));
            fwrite(&value, data_size, 1, fp);
          }
        }
      }
    }
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          const auto value = static_cast<float>(v.mach(i, j, k));
          fwrite(&value, data_size, 1, fp);
        }
      }
    }
#if MULTISPECIES==1
    for (int l = 0; l < n_spec; ++l) {
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            const auto value = static_cast<float>(v.yk(i, j, k, l));
            fwrite(&value, data_size, 1, fp);
          }
        }
      }
    }
#endif
  }
  fclose(fp);
}

void write_str(const char* str, FILE* file) {
  int value = 0;
  while (*str != '\0') {
    value = static_cast<int>(*str);
    fwrite(&value, sizeof(int), 1, file);
    ++str;
  }
  constexpr char null_char = '\0';
  value                    = static_cast<int>(null_char);
  fwrite(&value, sizeof(int), 1, file);
}
} // cfd