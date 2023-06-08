#pragma once

#include "Define.h"
#include "DParameter.h"
#include "DPLURHPP.cuh"

namespace cfd{
template<MixtureModel mixture_model, TurbMethod turb_method>
void implicit_treatment(const Block &block, const DParameter *param, DZone *d_ptr, const Parameter& parameter, DZone *h_ptr) {
  switch (parameter.get_int("implicit_method")) {
    case 0: // Explicit
      if constexpr (mixture_model==MixtureModel::FR){
        switch (parameter.get_int("chemSrcMethod")) {
          case 0: // Explicit treat the chemical source
            break;
          case 1: // Point implicit method
          default: // Explicit treat the chemical source
            break;
        }
      }
      return;
    case 1: // DPLUR
      DPLUR<mixture_model, turb_method>(block, param, d_ptr, h_ptr, parameter);
    default:
      return;
  }
}
}