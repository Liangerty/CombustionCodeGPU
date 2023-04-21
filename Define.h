#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#define __host__
#endif

using integer = int;
using real = double;
// using sizet = size_t;
using uint = unsigned int;


// First to specify if species are turned on. If not, air as a perfect gas will be assumed.
// #define MULTISPECIES 1
// Specify whether we use inheritence for polymorphism or we specify the choice before compiling.
// If set to 1, the choice of inviscid scheme and time advancement scheme should be given below.
// If set to 0, then the choices should be read in setup files and they are determined at runtime.
//#define STATIC_SETUP 1

//#if STATIC_SETUP==1
//enum class InviscidSchemeTag {
//  Roe=2, AUSM,WENO5Comp=51,WENO5Charac,WENO7Comp=71,WENO7Charac
//};
//// Tag for inviscid schemes. 2-Roe; 3-AUSM; 51-WENO5(component); 52-WENO5(characteristic); 71-WENO7(component); 72-WENO7(characteristic)...
//constexpr InviscidSchemeTag InviscidMethod = InviscidSchemeTag::Roe;
//// Tag for temporal scheme. Methods in single digit for steady simulation; methods bigger than 10 for transient simulation.
//// 1-ExplicitEuler; 2-LUSGS; 3-DPLUR
//// 11-ExplicitEuler; 21-Dual-time iteration with LUSGS; 22-Dual-time iteration with DPLUR; 31-TVD 3rd order RK
//#define TEMPORALSCHEME 2
////constexpr int TemporalMethod = 1;
//#endif // STATIC_SETUP

//constexpr int ViscousOrder = 2;

//enum class SpeciesModel { air, mixture };

//#if MULTISPECIES==1
//  constexpr SpeciesModel species_model = SpeciesModel::mixture;
//#else
//  constexpr SpeciesModel species_model = SpeciesModel::air;
//#endif // MULTISPECIES==1
//
