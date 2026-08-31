#include <cuda.h>

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif
#include <cstdint>
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;
extern "C" __global__ void __launch_bounds__(256) boundflow_s4_pack_ainput_endpoint_ternary_kernel(float* __restrict__ coefficient, signed char* __restrict__ endpoint_selector);
extern "C" __global__ void __launch_bounds__(256) boundflow_s4_select_input_endpoint_ternary_kernel(float* __restrict__ lower, float* __restrict__ selected_endpoint, signed char* __restrict__ selector, float* __restrict__ upper);
extern "C" __global__ void __launch_bounds__(256) boundflow_s4_pack_ainput_endpoint_ternary_kernel(float* __restrict__ coefficient, signed char* __restrict__ endpoint_selector) {
    float v_ = coefficient[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))];
  signed char condval;
  if ((((*(uint *)(&(v_))) & (uint)2139095040) == (uint)2139095040)) {
    condval = (signed char)-128;
  } else {
    signed char condval_1;
    if ((0x0p+0f/*0.000000e+00*/ < coefficient[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))])) {
      condval_1 = (signed char)1;
    } else {
      signed char condval_2;
      if ((coefficient[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] < 0x0p+0f/*0.000000e+00*/)) {
        condval_2 = (signed char)-1;
      } else {
        condval_2 = (signed char)0;
      }
      condval_1 = condval_2;
    }
    condval = condval_1;
  }
  endpoint_selector[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = condval;
}

extern "C" __global__ void __launch_bounds__(256) boundflow_s4_select_input_endpoint_ternary_kernel(float* __restrict__ lower, float* __restrict__ selected_endpoint, signed char* __restrict__ selector, float* __restrict__ upper) {
  float condval;
  if ((selector[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] == (signed char)1)) {
    condval = lower[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))];
  } else {
    float condval_1;
    if ((selector[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] == (signed char)-1)) {
      condval_1 = upper[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))];
    } else {
      float condval_2;
      if ((selector[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] == (signed char)0)) {
        condval_2 = ((lower[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] + upper[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))]) * 0x1p-1f/*5.000000e-01*/);
      } else {
          uint v_ = (uint)2143289344;
        condval_2 = (*(float *)(&(v_)));
      }
      condval_1 = condval_2;
    }
    condval = condval_1;
  }
  selected_endpoint[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = condval;
}

