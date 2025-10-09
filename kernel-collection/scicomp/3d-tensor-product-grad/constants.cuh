#ifndef _3D_GRADIENT_CONSTANTS_CUH
#define _3D_GRADIENT_CONSTANTS_CUH

#define MAX_N 16

template <class FP_T>
extern __constant__ FP_T op_const[MAX_N * MAX_N];

#endif
