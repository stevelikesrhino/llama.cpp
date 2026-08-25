#include "../fattn-mma-f16.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12080
#include "../fattn-mma-f16-sm120.cuh"

DECL_FATTN_MMA_F16_SM120_CASE(128,  8, 8);
DECL_FATTN_MMA_F16_SM120_CASE(256,  8, 8);
DECL_FATTN_MMA_F16_SM120_CASE(256, 32, 2);
DECL_FATTN_MMA_F16_SM120_CASE(512,  8, 8);
#endif
