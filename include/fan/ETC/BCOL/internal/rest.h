#include _WITCH_PATH(A/A.h)
#include _WITCH_PATH(VEC/VEC.h)
#include _WITCH_PATH(MATH/MATH.h)

#define __ETC_BCOL_P(p0) CONCAT3(ETC_BCOL_set_prefix, _, p0)
#define __ETC_BCOL_PP(p0) CONCAT4(_, ETC_BCOL_set_prefix, _, p0)

#if ETC_BCOL_set_PreferredFloatSize == 32
  #define __pfloat f32_t
  #define __floorf MATH_floor_f32
  #define __sin(v) MATH_sin_32(v)
  #define __cos(v) MATH_cos_32(v)
  #define __atan2 MATH_atan2_f32
  #define __sqrt(v) MATH_sqrt_32(v)
  #define __absf(v) MATH_abs_f32(v)
  #define __hypotenuse MATH_hypotenuse_f32
  #define __NormalResolve MATH_NormalResolve_f32
  #define __dot2 MATH_dot2_f32
  #define __copysignf MATH_copysign_f32
#elif ETC_BCOL_set_PreferredFloatSize == 64
  #define __pfloat f64_t
  #define __floorf MATH_floor_f64
  #define __sin(v) MATH_sin_64(v)
  #define __cos(v) MATH_cos_64(v)
  #define __atan2 MATH_atan2_f64
  #define __sqrt(v) MATH_sqrt_64(v)
  #define __absf(v) MATH_abs_f64(v)
  #define __hypotenuse MATH_hypotenuse_f64
  #define __NormalResolve MATH_NormalResolve_f64
  #define __dot2 MATH_dot2_f64
  #define __copysignf MATH_copysign_f64
#elif ETC_BCOL_set_PreferredFloatSize == 128
  #define __pfloat f128_t
  #define __floorf MATH_floor_f128
  #define __sin(v) MATH_sin_128(v)
  #define __cos(v) MATH_cos_128(v)
  #define __atan2 MATH_atan2_f128
  #define __sqrt(v) MATH_sqrt_128(v)
  #define __absf(v) MATH_abs_f128(v)
  #define __hypotenuse MATH_hypotenuse_f128
  #define __NormalResolve MATH_NormalResolve_f128
  #define __dot2 MATH_dot2_f128
  #define __copysignf MATH_copysign_f128
#else
  #error ?
#endif

#include _WITCH_PATH(ETC/BCOL/internal/Types/Types.h)

#include _WITCH_PATH(ETC/BCOL/internal/Object.h)

#include _WITCH_PATH(ETC/BCOL/internal/BaseFunctions.h)

#include _WITCH_PATH(ETC/BCOL/internal/Traverse.h)

#include _WITCH_PATH(ETC/BCOL/internal/Shape/Shape.h)

#include _WITCH_PATH(ETC/BCOL/internal/ObjectShape.h)

#if ETC_BCOL_set_SupportGrid == 1
  #include _WITCH_PATH(ETC/BCOL/internal/Grid.h)
#endif

#include _WITCH_PATH(ETC/BCOL/internal/Collision/Collision.h)

#include _WITCH_PATH(ETC/BCOL/internal/Step/Step.h)

#include _WITCH_PATH(ETC/BCOL/internal/CompiledShapes.h)

#include _WITCH_PATH(ETC/BCOL/internal/ImportHM.h)

#undef __copysignf
#undef __dot2
#undef __NormalResolve
#undef __hypotenuse
#undef __absf
#undef __sqrt
#undef __atan2
#undef __cos
#undef __sin
#undef __floorf
#undef __pfloat
#undef __ETC_BCOL_PP
#undef __ETC_BCOL_P
