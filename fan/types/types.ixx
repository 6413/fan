module;

#include <cstdint>

export module fan.types;

export {
  typedef std::intptr_t si_t;
  //typedef uintptr_t uint_t;
  typedef std::intptr_t sint_t;
  typedef std::int8_t sint8_t;
  typedef std::int16_t sint16_t;
  typedef std::int32_t sint32_t;
  //typedef int64_t sint64_t;

  typedef float f32_t;
  typedef double f64_t;

  typedef double f_t;

  typedef f32_t cf_t;
}