module;

export module fan.types;

import std;

export {
  using si_t = std::intptr_t;
  using sint_t = std::intptr_t;
  using sint8_t = std::int8_t;
  using sint16_t = std::int16_t;
  using sint32_t = std::int32_t;

  using f32_t = float;
  using f64_t = double;
  using f_t = double;
  using cf_t = f32_t;
}

export namespace fan {
  using bytes_t = std::vector<std::uint8_t>;
}