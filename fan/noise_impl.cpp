module;

#include <cmath>

#if defined(FAN_WINDOW)
  #include <fan/graphics/2D/algorithm/FastNoiseLite.h>
  template struct FastNoiseLite::Lookup<float>;
#endif

module fan.noise;

#if defined(FAN_WINDOW)

import std;

namespace fan {

  struct noise_t::impl_t {
    FastNoiseLite fn;
  };

  noise_t::noise_t(int seed_in) : seed(seed_in), impl(std::make_unique<impl_t>()) {
    impl->fn.SetSeed(seed_in);
  }

  noise_t::~noise_t() = default;
  noise_t::noise_t(noise_t&&) noexcept = default;
  noise_t& noise_t::operator=(noise_t&&) noexcept = default;

  f32_t noise_t::fade(f32_t t) {
    return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
  }

  f32_t noise_t::lerp(f32_t a, f32_t b, f32_t t) {
    return a + (b - a) * t;
  }

  std::uint32_t noise_t::hash(std::int32_t x, std::int32_t y, std::uint32_t seed) {
    std::uint32_t h = (std::uint32_t)x * 0x8da6b343u ^ (std::uint32_t)y * 0xd8163841u ^ seed * 0xcb1ab31fu;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
  }

  static FastNoiseLite::NoiseType to_noise_type(noise_t::base_t base) {
    using b = noise_t::base_t;
    switch (base) {
      case b::open_simplex2: return FastNoiseLite::NoiseType_OpenSimplex2;
      case b::open_simplex2s: return FastNoiseLite::NoiseType_OpenSimplex2S;
      case b::cellular: return FastNoiseLite::NoiseType_Cellular;
      case b::perlin: return FastNoiseLite::NoiseType_Perlin;
      case b::value_cubic: return FastNoiseLite::NoiseType_ValueCubic;
      case b::value: return FastNoiseLite::NoiseType_Value;
    }
    return FastNoiseLite::NoiseType_OpenSimplex2;
  }

  static FastNoiseLite::FractalType to_fractal_type(noise_t::fractal_t fractal) {
    using f = noise_t::fractal_t;
    switch (fractal) {
      case f::none: return FastNoiseLite::FractalType_None;
      case f::fbm: return FastNoiseLite::FractalType_FBm;
      case f::ridged: return FastNoiseLite::FractalType_Ridged;
      case f::ping_pong: return FastNoiseLite::FractalType_PingPong;
    }
    return FastNoiseLite::FractalType_FBm;
  }

  static FastNoiseLite::CellularDistanceFunction to_cellular_distance(noise_t::cellular_distance_function_t type) {
    using in_t = noise_t::cellular_distance_function_t;
    switch (type) {
      case in_t::euclidean: return FastNoiseLite::CellularDistanceFunction_Euclidean;
      case in_t::euclidean_sq: return FastNoiseLite::CellularDistanceFunction_EuclideanSq;
      case in_t::manhattan: return FastNoiseLite::CellularDistanceFunction_Manhattan;
      case in_t::hybrid: return FastNoiseLite::CellularDistanceFunction_Hybrid;
    }
    return FastNoiseLite::CellularDistanceFunction_EuclideanSq;
  }

  static FastNoiseLite::CellularReturnType to_cellular_return(noise_t::cellular_return_type_t type) {
    using in_t = noise_t::cellular_return_type_t;
    switch (type) {
      case in_t::cell_value: return FastNoiseLite::CellularReturnType_CellValue;
      case in_t::distance: return FastNoiseLite::CellularReturnType_Distance;
      case in_t::distance2: return FastNoiseLite::CellularReturnType_Distance2;
      case in_t::distance2_add: return FastNoiseLite::CellularReturnType_Distance2Add;
      case in_t::distance2_sub: return FastNoiseLite::CellularReturnType_Distance2Sub;
      case in_t::distance2_mul: return FastNoiseLite::CellularReturnType_Distance2Mul;
      case in_t::distance2_div: return FastNoiseLite::CellularReturnType_Distance2Div;
    }
    return FastNoiseLite::CellularReturnType_Distance;
  }

  bool noise_t::is_applied(base_t next_base, fractal_t next_fractal, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
    return state.valid &&
      state.base == next_base &&
      state.fractal == next_fractal &&
      state.seed == seed &&
      state.frequency == frequency &&
      state.gain == next_gain &&
      state.lacunarity == next_lacunarity &&
      state.weighted_strength == weighted_strength &&
      state.ping_pong_strength == ping_pong_strength &&
      state.cellular_jitter == cellular_jitter &&
      state.octaves == next_octaves &&
      state.cellular_distance_function == cellular_distance_function &&
      state.cellular_return_type == cellular_return_type;
  }

  bool noise_t::is_applied(base_t next_base, fractal_t next_fractal) const {
    return is_applied(next_base, next_fractal, octaves, lacunarity, gain);
  }

  void noise_t::apply() const {
    apply(base, fractal);
  }

  void noise_t::apply(base_t next_base, fractal_t next_fractal) const {
    apply(next_base, next_fractal, octaves, lacunarity, gain);
  }

  void noise_t::apply(base_t next_base, fractal_t next_fractal, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
    if (is_applied(next_base, next_fractal, next_octaves, next_lacunarity, next_gain)) {
      return;
    }

    impl->fn.SetSeed(seed);
    impl->fn.SetFrequency(frequency);
    impl->fn.SetNoiseType(to_noise_type(next_base));
    impl->fn.SetFractalType(to_fractal_type(next_fractal));
    impl->fn.SetFractalGain(next_gain);
    impl->fn.SetFractalLacunarity(next_lacunarity);
    impl->fn.SetFractalOctaves(next_octaves);
    impl->fn.SetFractalWeightedStrength(weighted_strength);
    impl->fn.SetFractalPingPongStrength(ping_pong_strength);
    impl->fn.SetCellularDistanceFunction(to_cellular_distance(cellular_distance_function));
    impl->fn.SetCellularReturnType(to_cellular_return(cellular_return_type));
    impl->fn.SetCellularJitter(cellular_jitter);

    state.valid = true;
    state.base = next_base;
    state.fractal = next_fractal;
    state.seed = seed;
    state.frequency = frequency;
    state.gain = next_gain;
    state.lacunarity = next_lacunarity;
    state.weighted_strength = weighted_strength;
    state.ping_pong_strength = ping_pong_strength;
    state.cellular_jitter = cellular_jitter;
    state.octaves = next_octaves;
    state.cellular_distance_function = cellular_distance_function;
    state.cellular_return_type = cellular_return_type;
  }

  f32_t noise_t::value_noise(f32_t x, f32_t y) const {
    std::int32_t x0 = (std::int32_t)std::floor(x);
    std::int32_t y0 = (std::int32_t)std::floor(y);
    f32_t tx = fade(x - x0);
    f32_t ty = fade(y - y0);
    auto h = [&](std::int32_t px, std::int32_t py) {
      return (f32_t)(hash(px, py, (std::uint32_t)seed) & 0x00ffffffu) / 16777215.f;
    };
    return lerp(lerp(h(x0, y0), h(x0 + 1, y0), tx), lerp(h(x0, y0 + 1), h(x0 + 1, y0 + 1), tx), ty);
  }

  f32_t noise_t::value_noise(const vec2& p) const {
    return value_noise(p.x, p.y);
  }

  f32_t noise_t::sample(base_t next_base, fractal_t next_fractal, f32_t x, f32_t y) const {
    apply(next_base, next_fractal);
    return impl->fn.GetNoise(x, y);
  }

  f32_t noise_t::sample(base_t next_base, fractal_t next_fractal, const vec2& p) const {
    return sample(next_base, next_fractal, p.x, p.y);
  }

  f32_t noise_t::sample(base_t next_base, fractal_t next_fractal, f32_t x, f32_t y, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
    apply(next_base, next_fractal, next_octaves, next_lacunarity, next_gain);
    return impl->fn.GetNoise(x, y);
  }

  f32_t noise_t::sample(base_t next_base, fractal_t next_fractal, const vec2& p, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
    return sample(next_base, next_fractal, p.x, p.y, next_octaves, next_lacunarity, next_gain);
  }

  f32_t noise_t::sample(f32_t x, f32_t y) const {
    return sample(base, fractal, x, y);
  }

  f32_t noise_t::sample(const vec2& p) const {
    return sample(p.x, p.y);
  }

  f32_t noise_t::sample_norm(base_t next_base, fractal_t next_fractal, f32_t x, f32_t y) const {
    return std::clamp(sample(next_base, next_fractal, x, y) * 0.5f + 0.5f, 0.f, 1.f);
  }

  f32_t noise_t::sample_norm(base_t next_base, fractal_t next_fractal, const vec2& p) const {
    return sample_norm(next_base, next_fractal, p.x, p.y);
  }

  f32_t noise_t::sample_norm(base_t next_base, fractal_t next_fractal, f32_t x, f32_t y, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
    return std::clamp(sample(next_base, next_fractal, x, y, next_octaves, next_lacunarity, next_gain) * 0.5f + 0.5f, 0.f, 1.f);
  }

  f32_t noise_t::sample_norm(base_t next_base, fractal_t next_fractal, const vec2& p, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
    return sample_norm(next_base, next_fractal, p.x, p.y, next_octaves, next_lacunarity, next_gain);
  }

  f32_t noise_t::sample_norm(f32_t x, f32_t y) const {
    return sample_norm(base, fractal, x, y);
  }

  f32_t noise_t::sample_norm(const vec2& p) const {
    return sample_norm(p.x, p.y);
  }

  f32_t noise_t::get_noise(f32_t x, f32_t y) const {
    return sample(x, y);
  }

  f32_t noise_t::get_noise(const vec2& p) const {
    return sample(p);
  }

  f32_t noise_t::fbm(f32_t x, f32_t y, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
    f32_t sum = 0;
    f32_t amp = 1;
    f32_t norm = 0;
    f32_t freq = 1;
    for (int i = 0; i < next_octaves; ++i) {
      sum += value_noise(x * freq, y * freq) * amp;
      norm += amp;
      freq *= next_lacunarity;
      amp *= next_gain;
    }
    return sum / norm;
  }

  f32_t noise_t::fbm(const vec2& p, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
    return fbm(p.x, p.y, next_octaves, next_lacunarity, next_gain);
  }

  f32_t noise_t::fbm_norm(f32_t x, f32_t y, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
    return fbm(x, y, next_octaves, next_lacunarity, next_gain);
  }

  f32_t noise_t::fbm_norm(const vec2& p, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
    return fbm_norm(p.x, p.y, next_octaves, next_lacunarity, next_gain);
  }

  f32_t noise_t::fbm_raw(f32_t x, f32_t y, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
    return fbm(x, y, next_octaves, next_lacunarity, next_gain) * 2.f - 1.f;
  }

  f32_t noise_t::fbm_raw(const vec2& p, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
    return fbm_raw(p.x, p.y, next_octaves, next_lacunarity, next_gain);
  }

  f32_t noise_t::ridged(f32_t x, f32_t y) const {
    return sample(base, fractal_t::ridged, x, y);
  }

  f32_t noise_t::ridged(const vec2& p) const {
    return ridged(p.x, p.y);
  }

  f32_t noise_t::ridged_norm(f32_t x, f32_t y) const {
    return sample_norm(base, fractal_t::ridged, x, y);
  }

  f32_t noise_t::ridged_norm(const vec2& p) const {
    return ridged_norm(p.x, p.y);
  }

  f32_t noise_t::ping_pong(f32_t x, f32_t y) const {
    return sample(base, fractal_t::ping_pong, x, y);
  }

  f32_t noise_t::ping_pong(const vec2& p) const {
    return ping_pong(p.x, p.y);
  }

  f32_t noise_t::ping_pong_norm(f32_t x, f32_t y) const {
    return sample_norm(base, fractal_t::ping_pong, x, y);
  }

  f32_t noise_t::ping_pong_norm(const vec2& p) const {
    return ping_pong_norm(p.x, p.y);
  }

  f32_t noise_t::simplex(f32_t x, f32_t y) const {
    return sample(base_t::open_simplex2, fractal_t::none, x, y);
  }

  f32_t noise_t::simplex(const vec2& p) const {
    return simplex(p.x, p.y);
  }

  f32_t noise_t::simplex_norm(f32_t x, f32_t y) const {
    return sample_norm(base_t::open_simplex2, fractal_t::none, x, y);
  }

  f32_t noise_t::simplex_norm(const vec2& p) const {
    return simplex_norm(p.x, p.y);
  }

  f32_t noise_t::simplex_fbm(f32_t x, f32_t y) const {
    return sample(base_t::open_simplex2, fractal_t::fbm, x, y);
  }

  f32_t noise_t::simplex_fbm(const vec2& p) const {
    return simplex_fbm(p.x, p.y);
  }

  f32_t noise_t::simplex_fbm_norm(f32_t x, f32_t y) const {
    return sample_norm(base_t::open_simplex2, fractal_t::fbm, x, y);
  }

  f32_t noise_t::simplex_fbm_norm(const vec2& p) const {
    return simplex_fbm_norm(p.x, p.y);
  }

  f32_t noise_t::perlin(f32_t x, f32_t y) const {
    return sample(base_t::perlin, fractal_t::none, x, y);
  }

  f32_t noise_t::perlin(const vec2& p) const {
    return perlin(p.x, p.y);
  }

  f32_t noise_t::perlin_norm(f32_t x, f32_t y) const {
    return sample_norm(base_t::perlin, fractal_t::none, x, y);
  }

  f32_t noise_t::perlin_norm(const vec2& p) const {
    return perlin_norm(p.x, p.y);
  }

  f32_t noise_t::perlin_fbm(f32_t x, f32_t y) const {
    return sample(base_t::perlin, fractal_t::fbm, x, y);
  }

  f32_t noise_t::perlin_fbm(const vec2& p) const {
    return perlin_fbm(p.x, p.y);
  }

  f32_t noise_t::perlin_fbm_norm(f32_t x, f32_t y) const {
    return sample_norm(base_t::perlin, fractal_t::fbm, x, y);
  }

  f32_t noise_t::perlin_fbm_norm(const vec2& p) const {
    return perlin_fbm_norm(p.x, p.y);
  }

  f32_t noise_t::value(f32_t x, f32_t y) const {
    return sample(base_t::value, fractal_t::none, x, y);
  }

  f32_t noise_t::value(const vec2& p) const {
    return value(p.x, p.y);
  }

  f32_t noise_t::value_norm(f32_t x, f32_t y) const {
    return sample_norm(base_t::value, fractal_t::none, x, y);
  }

  f32_t noise_t::value_norm(const vec2& p) const {
    return value_norm(p.x, p.y);
  }

  f32_t noise_t::value_fbm(f32_t x, f32_t y) const {
    return sample(base_t::value, fractal_t::fbm, x, y);
  }

  f32_t noise_t::value_fbm(const vec2& p) const {
    return value_fbm(p.x, p.y);
  }

  f32_t noise_t::value_fbm_norm(f32_t x, f32_t y) const {
    return sample_norm(base_t::value, fractal_t::fbm, x, y);
  }

  f32_t noise_t::value_fbm_norm(const vec2& p) const {
    return value_fbm_norm(p.x, p.y);
  }

  f32_t noise_t::value_cubic(f32_t x, f32_t y) const {
    return sample(base_t::value_cubic, fractal_t::none, x, y);
  }

  f32_t noise_t::value_cubic(const vec2& p) const {
    return value_cubic(p.x, p.y);
  }

  f32_t noise_t::value_cubic_norm(f32_t x, f32_t y) const {
    return sample_norm(base_t::value_cubic, fractal_t::none, x, y);
  }

  f32_t noise_t::value_cubic_norm(const vec2& p) const {
    return value_cubic_norm(p.x, p.y);
  }

  f32_t noise_t::cellular(f32_t x, f32_t y) const {
    return sample(base_t::cellular, fractal_t::none, x, y);
  }

  f32_t noise_t::cellular(const vec2& p) const {
    return cellular(p.x, p.y);
  }

  f32_t noise_t::cellular_norm(f32_t x, f32_t y) const {
    return sample_norm(base_t::cellular, fractal_t::none, x, y);
  }

  f32_t noise_t::cellular_norm(const vec2& p) const {
    return cellular_norm(p.x, p.y);
  }

  std::vector<std::uint8_t> noise_t::generate_data(const vec2& size, f32_t tex_min, f32_t tex_max) const {
    return generate_data(base, fractal, size, tex_min, tex_max);
  }

  std::vector<std::uint8_t> noise_t::generate_data(base_t next_base, fractal_t next_fractal, const vec2& size, f32_t tex_min, f32_t tex_max) const {
    std::uint32_t sx = (std::uint32_t)size.x;
    std::uint32_t sy = (std::uint32_t)size.y;
    std::vector<std::uint8_t> data((std::size_t)sx * sy * 3);
    f32_t scale = 255.0f / (tex_max - tex_min);
    std::uint32_t idx = 0;
    for (std::uint32_t y = 0; y < sy; ++y) {
      for (std::uint32_t x = 0; x < sx; ++x) {
        f32_t val = sample(next_base, next_fractal, (f32_t)x, (f32_t)y);
        std::uint8_t c = (std::uint8_t)std::clamp((val - tex_min) * scale, 0.0f, 255.0f);
        data[idx * 3 + 0] = c;
        data[idx * 3 + 1] = c;
        data[idx * 3 + 2] = c;
        ++idx;
      }
    }
    return data;
  }

  graphics::image_t noise_t::to_image(const vec2& size, f32_t tex_min, f32_t tex_max) const {
    auto data = generate_data(size, tex_min, tex_max);
    return from_data(size, data);
  }

  graphics::image_t noise_t::to_image(base_t next_base, fractal_t next_fractal, const vec2& size, f32_t tex_min, f32_t tex_max) const {
    auto data = generate_data(next_base, next_fractal, size, tex_min, tex_max);
    return from_data(size, data);
  }

  graphics::image_t noise_t::from_data(const vec2& size, const std::vector<std::uint8_t>& data) {
    graphics::image_load_properties_t lp;
    image::info_t ii;
    ii.data = (void*)data.data();
    ii.size = size;
    ii.channels = 3;
    return image_load(ii, lp);
  }

}

#endif