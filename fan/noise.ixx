module;

#if defined(FAN_WINDOW)
  #include <algorithm>
  #include <cmath>
  #include <cstdint>
  #include <vector>

  #include <fan/graphics/2D/algorithm/FastNoiseLite.h>
  
  template class FastNoiseLite::Lookup<float>;
#endif

export module fan.noise;

#if defined(FAN_WINDOW)

import std;

import fan.types;
import fan.types.vector;
import fan.graphics.common_context;
import fan.graphics.image_load;

export namespace fan {

  struct noise_t {
    enum class base_t {
      open_simplex2,
      open_simplex2s,
      perlin,
      value,
      value_cubic,
      cellular
    };

    enum class fractal_t {
      none,
      fbm,
      ridged,
      ping_pong
    };

    struct state_t {
      bool valid = false;
      base_t base = base_t::open_simplex2;
      fractal_t fractal = fractal_t::fbm;
      int seed = 0;
      f32_t frequency = 0;
      f32_t gain = 0;
      f32_t lacunarity = 0;
      f32_t weighted_strength = 0;
      f32_t ping_pong_strength = 0;
      f32_t cellular_jitter = 0;
      int octaves = 0;
      FastNoiseLite::CellularDistanceFunction cellular_distance_function = FastNoiseLite::CellularDistanceFunction_EuclideanSq;
      FastNoiseLite::CellularReturnType cellular_return_type = FastNoiseLite::CellularReturnType_Distance;
    };

    static f32_t fade(f32_t t) {
      return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
    }

    static f32_t lerp(f32_t a, f32_t b, f32_t t) {
      return a + (b - a) * t;
    }

    static std::uint32_t hash(std::int32_t x, std::int32_t y, std::uint32_t seed) {
      std::uint32_t h = (std::uint32_t)x * 0x8da6b343u ^ (std::uint32_t)y * 0xd8163841u ^ seed * 0xcb1ab31fu;
      h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
      return h;
    }

    static FastNoiseLite::NoiseType to_noise_type(base_t base) {
      switch (base) {
        case base_t::open_simplex2: return FastNoiseLite::NoiseType_OpenSimplex2;
        case base_t::open_simplex2s: return FastNoiseLite::NoiseType_OpenSimplex2S;
        case base_t::perlin: return FastNoiseLite::NoiseType_Perlin;
        case base_t::value: return FastNoiseLite::NoiseType_Value;
        case base_t::value_cubic: return FastNoiseLite::NoiseType_ValueCubic;
        case base_t::cellular: return FastNoiseLite::NoiseType_Cellular;
      }
      return FastNoiseLite::NoiseType_OpenSimplex2;
    }

    static FastNoiseLite::FractalType to_fractal_type(fractal_t fractal) {
      switch (fractal) {
        case fractal_t::none: return FastNoiseLite::FractalType_None;
        case fractal_t::fbm: return FastNoiseLite::FractalType_FBm;
        case fractal_t::ridged: return FastNoiseLite::FractalType_Ridged;
        case fractal_t::ping_pong: return FastNoiseLite::FractalType_PingPong;
      }
      return FastNoiseLite::FractalType_FBm;
    }

    bool is_applied(base_t next_base, fractal_t next_fractal, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
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

    bool is_applied(base_t next_base, fractal_t next_fractal) const {
      return is_applied(next_base, next_fractal, octaves, lacunarity, gain);
    }

    void apply() const {
      apply(base, fractal);
    }

    void apply(base_t next_base, fractal_t next_fractal) const {
      apply(next_base, next_fractal, octaves, lacunarity, gain);
    }

    void apply(base_t next_base, fractal_t next_fractal, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const {
      if (is_applied(next_base, next_fractal, next_octaves, next_lacunarity, next_gain)) {
        return;
      }

      fn.SetSeed(seed);
      fn.SetFrequency(frequency);
      fn.SetNoiseType(to_noise_type(next_base));
      fn.SetFractalType(to_fractal_type(next_fractal));
      fn.SetFractalGain(next_gain);
      fn.SetFractalLacunarity(next_lacunarity);
      fn.SetFractalOctaves(next_octaves);
      fn.SetFractalWeightedStrength(weighted_strength);
      fn.SetFractalPingPongStrength(ping_pong_strength);
      fn.SetCellularDistanceFunction(cellular_distance_function);
      fn.SetCellularReturnType(cellular_return_type);
      fn.SetCellularJitter(cellular_jitter);

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

    f32_t value_noise(f32_t x, f32_t y) const {
      std::int32_t x0 = (std::int32_t)std::floor(x);
      std::int32_t y0 = (std::int32_t)std::floor(y);
      f32_t tx = fade(x - x0);
      f32_t ty = fade(y - y0);
      auto h = [&](std::int32_t px, std::int32_t py) {
        return (f32_t)(hash(px, py, (std::uint32_t)seed) & 0x00ffffffu) / 16777215.f;
      };
      return lerp(lerp(h(x0, y0), h(x0 + 1, y0), tx), lerp(h(x0, y0 + 1), h(x0 + 1, y0 + 1), tx), ty);
    }

    f32_t value_noise(const vec2& p) const {
      return value_noise(p.x, p.y);
    }

    f32_t sample(base_t next_base, fractal_t next_fractal, f32_t x, f32_t y) const {
      apply(next_base, next_fractal);
      return fn.GetNoise(x, y);
    }

    f32_t sample(base_t next_base, fractal_t next_fractal, const vec2& p) const {
      return sample(next_base, next_fractal, p.x, p.y);
    }

    f32_t sample(base_t next_base, fractal_t next_fractal, f32_t x, f32_t y, int next_octaves, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const {
      apply(next_base, next_fractal, next_octaves, next_lacunarity, next_gain);
      return fn.GetNoise(x, y);
    }

    f32_t sample(base_t next_base, fractal_t next_fractal, const vec2& p, int next_octaves, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const {
      return sample(next_base, next_fractal, p.x, p.y, next_octaves, next_lacunarity, next_gain);
    }

    f32_t sample(f32_t x, f32_t y) const {
      return sample(base, fractal, x, y);
    }

    f32_t sample(const vec2& p) const {
      return sample(p.x, p.y);
    }

    f32_t sample_norm(base_t next_base, fractal_t next_fractal, f32_t x, f32_t y) const {
      return std::clamp(sample(next_base, next_fractal, x, y) * 0.5f + 0.5f, 0.f, 1.f);
    }

    f32_t sample_norm(base_t next_base, fractal_t next_fractal, const vec2& p) const {
      return sample_norm(next_base, next_fractal, p.x, p.y);
    }

    f32_t sample_norm(base_t next_base, fractal_t next_fractal, f32_t x, f32_t y, int next_octaves, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const {
      return std::clamp(sample(next_base, next_fractal, x, y, next_octaves, next_lacunarity, next_gain) * 0.5f + 0.5f, 0.f, 1.f);
    }

    f32_t sample_norm(base_t next_base, fractal_t next_fractal, const vec2& p, int next_octaves, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const {
      return sample_norm(next_base, next_fractal, p.x, p.y, next_octaves, next_lacunarity, next_gain);
    }

    f32_t sample_norm(f32_t x, f32_t y) const {
      return sample_norm(base, fractal, x, y);
    }

    f32_t sample_norm(const vec2& p) const {
      return sample_norm(p.x, p.y);
    }

    f32_t get_noise(f32_t x, f32_t y) const {
      return sample(x, y);
    }

    f32_t get_noise(const vec2& p) const {
      return sample(p);
    }

    f32_t fbm(f32_t x, f32_t y, int next_octaves = 5, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const {
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

    f32_t fbm(const vec2& p, int next_octaves = 5, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const {
      return fbm(p.x, p.y, next_octaves, next_lacunarity, next_gain);
    }

    f32_t fbm_norm(f32_t x, f32_t y, int next_octaves = 5, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const {
      return fbm(x, y, next_octaves, next_lacunarity, next_gain);
    }

    f32_t fbm_norm(const vec2& p, int next_octaves = 5, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const {
      return fbm_norm(p.x, p.y, next_octaves, next_lacunarity, next_gain);
    }

    f32_t fbm_raw(f32_t x, f32_t y, int next_octaves = 5, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const {
      return fbm(x, y, next_octaves, next_lacunarity, next_gain) * 2.f - 1.f;
    }

    f32_t fbm_raw(const vec2& p, int next_octaves = 5, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const {
      return fbm_raw(p.x, p.y, next_octaves, next_lacunarity, next_gain);
    }

    f32_t ridged(f32_t x, f32_t y) const {
      return sample(base, fractal_t::ridged, x, y);
    }

    f32_t ridged(const vec2& p) const {
      return ridged(p.x, p.y);
    }

    f32_t ridged_norm(f32_t x, f32_t y) const {
      return sample_norm(base, fractal_t::ridged, x, y);
    }

    f32_t ridged_norm(const vec2& p) const {
      return ridged_norm(p.x, p.y);
    }

    f32_t ping_pong(f32_t x, f32_t y) const {
      return sample(base, fractal_t::ping_pong, x, y);
    }

    f32_t ping_pong(const vec2& p) const {
      return ping_pong(p.x, p.y);
    }

    f32_t ping_pong_norm(f32_t x, f32_t y) const {
      return sample_norm(base, fractal_t::ping_pong, x, y);
    }

    f32_t ping_pong_norm(const vec2& p) const {
      return ping_pong_norm(p.x, p.y);
    }

    f32_t simplex(f32_t x, f32_t y) const {
      return sample(base_t::open_simplex2, fractal_t::none, x, y);
    }

    f32_t simplex(const vec2& p) const {
      return simplex(p.x, p.y);
    }

    f32_t simplex_norm(f32_t x, f32_t y) const {
      return sample_norm(base_t::open_simplex2, fractal_t::none, x, y);
    }

    f32_t simplex_norm(const vec2& p) const {
      return simplex_norm(p.x, p.y);
    }

    f32_t simplex_fbm(f32_t x, f32_t y) const {
      return sample(base_t::open_simplex2, fractal_t::fbm, x, y);
    }

    f32_t simplex_fbm(const vec2& p) const {
      return simplex_fbm(p.x, p.y);
    }

    f32_t simplex_fbm_norm(f32_t x, f32_t y) const {
      return sample_norm(base_t::open_simplex2, fractal_t::fbm, x, y);
    }

    f32_t simplex_fbm_norm(const vec2& p) const {
      return simplex_fbm_norm(p.x, p.y);
    }

    f32_t perlin(f32_t x, f32_t y) const {
      return sample(base_t::perlin, fractal_t::none, x, y);
    }

    f32_t perlin(const vec2& p) const {
      return perlin(p.x, p.y);
    }

    f32_t perlin_norm(f32_t x, f32_t y) const {
      return sample_norm(base_t::perlin, fractal_t::none, x, y);
    }

    f32_t perlin_norm(const vec2& p) const {
      return perlin_norm(p.x, p.y);
    }

    f32_t perlin_fbm(f32_t x, f32_t y) const {
      return sample(base_t::perlin, fractal_t::fbm, x, y);
    }

    f32_t perlin_fbm(const vec2& p) const {
      return perlin_fbm(p.x, p.y);
    }

    f32_t perlin_fbm_norm(f32_t x, f32_t y) const {
      return sample_norm(base_t::perlin, fractal_t::fbm, x, y);
    }

    f32_t perlin_fbm_norm(const vec2& p) const {
      return perlin_fbm_norm(p.x, p.y);
    }

    f32_t value(f32_t x, f32_t y) const {
      return sample(base_t::value, fractal_t::none, x, y);
    }

    f32_t value(const vec2& p) const {
      return value(p.x, p.y);
    }

    f32_t value_norm(f32_t x, f32_t y) const {
      return sample_norm(base_t::value, fractal_t::none, x, y);
    }

    f32_t value_norm(const vec2& p) const {
      return value_norm(p.x, p.y);
    }

    f32_t value_fbm(f32_t x, f32_t y) const {
      return sample(base_t::value, fractal_t::fbm, x, y);
    }

    f32_t value_fbm(const vec2& p) const {
      return value_fbm(p.x, p.y);
    }

    f32_t value_fbm_norm(f32_t x, f32_t y) const {
      return sample_norm(base_t::value, fractal_t::fbm, x, y);
    }

    f32_t value_fbm_norm(const vec2& p) const {
      return value_fbm_norm(p.x, p.y);
    }

    f32_t value_cubic(f32_t x, f32_t y) const {
      return sample(base_t::value_cubic, fractal_t::none, x, y);
    }

    f32_t value_cubic(const vec2& p) const {
      return value_cubic(p.x, p.y);
    }

    f32_t value_cubic_norm(f32_t x, f32_t y) const {
      return sample_norm(base_t::value_cubic, fractal_t::none, x, y);
    }

    f32_t value_cubic_norm(const vec2& p) const {
      return value_cubic_norm(p.x, p.y);
    }

    f32_t cellular(f32_t x, f32_t y) const {
      return sample(base_t::cellular, fractal_t::none, x, y);
    }

    f32_t cellular(const vec2& p) const {
      return cellular(p.x, p.y);
    }

    f32_t cellular_norm(f32_t x, f32_t y) const {
      return sample_norm(base_t::cellular, fractal_t::none, x, y);
    }

    f32_t cellular_norm(const vec2& p) const {
      return cellular_norm(p.x, p.y);
    }

    std::vector<std::uint8_t> generate_data(
      const vec2& size,
      f32_t tex_min = -1.0f,
      f32_t tex_max = 1.0f
    ) const {
      return generate_data(base, fractal, size, tex_min, tex_max);
    }

    std::vector<std::uint8_t> generate_data(
      base_t next_base,
      fractal_t next_fractal,
      const vec2& size,
      f32_t tex_min = -1.0f,
      f32_t tex_max = 1.0f
    ) const {
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

    graphics::image_t to_image(
      const vec2& size,
      f32_t tex_min = -1.0f,
      f32_t tex_max = 0.1f
    ) const {
      auto data = generate_data(size, tex_min, tex_max);
      return from_data(size, data);
    }

    graphics::image_t to_image(
      base_t next_base,
      fractal_t next_fractal,
      const vec2& size,
      f32_t tex_min = -1.0f,
      f32_t tex_max = 0.1f
    ) const {
      auto data = generate_data(next_base, next_fractal, size, tex_min, tex_max);
      return from_data(size, data);
    }

    static graphics::image_t from_data(
      const vec2& size,
      const std::vector<std::uint8_t>& data
    ) {
      graphics::image_load_properties_t lp;
      image::info_t ii;
      ii.data = (void*)data.data();
      ii.size = size;
      ii.channels = 3;
      return image_load(ii, lp);
    }

    mutable FastNoiseLite fn;
    mutable state_t state;
    base_t base = base_t::open_simplex2;
    fractal_t fractal = fractal_t::fbm;
    int seed = 1337;
    f32_t frequency = 1.f;
    f32_t gain = 0.5f;
    f32_t lacunarity = 2.0f;
    f32_t weighted_strength = 0.0f;
    f32_t ping_pong_strength = 2.0f;
    f32_t cellular_jitter = 1.0f;
    int octaves = 5;
    FastNoiseLite::CellularDistanceFunction cellular_distance_function = FastNoiseLite::CellularDistanceFunction_EuclideanSq;
    FastNoiseLite::CellularReturnType cellular_return_type = FastNoiseLite::CellularReturnType_Distance;
  };

}

#endif