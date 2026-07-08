module;

export module fan.noise;

#if defined(FAN_WINDOW)

import std;

import fan.types;
import fan.types.vector;
import fan.graphics.common_context;

export namespace fan {

  struct noise_t {
    enum class base_t {
      open_simplex2,
      open_simplex2s,
      cellular,
      perlin,
      value_cubic,
      value
    };

    enum class fractal_t {
      none,
      fbm,
      ridged,
      ping_pong
    };

    enum class cellular_distance_function_t {
      euclidean,
      euclidean_sq,
      manhattan,
      hybrid
    };

    enum class cellular_return_type_t {
      cell_value,
      distance,
      distance2,
      distance2_add,
      distance2_sub,
      distance2_mul,
      distance2_div
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
      cellular_distance_function_t cellular_distance_function = cellular_distance_function_t::euclidean_sq;
      cellular_return_type_t cellular_return_type = cellular_return_type_t::distance;
    };

    noise_t(int seed = 1337);
    ~noise_t();
    noise_t(noise_t&&) noexcept;
    noise_t& operator=(noise_t&&) noexcept;

    static f32_t fade(f32_t t);
    static f32_t lerp(f32_t a, f32_t b, f32_t t);
    static std::uint32_t hash(std::int32_t x, std::int32_t y, std::uint32_t seed);

    bool is_applied(base_t next_base, fractal_t next_fractal, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const;
    bool is_applied(base_t next_base, fractal_t next_fractal) const;
    
    void apply() const;
    void apply(base_t next_base, fractal_t next_fractal) const;
    void apply(base_t next_base, fractal_t next_fractal, int next_octaves, f32_t next_lacunarity, f32_t next_gain) const;

    f32_t value_noise(f32_t x, f32_t y) const;
    f32_t value_noise(const vec2& p) const;

    f32_t sample(base_t next_base, fractal_t next_fractal, f32_t x, f32_t y) const;
    f32_t sample(base_t next_base, fractal_t next_fractal, const vec2& p) const;
    f32_t sample(base_t next_base, fractal_t next_fractal, f32_t x, f32_t y, int next_octaves, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const;
    f32_t sample(base_t next_base, fractal_t next_fractal, const vec2& p, int next_octaves, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const;
    f32_t sample(f32_t x, f32_t y) const;
    f32_t sample(const vec2& p) const;

    f32_t sample_norm(base_t next_base, fractal_t next_fractal, f32_t x, f32_t y) const;
    f32_t sample_norm(base_t next_base, fractal_t next_fractal, const vec2& p) const;
    f32_t sample_norm(base_t next_base, fractal_t next_fractal, f32_t x, f32_t y, int next_octaves, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const;
    f32_t sample_norm(base_t next_base, fractal_t next_fractal, const vec2& p, int next_octaves, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const;
    f32_t sample_norm(f32_t x, f32_t y) const;
    f32_t sample_norm(const vec2& p) const;

    f32_t get_noise(f32_t x, f32_t y) const;
    f32_t get_noise(const vec2& p) const;

    f32_t fbm(f32_t x, f32_t y, int next_octaves = 5, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const;
    f32_t fbm(const vec2& p, int next_octaves = 5, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const;
    f32_t fbm_norm(f32_t x, f32_t y, int next_octaves = 5, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const;
    f32_t fbm_norm(const vec2& p, int next_octaves = 5, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const;
    f32_t fbm_raw(f32_t x, f32_t y, int next_octaves = 5, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const;
    f32_t fbm_raw(const vec2& p, int next_octaves = 5, f32_t next_lacunarity = 2.f, f32_t next_gain = 0.5f) const;

    f32_t ridged(f32_t x, f32_t y) const;
    f32_t ridged(const vec2& p) const;
    f32_t ridged_norm(f32_t x, f32_t y) const;
    f32_t ridged_norm(const vec2& p) const;

    f32_t ping_pong(f32_t x, f32_t y) const;
    f32_t ping_pong(const vec2& p) const;
    f32_t ping_pong_norm(f32_t x, f32_t y) const;
    f32_t ping_pong_norm(const vec2& p) const;

    f32_t simplex(f32_t x, f32_t y) const;
    f32_t simplex(const vec2& p) const;
    f32_t simplex_norm(f32_t x, f32_t y) const;
    f32_t simplex_norm(const vec2& p) const;

    f32_t simplex_fbm(f32_t x, f32_t y) const;
    f32_t simplex_fbm(const vec2& p) const;
    f32_t simplex_fbm_norm(f32_t x, f32_t y) const;
    f32_t simplex_fbm_norm(const vec2& p) const;

    f32_t perlin(f32_t x, f32_t y) const;
    f32_t perlin(const vec2& p) const;
    f32_t perlin_norm(f32_t x, f32_t y) const;
    f32_t perlin_norm(const vec2& p) const;

    f32_t perlin_fbm(f32_t x, f32_t y) const;
    f32_t perlin_fbm(const vec2& p) const;
    f32_t perlin_fbm_norm(f32_t x, f32_t y) const;
    f32_t perlin_fbm_norm(const vec2& p) const;

    f32_t value(f32_t x, f32_t y) const;
    f32_t value(const vec2& p) const;
    f32_t value_norm(f32_t x, f32_t y) const;
    f32_t value_norm(const vec2& p) const;

    f32_t value_fbm(f32_t x, f32_t y) const;
    f32_t value_fbm(const vec2& p) const;
    f32_t value_fbm_norm(f32_t x, f32_t y) const;
    f32_t value_fbm_norm(const vec2& p) const;

    f32_t value_cubic(f32_t x, f32_t y) const;
    f32_t value_cubic(const vec2& p) const;
    f32_t value_cubic_norm(f32_t x, f32_t y) const;
    f32_t value_cubic_norm(const vec2& p) const;

    f32_t cellular(f32_t x, f32_t y) const;
    f32_t cellular(const vec2& p) const;
    f32_t cellular_norm(f32_t x, f32_t y) const;
    f32_t cellular_norm(const vec2& p) const;

    std::vector<std::uint8_t> generate_data(const vec2& size, f32_t tex_min = -1.0f, f32_t tex_max = 1.0f) const;
    std::vector<std::uint8_t> generate_data(base_t next_base, fractal_t next_fractal, const vec2& size, f32_t tex_min = -1.0f, f32_t tex_max = 1.0f) const;

    graphics::image_t to_image(const vec2& size, f32_t tex_min = -1.0f, f32_t tex_max = 0.1f) const;
    graphics::image_t to_image(base_t next_base, fractal_t next_fractal, const vec2& size, f32_t tex_min = -1.0f, f32_t tex_max = 0.1f) const;
    static graphics::image_t from_data(const vec2& size, const std::vector<std::uint8_t>& data);

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
    cellular_distance_function_t cellular_distance_function = cellular_distance_function_t::euclidean_sq;
    cellular_return_type_t cellular_return_type = cellular_return_type_t::distance;

  private:
    struct impl_t;
    std::unique_ptr<impl_t> impl;
  };

}

#endif