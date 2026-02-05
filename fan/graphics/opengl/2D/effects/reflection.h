struct reflection_t {
  loco_t& get_loco() {
    return (*OFFSETLESS(this, loco_t, gl.reflection));
  }
#define loco get_loco()
  
  void open() {
    shader = loco.shader_create();
    loco.shader_set_vertex(shader, fan::graphics::read_shader("shaders/opengl/2D/effects/reflection.vs"));
    loco.shader_set_fragment(shader, fan::graphics::read_shader("shaders/opengl/2D/effects/reflection.fs"));
    loco.shader_compile(shader);
    loco.shader_set_paths(
      shader,
      "shaders/opengl/2D/effects/reflection.vs", 
      "shaders/opengl/2D/effects/reflection.fs"
    );
  }
  
  void close() {
    loco.shader_erase(shader);
  }
  
  void draw() {
    if (!enabled) {
      return;
    }
    
    loco.shader_use(shader);
    
    loco.shader_set_value(shader, "_t00", 0);
    loco.shader_set_value(shader, "water_y", water_y);
    loco.shader_set_value(shader, "water_height", water_height);
    loco.shader_set_value(shader, "alpha", alpha);
    loco.shader_set_value(shader, "_time", (f32_t)((fan::time::now() - loco.start_time.m_time) / 1e+9));
    loco.shader_set_value(shader, "window_size", loco.window.get_size());
    
    loco.shader_set_value(shader, "distortion_strength", distortion_strength);
    loco.shader_set_value(shader, "distortion_speed", distortion_speed);
    loco.shader_set_value(shader, "distortion_scale", distortion_scale);
    loco.shader_set_value(shader, "distortion_octaves", distortion_octaves);
    loco.shader_set_value(shader, "distortion_lacunarity", distortion_lacunarity);
    loco.shader_set_value(shader, "distortion_gain", distortion_gain);
    
    loco.shader_set_value(shader, "wave_amplitude", wave_amplitude);
    loco.shader_set_value(shader, "wave_frequency", wave_frequency);
    loco.shader_set_value(shader, "wave_speed", wave_speed);
    
    loco.shader_set_value(shader, "caustic_strength", caustic_strength);
    loco.shader_set_value(shader, "caustic_speed", caustic_speed);
    loco.shader_set_value(shader, "caustic_scale", caustic_scale);
    
    loco.shader_set_value(shader, "highlight_strength", highlight_strength);
    loco.shader_set_value(shader, "highlight_power", highlight_power);
    loco.shader_set_value(shader, "shimmer_strength", shimmer_strength);
    loco.shader_set_value(shader, "depth_fade_power", depth_fade_power);
    loco.shader_set_value(shader, "tint_strength", tint_strength);
    
    loco.shader_set_value(shader, "shallow_color", shallow_color);
    loco.shader_set_value(shader, "deep_color", deep_color);
    
    using namespace fan::graphics::gui;
    
    if (show_controls) {
      text("=== Water Position ===");
      drag("Water Y##water_pos", &water_y, 0.01f);
      drag("Water Height##water_pos", &water_height, 0.01f);
      drag("Alpha##water_pos", &alpha, 0.01f);

      separator();
      text("=== Distortion ===");
      drag("Strength##distortion", &distortion_strength, 0.001f);
      drag("Speed##distortion", &distortion_speed, 0.01f);
      drag("Scale##distortion", &distortion_scale, 0.1f);
      drag("Octaves##distortion", &distortion_octaves, 0.1f, 1.0f, 8.0f);
      drag("Lacunarity##distortion", &distortion_lacunarity, 0.01f);
      drag("Gain##distortion", &distortion_gain, 0.01f);

      separator();
      text("=== Waves ===");
      drag("Amplitude##wave", &wave_amplitude, 0.001f);
      drag("Frequency##wave", &wave_frequency, 0.1f);
      drag("Speed##wave", &wave_speed, 0.1f);

      separator();
      text("=== Caustics ===");
      drag("Strength##caustic", &caustic_strength, 0.01f);
      drag("Speed##caustic", &caustic_speed, 0.01f);
      drag("Scale##caustic", &caustic_scale, 0.1f);

      separator();
      text("=== Visual Effects ===");
      drag("Highlight Strength##visual", &highlight_strength, 0.01f);
      drag("Highlight Power##visual", &highlight_power, 0.1f);
      drag("Shimmer Strength##visual", &shimmer_strength, 0.01f);
      drag("Depth Fade Power##visual", &depth_fade_power, 0.1f);
      drag("Tint Strength##visual", &tint_strength, 0.1f);

      separator();
      text("=== Colors ===");
      color_edit4("Shallow Color", &shallow_color);
      color_edit4("Deep Color", &deep_color);
    }
    
    fan_opengl_call(glActiveTexture(GL_TEXTURE0));
    loco.image_bind(loco.gl.color_buffers[0]);
    
    fan_opengl_call(glEnable(GL_BLEND));
    fan_opengl_call(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    fan_opengl_call(glBindVertexArray(loco.gl.fb_vao));
    fan_opengl_call(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
    fan_opengl_call(glBindVertexArray(0));
  }
  
  fan::graphics::shader_t shader;
  
  f32_t water_y = 0.5f;
  f32_t water_height = 0.5f;
  f32_t alpha = 0.85f;

  f32_t distortion_strength = 0.037f;
  f32_t distortion_speed = 0.120f;
  f32_t distortion_scale = 6.5f;
  f32_t distortion_octaves = 6.0f;
  f32_t distortion_lacunarity = 1.95f;
  f32_t distortion_gain = 0.86f;

  f32_t wave_amplitude = 0.022f;
  f32_t wave_frequency = 35.8f;
  f32_t wave_speed = 0.1f;

  f32_t caustic_strength = 0.11f;
  f32_t caustic_speed = 1.56f;
  f32_t caustic_scale = 0.5f;

  f32_t highlight_strength = 0.15f;
  f32_t highlight_power = 2.0f;
  f32_t shimmer_strength = 0.12f;
  f32_t depth_fade_power = 1.2f;
  f32_t tint_strength = 0.5f;

  fan::color shallow_color{0.95f, 0.97f, 1.0f};
  fan::color deep_color{0.7f, 0.85f, 0.95f};

  bool enabled = false;
  bool show_controls = false;
  
#undef loco
};