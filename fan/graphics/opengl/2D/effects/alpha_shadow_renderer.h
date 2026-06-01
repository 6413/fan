struct alpha_shadow_renderer_t {
  loco_t& get_loco() { return *gloco(); }
  #define loco get_loco()

  struct light_t {
    fan::vec2 position = 0;
    f32_t radius = 512.f;
    fan::color color = fan::colors::white;
    fan::graphics::render_view_t* render_view = &fan::graphics::get_orthographic_render_view();
    f32_t softness = 0.02f;
    f32_t falloff_power = 2.f;
  };

  struct caster_t {
    fan::graphics::shape_t* shape = nullptr;
    f32_t alpha_threshold = 0.05f;
  };

  struct vertex_t {
    fan::vec2 position;
    fan::vec2 uv;
  };

  void open(std::int32_t occluder_resolution_ = 1024, std::int32_t angle_resolution_ = 2048, std::int32_t radial_samples_ = 160) {
    occluder_resolution = occluder_resolution_;
    angle_resolution    = angle_resolution_;
    radial_samples      = radial_samples_;

    auto make_shader = [&](const char* vs_path, const char* fs_path) {
      fan::graphics::shader_t nr = loco.shader_create();
      loco.shader_set_vertex(nr, vs_path, fan::graphics::read_shader(vs_path));
      loco.shader_set_fragment(nr, fs_path, fan::graphics::read_shader(fs_path));
      loco.shader_compile(nr);
      return nr;
    };

    using namespace fan::shader_paths::gl;

    occluder_shader = make_shader(alpha_shadow_quad_vs, alpha_shadow_occluder_fs);
    radial_shader   = make_shader(alpha_shadow_quad_vs, alpha_shadow_radial_fs);
    light_shader    = make_shader(alpha_shadow_quad_vs, alpha_shadow_light_fs);
    solid_shader    = make_shader(alpha_shadow_quad_vs, alpha_shadow_solid_fs);

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_t) * 6, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*)offsetof(vertex_t, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*)offsetof(vertex_t, uv));
    glBindVertexArray(0);

    create_target(occluder_fbo, occluder_texture, occluder_resolution, occluder_resolution, GL_R16F,    GL_RED,  GL_FLOAT, GL_NEAREST, GL_NEAREST, GL_CLAMP_TO_EDGE);
    create_target(shadow_fbo,   shadow_texture,   angle_resolution,    1,                   GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_NEAREST, GL_NEAREST, GL_REPEAT);
  }

  void close() {
    for (fan::graphics::shader_t s : {occluder_shader, radial_shader, light_shader, solid_shader}) {
      if (!s.iic()) { loco.shader_erase(s); }
    }
    if (occluder_texture) glDeleteTextures(1, &occluder_texture);
    if (shadow_texture)   glDeleteTextures(1, &shadow_texture);
    if (occluder_fbo)     glDeleteFramebuffers(1, &occluder_fbo);
    if (shadow_fbo)       glDeleteFramebuffers(1, &shadow_fbo);
    if (vbo)              glDeleteBuffers(1, &vbo);
    if (vao)              glDeleteVertexArrays(1, &vao);
    *this = {};
  }

  void render_overlay(std::span<const caster_t> casters, std::span<const light_t> lights, f32_t darkness = 0.78f) {
    if (occluder_shader.iic()) { open(); }

    GLint old_fbo; glGetIntegerv(GL_FRAMEBUFFER_BINDING, &old_fbo);
    GLint old_vp[4]; glGetIntegerv(GL_VIEWPORT, old_vp);
    save_blend();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glBindVertexArray(vao);

    render_darkness(darkness, old_fbo, old_vp);
    for (const light_t& light : lights) {
      render_occluders(light, casters);
      render_shadow_map();
      render_light(light, old_fbo, old_vp);
    }

    restore_blend();
    glBindFramebuffer(GL_FRAMEBUFFER, old_fbo);
    glViewport(old_vp[0], old_vp[1], old_vp[2], old_vp[3]);
  }


  struct blend_state_t {
    GLint src_rgb, dst_rgb, src_a, dst_a, eq_rgb, eq_a;
    GLboolean enabled;
  };

  blend_state_t saved_blend{};

  void save_blend() {
    saved_blend.enabled = glIsEnabled(GL_BLEND);
    glGetIntegerv(GL_BLEND_SRC_RGB,       &saved_blend.src_rgb);
    glGetIntegerv(GL_BLEND_DST_RGB,       &saved_blend.dst_rgb);
    glGetIntegerv(GL_BLEND_SRC_ALPHA,     &saved_blend.src_a);
    glGetIntegerv(GL_BLEND_DST_ALPHA,     &saved_blend.dst_a);
    glGetIntegerv(GL_BLEND_EQUATION_RGB,  &saved_blend.eq_rgb);
    glGetIntegerv(GL_BLEND_EQUATION_ALPHA,&saved_blend.eq_a);
  }

  void restore_blend() {
    if (saved_blend.enabled) glEnable(GL_BLEND); else glDisable(GL_BLEND);
    glBlendFuncSeparate(saved_blend.src_rgb, saved_blend.dst_rgb, saved_blend.src_a, saved_blend.dst_a);
    glBlendEquationSeparate(saved_blend.eq_rgb, saved_blend.eq_a);
  }

  static void create_target(std::uint32_t& fbo, std::uint32_t& tex,
    std::int32_t w, std::int32_t h,
    std::uint32_t ifmt, std::uint32_t fmt, std::uint32_t type,
    std::uint32_t min_f, std::uint32_t mag_f, std::uint32_t wrap)
  {
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_f);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_f);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      fan::throw_error_impl("alpha_shadow_renderer: framebuffer incomplete");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  static fan::vec2 rotate(fan::vec2 p, f32_t a) {
    f32_t c = std::cos(a), s = std::sin(a);
    return {p.x*c - p.y*s, p.x*s + p.y*c};
  }

  static fan::vec2 w2s(fan::vec2 p, const fan::graphics::render_view_t& rv) {
    return fan::graphics::world_to_screen(p, rv.viewport, rv.camera);
  }

  static fan::vec2 clip(fan::vec2 p) {
    fan::vec2 ws = fan::graphics::get_window().get_size();
    return {p.x / ws.x * 2.f - 1.f, 1.f - p.y / ws.y * 2.f};
  }

  void draw(const std::array<vertex_t, 6>& v) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(v), v.data());
    glDrawArrays(GL_TRIANGLES, 0, 6);
  }

  std::array<vertex_t, 6> fullscreen_quad() {
    return {{{{-1,-1},{0,0}},{{1,-1},{1,0}},{{1,1},{1,1}},{{-1,-1},{0,0}},{{1,1},{1,1}},{{-1,1},{0,1}}}};
  }

  void render_darkness(f32_t alpha, GLint fbo, const GLint vp[4]) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    loco.shader_use(solid_shader);
    loco.shader_set_value(solid_shader, "color", fan::color(0, 0, 0, alpha));
    draw(fullscreen_quad());
  }

  void render_occluders(const light_t& light, std::span<const caster_t> casters) {
    fan::vec2 ls = w2s(light.position, *light.render_view);
    fan::vec2 le = w2s(light.position + fan::vec2(light.radius, 0), *light.render_view);
    f32_t lr = std::max(1.f, std::abs(le.x - ls.x));

    glBindFramebuffer(GL_FRAMEBUFFER, occluder_fbo);
    glViewport(0, 0, occluder_resolution, occluder_resolution);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);

    loco.shader_use(occluder_shader);
    loco.shader_set_value(occluder_shader, "sprite_texture", 0);

    for (const caster_t& caster : casters) {
      if (!caster.shape || !*caster.shape) continue;
      fan::graphics::texture_pack::ti_t ti = caster.shape->get_tp();
      if (!ti.image.valid()) continue;
      fan::vec2 isz = ti.image.get_size();
      if (isz.x <= 0 || isz.y <= 0) continue;

      fan::vec2 pos   = caster.shape->get_position();
      fan::vec2 size  = caster.shape->get_size();
      fan::vec2 pivot = caster.shape->get_rotation_point();
      f32_t angle     = caster.shape->get_angle().z;

      auto mp = [&](fan::vec2 local) {
        fan::vec2 world = pos + pivot + rotate(local - pivot, angle);
        fan::vec2 sp = w2s(world, *light.render_view);
        fan::vec2 p = (sp - ls) / lr;
        return fan::vec2(p.x, -p.y);
      };

      fan::vec2 uv0 = ti.position / isz;
      fan::vec2 uv1 = uv0 + ti.size / isz;

      std::array<vertex_t, 6> verts{{
        {mp({-size.x,-size.y}), {uv0.x,uv0.y}},
        {mp({ size.x,-size.y}), {uv1.x,uv0.y}},
        {mp({ size.x, size.y}), {uv1.x,uv1.y}},
        {mp({-size.x,-size.y}), {uv0.x,uv0.y}},
        {mp({ size.x, size.y}), {uv1.x,uv1.y}},
        {mp({-size.x, size.y}), {uv0.x,uv1.y}},
      }};

      bool outside = true;
      for (auto& v : verts) {
        if (std::abs(v.position.x) <= 1.25f && std::abs(v.position.y) <= 1.25f) { outside = false; break; }
      }
      if (outside) continue;

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, ti.image.get_handle());
      loco.shader_set_value(occluder_shader, "alpha_threshold", caster.alpha_threshold);
      draw(verts);
    }
  }

  void render_shadow_map() {
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo);
    glViewport(0, 0, angle_resolution, 1);
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_BLEND);
    loco.shader_use(radial_shader);
    loco.shader_set_value(radial_shader, "occluder_texture", 0);
    loco.shader_set_value(radial_shader, "radial_samples", radial_samples);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, occluder_texture);
    draw(fullscreen_quad());
  }

  void render_light(const light_t& light, GLint fbo, const GLint vp[4]) {
    fan::vec2 center = w2s(light.position, *light.render_view);
    fan::vec2 edge   = w2s(light.position + fan::vec2(light.radius, 0), *light.render_view);
    f32_t r = std::max(1.f, std::abs(edge.x - center.x));
    fan::vec2 p0 = center - r, p1 = center + r;

    std::array<vertex_t, 6> verts{{
      {clip({p0.x,p1.y}), {0,0}}, {clip({p1.x,p1.y}), {1,0}}, {clip({p1.x,p0.y}), {1,1}},
      {clip({p0.x,p1.y}), {0,0}}, {clip({p1.x,p0.y}), {1,1}}, {clip({p0.x,p0.y}), {0,1}},
    }};

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(vp[0], vp[1], vp[2], vp[3]);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE);

    loco.shader_use(light_shader);
    loco.shader_set_value(light_shader, "shadow_texture",  0);
    loco.shader_set_value(light_shader, "light_color",     light.color);
    loco.shader_set_value(light_shader, "softness",        light.softness);
    loco.shader_set_value(light_shader, "falloff_power",   light.falloff_power);
    loco.shader_set_value(light_shader, "angle_texel",     1.f / f32_t(angle_resolution));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, shadow_texture);
    draw(verts);
  }

  std::int32_t occluder_resolution = 1024;
  std::int32_t angle_resolution    = 2048;
  std::int32_t radial_samples      = 160;

  std::uint32_t occluder_fbo = 0, occluder_texture = 0;
  std::uint32_t shadow_fbo   = 0, shadow_texture   = 0;
  std::uint32_t vao = 0, vbo = 0;

  fan::graphics::shader_t occluder_shader;
  fan::graphics::shader_t radial_shader;
  fan::graphics::shader_t light_shader;
  fan::graphics::shader_t solid_shader;

  std::vector<caster_t> casters;
  std::vector<light_t> lights;
  f32_t darkness = 0.78f;

  #undef loco
};