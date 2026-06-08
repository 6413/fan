struct shaders_t {
  // 2D objects
  fan::graphics::shader_t capsule;
  fan::graphics::shader_t circle;
  fan::graphics::shader_t grid;
  fan::graphics::shader_t light;
  fan::graphics::shader_t line;
  fan::graphics::shader_t polygon;
  fan::graphics::shader_t rectangle;
  fan::graphics::shader_t shadow;
  fan::graphics::shader_t sprite;
  fan::graphics::shader_t unlit_sprite;
  fan::graphics::shader_t universal_image_renderer;
  // 2D effects
  fan::graphics::shader_t gradient;
  fan::graphics::shader_t particles;
  fan::graphics::shader_t shader_shape;
#if defined(LOCO_FRAMEBUFFER)
  fan::graphics::shader_t downsample;
  fan::graphics::shader_t final;
  fan::graphics::shader_t reflection;
  fan::graphics::shader_t upsample;
  fan::graphics::shader_t clouds;
#endif
  // misc
  fan::graphics::shader_t empty_shader;
#if defined(FAN_3D)
  fan::graphics::shader_t line3d;
  fan::graphics::shader_t rectangle3d;
#endif
};