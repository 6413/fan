image_t default_texture;

void load_engine_images() {
  default_texture = create_missing_texture();

  fan::graphics::icons.play = image_load("icons/play.png");
  fan::graphics::icons.pause = image_load("icons/pause.png");
  fan::graphics::icons.settings = image_load("icons/settings.png");
}

void unload_engine_images() {
  image_unload(default_texture);

  image_unload(fan::graphics::icons.play);
  image_unload(fan::graphics::icons.pause);
  image_unload(fan::graphics::icons.settings);

#if defined(fan_opengl)
  for (auto& i : gl.color_buffers) {
    image_unload(i);
  }
#endif
}