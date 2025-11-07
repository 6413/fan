image_t default_texture;

void load_engine_images() {
  default_texture = create_missing_texture();

  fan::graphics::icons.play = image_load("icons/play.png");
  fan::graphics::icons.pause = image_load("icons/pause.png");
  fan::graphics::icons.settings = image_load("icons/settings.png");
}