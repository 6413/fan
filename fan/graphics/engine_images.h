image_t default_texture;

struct icons_t {
  image_t play;
  image_t pause;
  image_t settings;
}icons;

void load_engine_images() {
  default_texture = create_missing_texture();

  icons.play = image_load("icons/play.png");
  icons.pause = image_load("icons/pause.png");
  icons.settings = image_load("icons/settings.png");
}