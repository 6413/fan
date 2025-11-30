void open(void* sod) {
  health_empty = fan::graphics::image_t("gui/hp_empty.png");
  health_full = fan::graphics::image_t("gui/hp_full.png");
}

void close() {

}

void update() {
  using namespace fan::graphics;
  fan::vec2 wnd_size = fan::window::get_size();
  //gui::set_next_window_pos(0);
  //gui::set_next_window_size(wnd_size);
  //gui::begin("##platformer_gui", nullptr,
  //  gui::window_flags_no_background | gui::window_flags_no_nav |
  //  gui::window_flags_no_title_bar
  //);
  int heart_count = pile->player.body.get_max_health() / 10.f;
  for (int i = 0; i < heart_count; ++i) {
    gui::same_line();
    fan::graphics::image_t hp_image = health_empty;
    //0-1
    f32_t progress = pile->player.body.get_health() / pile->player.body.get_max_health();
    if (progress * heart_count > i) {
      hp_image = health_full;
    }
    gui::image(hp_image, (wnd_size / 32.f).max());
  }
 // gui::end();
}

fan::graphics::image_t health_empty;
fan::graphics::image_t health_full;