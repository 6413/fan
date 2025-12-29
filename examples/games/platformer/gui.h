#define gui fan::graphics::gui

void open(void* sod) {
  health_empty = fan::graphics::image_t("gui/hp_empty.png");
  health_full = fan::graphics::image_t("gui/hp_full.png");
  health_potion = fan::graphics::image_t("gui/health_potion.png");
  gui::load_fonts(font_pixel, "fonts/PixelatedEleganceRegular-ovyAA.ttf");
}

void close() {

}

void update() {
  using namespace fan::graphics;
  fan::vec2 wnd_size = fan::window::get_size();

  f32_t heart_size = (wnd_size / 32.f).max();
  f32_t potion_size = (wnd_size / 48.f).max();

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
    gui::image(hp_image, heart_size);
  }
  auto cp = 0;
  auto& v = gui::get_style();
  gui::set_cursor_pos_x(cp + v.ItemSpacing.x + heart_size / 2.f - potion_size / 2.f);
  gui::image(health_potion, potion_size);
  gui::same_line();
  gui::push_font(gui::get_font(font_pixel, gui::get_font_size()));
  std::string potion_text = "x " + std::to_string(pile->player.potion_count);
  f32_t text_height = gui::get_text_size(potion_text).y;
  gui::set_cursor_pos_y(gui::get_cursor_pos_y() + potion_size - text_height);
  gui::text(potion_text);
  gui::pop_font();
 // gui::end();
}

fan::graphics::image_t health_empty;
fan::graphics::image_t health_full;
fan::graphics::image_t health_potion;

gui::font_t* font_pixel[std::size(gui::font_sizes)]{};
#undef gui