import fan;

struct particle_editor_t {
  fan::graphics::engine_t& engine;
  fan::graphics::fan::graphics::shape_t particle_shape;
  fan::color bg_color;
  fan::color base_color;
  f32_t color_intensity;
  
  fan::graphics::file_save_dialog_t save_file_dialog;
  fan::graphics::file_open_dialog_t open_file_dialog;
  std::string filename;

  fan::graphics::engine_t::particles_t::ri_t& get_ri() {
    return *(fan::graphics::engine_t::particles_t::ri_t*)particle_shape.GetData(engine.shaper);
  }

  void handle_file_operations() {
    if (open_file_dialog.is_finished()) {
      if (filename.size() != 0) {
        std::string data;
        fan::io::file::read(filename, &data);
        particle_shape = fan::json::parse(data);
      }
      open_file_dialog.finished = false;
    }

    if (save_file_dialog.is_finished()) {
      if (filename.size() != 0) {
        fan::json json_data = particle_shape;
        fan::io::file::write(filename, json_data.dump(2), std::ios_base::binary);
      }
      save_file_dialog.finished = false;
    }
  }

  void render_menu() {
    if (fan::graphics::gui::begin_main_menu_bar()) {
      if (fan::graphics::gui::begin_menu("File")) {
        if (fan::graphics::gui::menu_item("Open..", "Ctrl+O")) {
          open_file_dialog.load("json;fmm", &filename);
        }
        if (fan::graphics::gui::menu_item("Save as", "Ctrl+Shift+S")) {
          save_file_dialog.save("json;fmm", &filename);
        }
        fan::graphics::gui::end_menu();
      }
      fan::graphics::gui::end_main_menu_bar();
    }
  }

  void render_settings() {
    fan::graphics::gui::begin("particle settings");
    fan::graphics::gui::color_edit4("background color", &bg_color);
    fan::graphics::gui::shape_properties(particle_shape);
    fan::graphics::gui::end();
  }
};

int main() {
  fan::graphics::engine_t engine;
  engine.clear_color = fan::colors::black;
  engine.set_vsync(0);

  fan::color bg_color = fan::color::from_rgba(0xB8C4BFFF);
  fan::color particle_color = fan::color::from_rgba(0x33333369);
  auto particle_texture = engine.image_load("images/waterdrop.webp");
  fan::vec2 window_size = engine.window.get_size();

  fan::graphics::engine_t::particles_t::properties_t p;
  p.position = fan::vec3(32.108001708984375, -1303.083984375, 10.0);
  p.color = particle_color;
  p.count = 1191;
  p.size = 28.63800048828125;
  p.begin_angle = 0;
  p.end_angle = -0.1599999964237213;
  p.angle = fan::vec3(0.0, 0.0, -0.49399998784065247);
  p.alive_time = 1768368768;
  p.gap_size = fan::vec2(400.89898681640625, 1.0);
  p.max_spread_size = fan::vec2(2648.02099609375, 1.0);
  p.shape = fan::graphics::engine_t::particles_t::shapes_e::rectangle;
  p.position_velocity = fan::vec2(0.0, 9104.126953125);
  p.image = particle_texture;

  particle_editor_t editor{engine, p, bg_color, particle_color, 1.0f, {}, {}, {}};

  engine.loop([&] {
    fan::graphics::rectangle({ 
      .position=fan::vec3(engine.window.get_size() / 2.f, 0), 
      .size = engine.window.get_size() / 2.f, 
      .color = editor.bg_color
    });

    editor.render_menu();
    editor.handle_file_operations();
    editor.render_settings();
  });
}