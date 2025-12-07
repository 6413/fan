module;
#if defined(fan_gui)
  #include <string>
  #include <algorithm>
  #include <sstream>
#endif

export module fan.ascii_renderer;

#if defined(fan_gui)

import fan.graphics;
import fan.types.color;
import fan.graphics.gui.types;

import fan.graphics.gui.base;

export struct ascii_renderer_t {

  struct properties_t {
    f32_t font_height = 8.f;
    f32_t font_width_ratio = 0.5f;
    f32_t brightness_multiplier = 1.f;
    f32_t zoom = 1.f;
    bool enabled = true;
    static constexpr const char* ascii_gradient =
      " .`-_':,;^=+/\"|)\\<>)iv%xclrs{*}I?!][1taeo7zjLunT#JCwfy325Fh9kP6XpqAbVd4GYUEW8R0KSZDNOQ$Z@B";
  };

  ascii_renderer_t() = default;
  ascii_renderer_t(const properties_t& p) : properties(p) {}

  void render(
    const uint8_t* pixel_data,
    uint32_t width,
    uint32_t height,
    uint32_t bytes_per_pixel = 4,
    uint32_t stride = 0
  ) {
    using namespace fan::graphics;
    if (!properties.enabled || !pixel_data) return;

    if (stride == 0) stride = width * bytes_per_pixel;

    fan::vec2 ws = fan::window::get_size();
    int gw = static_cast<int>(ws.x / (properties.font_height * properties.font_width_ratio));
    int gh = static_cast<int>(ws.y / properties.font_height);

    f32_t px_per_char_x = (f32_t)width / gw / properties.zoom;
    f32_t px_per_char_y = (f32_t)height / gh / properties.zoom;

    gui::font_t* f = gui::get_font(properties.font_height);

    gui::set_next_window_pos({ 0,0 });
    gui::set_next_window_size(ws);
    gui::push_style_color(gui::col_window_bg, { 0,0,0,1 });
    gui::begin("ascii", 0,
      gui::window_flags_no_title_bar |
      gui::window_flags_no_resize |
      gui::window_flags_no_move |
      gui::window_flags_no_scrollbar |
      gui::window_flags_no_inputs
    );
    gui::push_style_var(gui::style_var_item_spacing, { 0,0 });
    gui::push_font(f);

    line_buffer.resize(gw);

    for (int y = 0; y < gh; y++) {
      for (int x = 0; x < gw; x++) {
        fan::color max_color{ 0,0,0,1 };

        for (int by = 0; by < (int)px_per_char_y; by++) {
          int py = std::clamp(int(y * px_per_char_y + by), 0, (int)height - 1);
          for (int bx = 0; bx < (int)px_per_char_x; bx++) {
            int px = std::clamp(int(x * px_per_char_x + bx), 0, (int)width - 1);
            int pi = py * stride + px * bytes_per_pixel;

            fan::color c = fan::color::rgb(pixel_data[pi + 2], pixel_data[pi + 1], pixel_data[pi + 0]);

            if (c.get_brightest_channel() > max_color.get_brightest_channel()) {
              max_color = c;
            }
          }
        }

        f32_t brightness = std::clamp(
          max_color.get_brightest_channel() * properties.brightness_multiplier,
          0.f, 1.f
        );

        line_buffer[x] = brightness_to_ascii(brightness);
      }
      gui::text(line_buffer);
    }

    gui::pop_font();
    gui::pop_style_var();
    gui::end();
    gui::pop_style_color();
  }

  char brightness_to_ascii(f32_t b) const {
    int i = static_cast<int>(b * (strlen(properties.ascii_gradient) - 1));
    return properties.ascii_gradient[std::clamp(i, 0, (int)strlen(properties.ascii_gradient) - 1)];
  }

  properties_t properties;
  std::string line_buffer;
};
#endif