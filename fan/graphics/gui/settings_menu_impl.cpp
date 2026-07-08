module;

#if defined (FAN_WINDOW)

#include <fan/utility.h>

#if defined(FAN_GUI)
#endif

#endif

module fan.graphics.gui.settings_menu;

#if defined (FAN_WINDOW)

#if defined(FAN_GUI)

import fan.types.vector;
import fan.types.json;
import fan.graphics.gui.keybinds_menu;
#if defined(FAN_AUDIO)
  import fan.audio;
#endif
import fan.memory;
import fan.io.file;
import fan.graphics.loco;
import fan.graphics.gui.base;
import fan.graphics.culling;

namespace fan::graphics::gui {
  struct keybind_settings_bridge_t {
    static void menu_left(settings_menu_t* settings_menu, const fan::vec2& pos, const fan::vec2& size) {
      keybind_menu_t::menu_keybinds_left(&OFFSETLESS(settings_menu, settings_menu_t, keybind_menu)->keybind_menu, pos, size);
    }
    static void menu_right(settings_menu_t* settings_menu, const fan::vec2& pos, const fan::vec2& size) {
      keybind_menu_t::menu_keybinds_right(&OFFSETLESS(settings_menu, settings_menu_t, keybind_menu)->keybind_menu, pos, size);
    }
  };

  static void sync_vulkan_post_processing(const settings_config_t::post_processing_t& pp) {
    *gloco()->get_bloom_strength_ptr()      = pp.bloom_strength;
    *gloco()->get_bloom_threshold_ptr()     = pp.bloom_threshold;
    *gloco()->get_bloom_knee_ptr()          = pp.bloom_knee;
    *gloco()->get_bloom_filter_radius_ptr() = pp.bloom_filter_radius;
    *gloco()->get_bloom_tint_ptr()          = pp.bloom_tint;
    *gloco()->get_gamma_ptr()               = pp.gamma;
    *gloco()->get_exposure_ptr()            = pp.exposure;
    *gloco()->get_contrast_ptr()            = pp.contrast;
  }

  void settings_config_t::load_from_json(const fan::json& j) {
    if (j.contains("display")) {
      const auto& d = j["display"];
      d.get_if("display_mode", display.display_mode);
      d.get_if("target_fps", display.target_fps);
      d.get_if("resolution_index", display.resolution_index);
      d.get_if("renderer", display.renderer);
      if (d.contains("window_position")) {
        display.window_position.x = d["window_position"]["x"];
        display.window_position.y = d["window_position"]["y"];
      }
      if (d.contains("custom_resolution")) {
        display.custom_resolution.x = d["custom_resolution"]["x"];
        display.custom_resolution.y = d["custom_resolution"]["y"];
      }
    }
    if (j.contains("performance")) {
      const auto& p = j["performance"];
      p.get_if("vsync", performance.vsync);
      p.get_if("show_fps", performance.show_fps);
      p.get_if("track_heap", performance.track_heap);
      p.get_if("track_opengl_calls", performance.track_opengl_calls);
    }
    if (j.contains("debug")) {
      const auto& d = j["debug"];
      d.get_if("frustum_culling", debug.frustum_culling_enabled);
      d.get_if("visualize_culling", debug.visualize_culling);
      d.get_if("settings_bg_alpha", debug.settings_bg_alpha);
      d.get_if("fill_mode", debug.fill_mode);
      if (d.contains("culling_padding")) {
        debug.culling_padding.x = d["culling_padding"]["x"];
        debug.culling_padding.y = d["culling_padding"]["y"];
      }
    }
    if (j.contains("audio")) {
      j["audio"].get_if("volume", audio.volume);
    }
    if (j.contains("post_processing")) {
      const auto& pp = j["post_processing"];
      pp.get_if("mode", post_processing.mode);
      if (pp.contains("clear_color")) {
        pp.get_if("clear_color", post_processing.clear_color);
      }
      else {
        post_processing.clear_color = gloco()->get_clear_color();
      }
      post_processing.clear_color.a = 1.f;
      if (pp.contains("ambient_color")) {
        pp.get_if("ambient_color", post_processing.ambient_color);
      }
      else {
        post_processing.ambient_color = gloco()->get_lighting().ambient;
      }
      if (!pp.contains("mode")) {
        bool bloom = false;
        pp.get_if("bloom", bloom);
        post_processing.mode = bloom ? 1 : 0;
      }
      if (post_processing.mode < 0 || post_processing.mode > 3) {
        post_processing.mode = 0;
      }
      gloco()->open_props.post_process_mode = (fan::graphics::post_process_mode_e)post_processing.mode;
      pp.get_if("bloom_strength", post_processing.bloom_strength);
      pp.get_if("bloom_threshold", post_processing.bloom_threshold);
      pp.get_if("bloom_knee", post_processing.bloom_knee);
      pp.get_if("bloom_tint", post_processing.bloom_tint);
      pp.get_if("bloom_filter_radius", post_processing.bloom_filter_radius);
      pp.get_if("blur_amount", post_processing.blur_amount);
      pp.get_if("blur_filter_radius", post_processing.blur_filter_radius);
      pp.get_if("blur_focus_enabled", post_processing.blur_focus_enabled);
      pp.get_if("blur_focus_follow_mouse", post_processing.blur_focus_follow_mouse);
      if (pp.contains("blur_focus_position")) {
        post_processing.blur_focus_position.x = pp["blur_focus_position"]["x"];
        post_processing.blur_focus_position.y = pp["blur_focus_position"]["y"];
      }
      pp.get_if("blur_focus_radius", post_processing.blur_focus_radius);
      pp.get_if("blur_focus_falloff", post_processing.blur_focus_falloff);
      pp.get_if("gamma", post_processing.gamma);
      pp.get_if("exposure", post_processing.exposure);
      pp.get_if("contrast", post_processing.contrast);
      gloco()->open_props.blur_amount = std::clamp(post_processing.blur_amount, 0.f, 1.f);
      gloco()->open_props.blur_filter_radius = post_processing.blur_filter_radius;
      gloco()->open_props.blur_focus_enabled = post_processing.blur_focus_enabled;
      gloco()->open_props.blur_focus_follow_mouse = post_processing.blur_focus_follow_mouse;
      gloco()->open_props.blur_focus_position = post_processing.blur_focus_position;
      gloco()->open_props.blur_focus_radius = post_processing.blur_focus_radius;
      gloco()->open_props.blur_focus_falloff = post_processing.blur_focus_falloff;
    }
  }

  fan::json settings_config_t::to_json() const {
    fan::json j;

    j["display"]["display_mode"] = display.display_mode;
    j["display"]["target_fps"] = display.target_fps;
    j["display"]["resolution_index"] = display.resolution_index;
    j["display"]["renderer"] = display.renderer;
    j["display"]["window_position"]["x"] = display.window_position.x;
    j["display"]["window_position"]["y"] = display.window_position.y;
    j["display"]["custom_resolution"]["x"] = display.custom_resolution.x;
    j["display"]["custom_resolution"]["y"] = display.custom_resolution.y;

    j["performance"]["vsync"] = performance.vsync;
    j["performance"]["show_fps"] = performance.show_fps;
    j["performance"]["track_heap"] = performance.track_heap;
    j["performance"]["track_opengl_calls"] = performance.track_opengl_calls;

    j["debug"]["frustum_culling"] = debug.frustum_culling_enabled;
    j["debug"]["visualize_culling"] = debug.visualize_culling;
    j["debug"]["settings_bg_alpha"] = debug.settings_bg_alpha;
    j["debug"]["fill_mode"] = debug.fill_mode;
    j["debug"]["culling_padding"]["x"] = debug.culling_padding.x;
    j["debug"]["culling_padding"]["y"] = debug.culling_padding.y;

    j["audio"]["volume"] = audio.volume;

    j["post_processing"]["mode"] = (int)gloco()->open_props.post_process_mode;
    fan::color clear_color = post_processing.clear_color;
    clear_color.a = 1.f;
    j["post_processing"]["clear_color"] = clear_color;
    j["post_processing"]["ambient_color"] = post_processing.ambient_color;
    j["post_processing"]["bloom_strength"] = post_processing.bloom_strength;
    j["post_processing"]["bloom_threshold"] = post_processing.bloom_threshold;
    j["post_processing"]["bloom_knee"] = post_processing.bloom_knee;
    j["post_processing"]["bloom_tint"] = post_processing.bloom_tint;
    j["post_processing"]["gamma"] = post_processing.gamma;
    j["post_processing"]["bloom_filter_radius"] = post_processing.bloom_filter_radius;
    j["post_processing"]["blur_amount"] = post_processing.blur_amount;
    j["post_processing"]["blur_filter_radius"] = post_processing.blur_filter_radius;
    j["post_processing"]["blur_focus_enabled"] = post_processing.blur_focus_enabled;
    j["post_processing"]["blur_focus_follow_mouse"] = post_processing.blur_focus_follow_mouse;
    j["post_processing"]["blur_focus_position"]["x"] = post_processing.blur_focus_position.x;
    j["post_processing"]["blur_focus_position"]["y"] = post_processing.blur_focus_position.y;
    j["post_processing"]["blur_focus_radius"] = post_processing.blur_focus_radius;
    j["post_processing"]["blur_focus_falloff"] = post_processing.blur_focus_falloff;
    j["post_processing"]["exposure"] = post_processing.exposure;
    j["post_processing"]["contrast"] = post_processing.contrast;

    return j;
  }

  settings_menu_t::settings_menu_t() {
    load();
    apply_config(true, false);

    page_t page;
    page.toggle = true;
    page.name = "Graphics";
    page.render_page_left = menu_graphics_left;
    page.render_page_right = menu_graphics_right;
    pages.emplace_back(page);

    page.toggle = false;
    page.name = "Audio";
    page.render_page_left = menu_audio_left;
    page.render_page_right = menu_audio_right;
    pages.emplace_back(page);

    page.name = "Keybinds";
    page.render_page_left = keybind_settings_bridge_t::menu_left;
    page.render_page_right = keybind_settings_bridge_t::menu_right;
    page.split_ratio = 0.5f;
    pages.emplace_back(page);

    keybind_menu.save_cb = [this]() { save(); };
    keybind_menu.mark_dirty_cb = [this]() { mark_dirty(); };
  }

  settings_menu_t::~settings_menu_t() {
    if (is_dirty) {
      save();
    }
  }

  void settings_menu_t::init_runtime() {
    set_settings_theme();
    keybind_menu.sync_from_input_action();
    query_current_resolution();

    if (gloco()->target_fps == config.display.target_fps) {
      config.display.target_fps = gloco()->target_fps;
    }

    apply_config(false, true);

    resize_handle = gloco()->on_resize(
      [this](const fan::window_t::resize_data_t& r) {
      on_window_resize(r.size);
      if (gloco()->window.display_mode == fan::window_t::mode::windowed) {
        config.display.custom_resolution = r.size;
        config.display.resolution_index = -1;
        mark_dirty();
      }
    }
    );
    move_handle = gloco()->window.add_move_callback(
      [this](const auto& d) {
      if (d.window->display_mode == fan::window_t::mode::windowed) {
        config.display.window_position = d.position;
        mark_dirty();
      }
    }
    );
    gloco()->open_props.post_process_mode = (fan::graphics::post_process_mode_e)config.post_processing.mode;
    config.post_processing.clear_color.a = 1.f;
    gloco()->set_clear_color(config.post_processing.clear_color);
    gloco()->get_lighting().set_target(config.post_processing.ambient_color);
    gloco()->open_props.blur_amount = std::clamp(config.post_processing.blur_amount, 0.f, 1.f);
    gloco()->open_props.blur_filter_radius = config.post_processing.blur_filter_radius;
    gloco()->open_props.blur_focus_enabled = config.post_processing.blur_focus_enabled;
    gloco()->open_props.blur_focus_follow_mouse = config.post_processing.blur_focus_follow_mouse;
    gloco()->open_props.blur_focus_position = config.post_processing.blur_focus_position;
    gloco()->open_props.blur_focus_radius = config.post_processing.blur_focus_radius;
    gloco()->open_props.blur_focus_falloff = config.post_processing.blur_focus_falloff;
    *gloco()->get_bloom_threshold_ptr() = config.post_processing.bloom_threshold;
    *gloco()->get_bloom_knee_ptr() = config.post_processing.bloom_knee;
    *gloco()->get_bloom_tint_ptr() = config.post_processing.bloom_tint;
    *gloco()->get_bloom_filter_radius_ptr() = config.post_processing.bloom_filter_radius;
    sync_vulkan_post_processing(config.post_processing);
    gloco()->set_culling_enabled(config.debug.frustum_culling_enabled);
  }

  bool settings_menu_t::draw_toggle_row(const char* label, bool* enabled) {
    gui::table_next_row();
    gui::table_next_column();
    gui::align_text_to_frame_padding();
    gui::text(label);
    gui::table_next_column();
    return gui::checkbox(enabled);
  }

  void settings_menu_t::draw_sub_row(const char* sublabel, auto widget_fn, f32_t sublabel_indent, f32_t subwidget_indent) {
    gui::table_next_row();
    gui::table_next_column();
    gui::align_text_to_frame_padding();
    gui::set_cursor_pos_x(gui::get_cursor_pos_x() + sublabel_indent);
    gui::text(sublabel);
    gui::table_next_column();
    gui::set_cursor_pos_x(gui::get_cursor_pos_x() + subwidget_indent);
    widget_fn();
  }

  void settings_menu_t::begin_menu_left(const char* name, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    gui::push_font(gui::get_font(24));
    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(next_window_size);
    gui::set_next_window_bg_alpha(bg_alpha);
    gui::begin(name, nullptr, wnd_flags);
  }
  void settings_menu_t::end_menu_left() {
    gui::end();
    gui::pop_font();
  }

  void settings_menu_t::menu_graphics_left(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    begin_menu_left("##Menu Graphics Left", next_window_position, next_window_size);
    {
      gui::text(title_color, "DISPLAY");
      gui::begin_table("settings_left_table_display", 2, gui::table_flags_borders_inner_h | gui::table_flags_borders_outer_h);
      {
        gui::table_next_row();
        menu->render_display_mode();
        gui::table_next_row();
        menu->render_target_fps();
        gui::table_next_row();
        menu->render_resolution_dropdown();
      }
      gui::end_table();
    }
    gui::new_line();
    gui::new_line();

    {
      gui::text(title_color, "POST PROCESSING");
      gui::begin_table("settings_left_table_post_processing", 2,
        gui::table_flags_borders_inner_h | gui::table_flags_borders_outer_h | gui::table_flags_no_clip);

      gui::table_next_row();
      gui::table_next_column();

      draw_sub_row("Clear color", [&] {
        fan::color& c = menu->config.post_processing.clear_color;
        if (gui::color_edit3(&c)) {
          c.a = 1.f;
          gloco()->set_clear_color(c);
          menu->mark_dirty();
          menu->save();
          menu->is_dirty = false;
        }
      });

      draw_sub_row("Lighting ambient", [&] {
        fan::color& c = menu->config.post_processing.ambient_color;
        if (gui::color_edit4(&c)) {
          gloco()->get_lighting().set_target(c);
          menu->mark_dirty();
          menu->save();
          menu->is_dirty = false;
        }
      });

      static const char* post_process_modes[] = {"None", "Bloom", "Blur", "Bloom + Blur"};

      gui::table_next_column();
      gui::text("Post process");
      gui::table_next_column();
      int mode = menu->config.post_processing.mode;
      if (mode < 0 || mode >= (int)std::size(post_process_modes)) {
        mode = 0;
      }
      if (gui::begin_combo("##PostProcessMode", post_process_modes[mode])) {
        for (int i = 0; i < std::size(post_process_modes); ++i) {
          bool is_selected = mode == i;
          if (gui::selectable(post_process_modes[i], is_selected)) {
            mode = i;
            menu->config.post_processing.mode = mode;
            gloco()->open_props.post_process_mode = (fan::graphics::post_process_mode_e)mode;
            menu->mark_dirty();
          }
          if (is_selected) {
            gui::set_item_default_focus();
          }
        }
        gui::end_combo();
      }

      if (mode == (int)fan::graphics::post_process_mode_e::bloom || mode == (int)fan::graphics::post_process_mode_e::bloom_blur) {
        draw_sub_row("Strength", [&] {
          if (gui::slider(&menu->config.post_processing.bloom_strength, 0.f, 1.f, gui::slider_flags_always_clamp)) {
            sync_vulkan_post_processing(menu->config.post_processing);
            menu->mark_dirty();
          }
        });
        draw_sub_row("Threshold", [&] {
          if (gui::slider(&menu->config.post_processing.bloom_threshold, 0.0f, 5.0f, gui::slider_flags_always_clamp)) {
            *gloco()->get_bloom_threshold_ptr() = menu->config.post_processing.bloom_threshold;
            sync_vulkan_post_processing(menu->config.post_processing);
            menu->mark_dirty();
          }
        });
        draw_sub_row("Knee (Softness)", [&] {
          if (gui::slider(&menu->config.post_processing.bloom_knee, 0.0f, 1.0f, gui::slider_flags_always_clamp)) {
            *gloco()->get_bloom_knee_ptr() = menu->config.post_processing.bloom_knee;
            sync_vulkan_post_processing(menu->config.post_processing);
            menu->mark_dirty();
          }
        });
        draw_sub_row("Tint", [&] {
          if (gui::color_edit3(&menu->config.post_processing.bloom_tint)) {
            *gloco()->get_bloom_tint_ptr() = menu->config.post_processing.bloom_tint;
            sync_vulkan_post_processing(menu->config.post_processing);
            menu->mark_dirty();
          }
        });

        draw_sub_row("Filter radius", [&] {
          if (gui::slider(&menu->config.post_processing.bloom_filter_radius, 0.f, 1.f, gui::slider_flags_always_clamp)) {
            *gloco()->get_bloom_filter_radius_ptr() = menu->config.post_processing.bloom_filter_radius;
            sync_vulkan_post_processing(menu->config.post_processing);
            menu->mark_dirty();
          }
        });
      }
      if (mode == (int)fan::graphics::post_process_mode_e::blur || mode == (int)fan::graphics::post_process_mode_e::bloom_blur) {
        draw_sub_row("Amount", [&] {
          if (gui::slider(&menu->config.post_processing.blur_amount, 0.f, 1.0f, gui::slider_flags_always_clamp)) {
            menu->config.post_processing.blur_amount = std::clamp(menu->config.post_processing.blur_amount, 0.f, 1.f);
            gloco()->open_props.blur_amount = menu->config.post_processing.blur_amount;
            menu->mark_dirty();
          }
        });
        draw_sub_row("Filter radius", [&] {
          if (gui::slider(&menu->config.post_processing.blur_filter_radius, 0.f, 0.08f, gui::slider_flags_always_clamp)) {
            gloco()->open_props.blur_filter_radius = menu->config.post_processing.blur_filter_radius;
            menu->mark_dirty();
          }
        });
        draw_sub_row("Circle focus", [&] {
          if (gui::checkbox(&menu->config.post_processing.blur_focus_enabled)) {
            gloco()->open_props.blur_focus_enabled = menu->config.post_processing.blur_focus_enabled;
            menu->mark_dirty();
          }
        });
        if (menu->config.post_processing.blur_focus_enabled) {
          auto sync_focus_position = [&] {
            menu->config.post_processing.blur_focus_position.x = std::clamp(menu->config.post_processing.blur_focus_position.x, 0.f, 1.f);
            menu->config.post_processing.blur_focus_position.y = std::clamp(menu->config.post_processing.blur_focus_position.y, 0.f, 1.f);
            gloco()->open_props.blur_focus_position = menu->config.post_processing.blur_focus_position;
            menu->mark_dirty();
          };

          draw_sub_row("Follow mouse", [&] {
            if (gui::checkbox(&menu->config.post_processing.blur_focus_follow_mouse)) {
              gloco()->open_props.blur_focus_follow_mouse = menu->config.post_processing.blur_focus_follow_mouse;
              menu->mark_dirty();
            }
          });

          if (!menu->config.post_processing.blur_focus_follow_mouse) {
            draw_sub_row("Focus X", [&] {
              if (gui::slider(&menu->config.post_processing.blur_focus_position.x, 0.f, 1.f, gui::slider_flags_always_clamp)) {
                sync_focus_position();
              }
            });
            draw_sub_row("Focus Y", [&] {
              if (gui::slider(&menu->config.post_processing.blur_focus_position.y, 0.f, 1.f, gui::slider_flags_always_clamp)) {
                sync_focus_position();
              }
            });
            draw_sub_row("Focus presets", [&] {
              if (gui::button("Center")) {
                menu->config.post_processing.blur_focus_position = fan::vec2(0.5f, 0.5f);
                sync_focus_position();
              }
              gui::same_line();
              if (gui::button("Mouse")) {
                fan::vec2 mouse_position = gloco()->get_raw_mouse_position();
                fan::vec2 window_size = gloco()->window.get_size();
                menu->config.post_processing.blur_focus_position = fan::vec2(
                  window_size.x == 0.f ? 0.f : mouse_position.x / window_size.x,
                  window_size.y == 0.f ? 0.f : mouse_position.y / window_size.y
                );
                sync_focus_position();
              }
            });
          }

          draw_sub_row("Focus radius", [&] {
            if (gui::slider(&menu->config.post_processing.blur_focus_radius, 0.f, 1.f, gui::slider_flags_always_clamp)) {
              gloco()->open_props.blur_focus_radius = menu->config.post_processing.blur_focus_radius;
              menu->mark_dirty();
            }
          });
          draw_sub_row("Focus falloff", [&] {
            if (gui::slider(&menu->config.post_processing.blur_focus_falloff, 0.001f, 1.f, gui::slider_flags_always_clamp)) {
              gloco()->open_props.blur_focus_falloff = menu->config.post_processing.blur_focus_falloff;
              menu->mark_dirty();
            }
          });
        }
      }
      draw_sub_row("Gamma", [&] {
        if (gui::drag(&menu->config.post_processing.gamma, 0.01f, 0.1f, 5.0f)) {
          sync_vulkan_post_processing(menu->config.post_processing);
          menu->mark_dirty();
        }
      });
      draw_sub_row("Exposure", [&] {
        if (gui::drag(&menu->config.post_processing.exposure, 0.01f, 0.0f, 10.0f)) {
          sync_vulkan_post_processing(menu->config.post_processing);
          menu->mark_dirty();
        }
      });
      draw_sub_row("Contrast", [&] {
        if (gui::drag(&menu->config.post_processing.contrast, 0.01f, 0.0f, 5.0f)) {
          sync_vulkan_post_processing(menu->config.post_processing);
          menu->mark_dirty();
        }
      });
      gui::end_table();
    }
    
    gui::new_line();
    gui::new_line();
    {
      gui::text(title_color, "PERFORMANCE STATS");
      gui::new_line();
      gui::begin_table("settings_left_table_performance_stats", 2,
        gui::table_flags_borders_inner_h | gui::table_flags_borders_outer_h | gui::table_flags_no_clip);
      {
        bool vsync = gloco()->get_vsync();
        if (draw_toggle_row("Enable VSync", &vsync)) {
          gloco()->set_vsync(vsync);
          menu->config.performance.vsync = vsync;
          menu->mark_dirty();
        }
        if (draw_toggle_row("Show fps", (bool*)&gloco()->gui.show_fps)) {
          menu->config.performance.show_fps = gloco()->gui.show_fps;
          menu->mark_dirty();
        }
      #if defined(fan_std23)
        if (draw_toggle_row("Track Heap memory", (bool*)&fan::memory::heap_profiler_t::instance().enabled)) {
          gloco()->gui.console.commands.call("debug_memory " + std::to_string((int)fan::memory::heap_profiler_t::instance().enabled));
          menu->config.performance.track_heap = fan::memory::heap_profiler_t::instance().enabled;
          menu->mark_dirty();
        }
      #endif
        //if (draw_toggle_row("Track OpenGL calls", (bool*)&fan_track_opengl_calls())) {
        //  menu->config.performance.track_opengl_calls = fan_track_opengl_calls();
        //  menu->mark_dirty();
        //}
      }
      gui::end_table();
    }
    gui::new_line();
    gui::new_line();
    {
      gui::text(title_color, "DEBUG");
      gui::begin_table("settings_left_table_debug", 2, gui::table_flags_borders_inner_h | gui::table_flags_borders_outer_h);

    #if defined(FAN_2D)
      {
        gui::table_next_row();
        gui::table_next_column();
        gui::text("Frustum culling");
        gui::table_next_column();
        gui::text("Enable frustum culling");
        if (gui::checkbox(&((fan::graphics::culling::culling_t*)gloco()->shapes.visibility)->enabled)) {
          gloco()->set_culling_enabled(((fan::graphics::culling::culling_t*)gloco()->shapes.visibility)->enabled);
          menu->config.debug.frustum_culling_enabled = ((fan::graphics::culling::culling_t*)gloco()->shapes.visibility)->enabled;
          menu->mark_dirty();
        }
        if (draw_toggle_row("Visualize culling", &gloco()->is_visualizing_culling)) {
          menu->config.debug.visualize_culling = gloco()->is_visualizing_culling;
          menu->mark_dirty();
        }
        if (gloco()->is_visualizing_culling) {
          draw_sub_row("padding (default render view)", [&] {
            if (gui::drag(&((fan::graphics::culling::culling_t*)gloco()->shapes.visibility)->padding, 1)) {
              for (auto& [cam_id, cam_state] : ((fan::graphics::culling::culling_t*)gloco()->shapes.visibility)->camera_states) {
                cam_state.view_dirty = true;
              }
              menu->config.debug.culling_padding = ((fan::graphics::culling::culling_t*)gloco()->shapes.visibility)->padding;
              menu->mark_dirty();
            }
            if (menu->config.debug.settings_bg_alpha > 0.0f && gui::is_item_active()) {
              settings_menu_t::bg_alpha = 0.0f;
            }
          });
        }
      }
    #endif
      {
        gui::table_next_row();
        gui::table_next_column();
        gui::text("Background Alpha");
        gui::table_next_column();
        if (gui::slider(&menu->config.debug.settings_bg_alpha, 0.0f, 1.0f, gui::slider_flags_always_clamp)) {
          menu->mark_dirty();
        }
      }
    #if defined(FAN_2D)
      {
        static const char* fill_modes[] = {"Fill", "Line"};
        gui::table_next_row();
        gui::table_next_column();
        gui::text("Fill mode");
        gui::table_next_column();
        static int fill_mode = 0;
        if (gui::begin_combo("##Fill_mode", fill_modes[fill_mode])) {
          for (int i = 0; i < std::size(fill_modes); ++i) {
            bool is_selected = (fill_mode == i);
            if (gui::selectable(fill_modes[i], is_selected)) {
              fill_mode = i;
              gloco()->renderer_state.force_line_draw = (i == 1);
              menu->config.debug.fill_mode = i;
              menu->mark_dirty();
            }
            if (is_selected) gui::set_item_default_focus();
          }
          gui::end_combo();
        }
      }
    #endif
      gui::end_table();
    }
    end_menu_left();
  }

  void settings_menu_t::menu_graphics_right(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    gui::set_next_window_pos(next_window_position);
    gui::set_next_window_size(next_window_size);
    gui::set_next_window_bg_alpha(bg_alpha);
    gui::begin("##Menu Graphics Right", nullptr, wnd_flags);
    gui::push_font(gui::get_font(32, gui::font::bold));
    gui::text(title_color, "Setting Info");
    gui::pop_font();
    fan::vec2 cursor_pos = gui::get_cursor_pos();
    gui::draw_list_t* draw_list = gui::get_window_draw_list();
    fan::vec2 line_start = gui::get_cursor_screen_pos();
    line_start.x -= cursor_pos.x;
    line_start.y -= cursor_pos.y;
    fan::vec2 line_end = line_start;
    line_end.y += gui::get_content_region_max().y;
    draw_list->AddLine(line_start, line_end, fan::color(255, 255, 255, 255).get_gui_color());
    gui::end();
  }

  void settings_menu_t::menu_audio_left(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
  #if defined(FAN_AUDIO)
    begin_menu_left("##Menu Audio Left", next_window_position, next_window_size);
    {
      gui::begin_table("settings_left_table_display", 2, gui::table_flags_borders_inner_h | gui::table_flags_borders_outer_h);
      {
        gui::table_next_row();
        gui::table_next_column();
        gui::text("Volume");
        gui::table_next_column();
        f32_t volume = fan::audio::get_volume();
        if (gui::slider(&volume, 0.f, 1.f, gui::slider_flags_always_clamp)) {
          fan::audio::set_volume(volume);
          menu->config.audio.volume = volume;
          menu->mark_dirty();
        }
      }
      gui::end_table();
    }
    end_menu_left();
  #endif
  }
  void settings_menu_t::menu_audio_right(settings_menu_t* menu, const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    settings_menu_t::menu_graphics_right(menu, next_window_position, next_window_size);
  }

  void settings_menu_t::query_current_resolution() {
    if (config.display.resolution_index != -1 || config.display.custom_resolution.x != -1) {
      return;
    }
    fan::vec2i current = gloco()->window.get_size();
    for (int i = 0; i < std::size(fan::resolutions); ++i) {
      if (fan::resolutions[i].size == current) {
        config.display.resolution_index = i;
        return;
      }
    }
    config.display.custom_resolution = current;
  }

  void settings_menu_t::on_window_resize(const fan::vec2i& new_size) {
    bool matched = false;
    for (int i = 0; i < std::size(fan::resolutions); ++i) {
      if (fan::resolutions[i].size == new_size) {
        config.display.resolution_index = i;
        config.display.custom_resolution = {-1, -1};
        matched = true;
        break;
      }
    }
    if (!matched) {
      config.display.resolution_index = -1;
      config.display.custom_resolution = new_size;
    }
    mark_dirty();
  }

  void settings_menu_t::apply_config(bool construct, bool runtime) {
    if (!runtime) {
      gloco()->open_props.window_open_mode = config.display.display_mode;
      if (config.display.window_position.x != -1 && gloco()->open_props.window_position.x == -1) {
        gloco()->open_props.window_position = config.display.window_position;
      }
      gloco()->open_props.vsync = config.performance.vsync;
      if (config.display.custom_resolution.x != -1 && gloco()->open_props.window_size.x == -1) {
        gloco()->open_props.window_size = config.display.custom_resolution;
      }
      else if (config.display.resolution_index != -1 && gloco()->open_props.window_size.x == -1) {
        gloco()->open_props.window_size = fan::resolutions[config.display.resolution_index].size;
      }
      return;
    }
    gloco()->set_target_fps(config.display.target_fps);
    gloco()->set_vsync(config.performance.vsync);
    gloco()->gui.show_fps = config.performance.show_fps;
  #if defined(FAN_AUDIO)
    fan::audio::set_volume(config.audio.volume);
  #endif
  }

  void settings_menu_t::mark_dirty() {
    is_dirty = true;
    save_timer.start_millis(save_delay_ms);
  }
  void settings_menu_t::update() {
    if (is_dirty && save_timer.finished()) {
      save();
      is_dirty = false;
    }
  }
  void settings_menu_t::change_target_fps(int dir) {
    int idx = 0;
    for (int i = 0; i < std::size(fps_values); ++i) {
      if (fps_values[i] == gloco()->target_fps) {
        idx = i;
        break;
      }
    }
    idx = (idx + dir + std::size(fps_values)) % std::size(fps_values);
    gloco()->set_target_fps(fps_values[idx]);
    config.display.target_fps = fps_values[idx];
    mark_dirty();
  }

  void settings_menu_t::render_display_mode() {
    static constexpr const char* display_mode_names[] = {"Windowed", "Borderless", "Fullscreen"};
    gui::table_next_column();
    gui::align_text_to_frame_padding();
    gui::text("Display Mode");
    gui::table_next_column();
    if (gui::begin_combo("##Display Mode", display_mode_names[gloco()->open_props.window_open_mode - 1])) {
      for (int i = 0; i < std::size(display_mode_names); ++i) {
        bool is_selected = (gloco()->open_props.window_open_mode - 1 == i);
        if (gui::selectable(display_mode_names[i], is_selected)) {
          gloco()->open_props.window_open_mode = i + 1;
          config.display.display_mode = i + 1;
          mark_dirty();
          gloco()->window.set_display_mode(config.display.display_mode);
        }
        if (is_selected) gui::set_item_default_focus();
      }
      gui::end_combo();
    }
  }

  void settings_menu_t::render_target_fps() {
    gui::table_next_column();
    gui::align_text_to_frame_padding();
    gui::text("Target Framerate");
    gui::table_next_column();
    gui::same_line();
    if (gui::arrow_button("##left_arrow", gui::dir_left)) change_target_fps(-1);
    gui::same_line();
    gui::text(gloco()->target_fps);
    gui::same_line();
    if (gui::arrow_button("##right_arrow", gui::dir_right)) change_target_fps(1);
  }

  void settings_menu_t::render_resolution_dropdown() {
    fan::vec2i current_size = gloco()->window.get_size();
    int current_resolution = -1;
    for (int i = 0; i < std::size(fan::resolutions); ++i) {
      if (fan::resolutions[i].size == fan::vec2i(current_size)) {
        current_resolution = i;
        break;
      }
    }
    if (current_resolution == -1) current_resolution = std::size(fan::resolutions);

    gui::table_next_column();
    gui::align_text_to_frame_padding();
    gui::text("Resolution");
    gui::table_next_column();
    fan::vec2i window_size = gloco()->window.get_size();
    std::string custom_res = std::to_string(window_size.x) + "x" + std::to_string(window_size.y);
    const char* current_label = (current_resolution == std::size(fan::resolutions))
      ? custom_res.c_str()
      : fan::resolutions[current_resolution];

    if (gui::begin_combo("##ResolutionCombo", current_label)) {
      for (int i = 0; i < std::size(fan::resolutions); ++i) {
        bool is_selected = (current_resolution == i);
        if (gui::selectable(fan::resolutions[i].label, is_selected)) {
          current_resolution = i;
          gloco()->window.set_size(fan::resolutions[i]);
          config.display.resolution_index = i;
          config.display.custom_resolution = fan::vec2i(-1, -1);
          mark_dirty();
        }
        if (is_selected) gui::set_item_default_focus();
      }
      if (current_resolution == std::size(fan::resolutions) && gui::selectable(custom_res.c_str(), true)) {
        config.display.resolution_index = -1;
        config.display.custom_resolution = window_size;
      }
      gui::end_combo();
    }
  }

  void settings_menu_t::render_separator_full_width(f32_t y_offset) {
    auto* draw_list = gui::get_window_draw_list();
    fan::vec2 win_pos = gui::get_window_pos();
    fan::vec2 win_size = gui::get_window_size();
    f32_t y = win_pos.y + gui::get_cursor_pos().y + y_offset;
    draw_list->AddLine(
      fan::vec2(win_pos.x, y),
      fan::vec2(win_pos.x + win_size.x, y),
      gui::get_color_u32(gui::col_separator),
      1.0f
    );
  }

  void settings_menu_t::render_settings_left(const fan::vec2& next_window_position, const fan::vec2& next_window_size) {
    pages[current_page].render_page_left(this, next_window_position, next_window_size);
  }
  void settings_menu_t::render_settings_right(const fan::vec2& next_window_position, const fan::vec2& next_window_size, f32_t min_x) {
    pages[current_page].render_page_right(this, next_window_position, next_window_size);
  }

  fan::vec2 settings_menu_t::render_settings_top(f32_t min_x) {
    fan::vec2 main_window_size = gloco()->window.get_size();
    gui::set_next_window_pos(fan::vec2(0, 0));
    gui::set_next_window_size(fan::vec2(main_window_size.x, main_window_size.y / 5));
    gui::set_next_window_bg_alpha(bg_alpha);
    gui::begin("##Fan Settings Nav", nullptr,
      gui::window_flags_no_move | gui::window_flags_no_collapse |
      gui::window_flags_no_resize | gui::window_flags_no_title_bar |
      gui::window_flags_always_vertical_scrollbar
    );
    gui::push_font(gui::get_font(48, gui::font::bold));
    gui::indent(min_x);
    gui::text("Settings");
    gui::pop_font();
    gui::unindent(min_x);
    render_separator_full_width();
    f32_t options_x = 256.f;
    gui::indent(options_x);
    gui::push_font(gui::get_font(32, gui::font::bold));
    gui::push_style_var(gui::style_var_frame_padding, fan::vec2(64, 5.f));
    gui::begin_table("##settings_top_table", pages.size());
    gui::table_next_row();
    for (std::size_t i = 0; i < std::size(pages); ++i) {
      gui::table_next_column();
      bool& is_toggled = pages[i].toggle;
      gui::push_style_color(gui::col_button, is_toggled
        ? gui::get_color(gui::col_button_hovered)
        : fan::colors::transparent);
      if (gui::button(pages[i].name.c_str())) {
        pages[i].toggle = !pages[i].toggle;
        if (pages[i].toggle) {
          reset_page_selection();
          pages[i].toggle = 1;
          current_page = i;
        }
      }
      gui::pop_style_color();
    }
    gui::end_table();
    gui::pop_style_var();
    gui::pop_font();
    gui::unindent(options_x);
    render_separator_full_width();
    fan::vec2 window_size = gui::get_window_size();
    gui::unindent();
    gui::end();
    return window_size;
  }

  void settings_menu_t::render() {

    bg_alpha = config.debug.settings_bg_alpha;

    render_settings_top(min_x);
    fan::vec2 size = gloco()->window.get_size();
    f32_t ratio = pages[current_page].split_ratio;
    const f32_t top_h = size.y * 0.2f;
    render_settings_left({0.f, top_h}, {size.x * ratio, size.y - top_h});
    render_settings_right({size.x * ratio, top_h}, {size.x * (1.f - ratio), size.y - top_h}, min_x);
    keybind_menu.update();
  }

  void settings_menu_t::reset_page_selection() {
    for (auto& p : pages) p.toggle = false;
  }

  void settings_menu_t::set_settings_theme() {
    auto& style = gui::get_style();
    style.Alpha = 1.0f;
    style.DisabledAlpha = 0.5f;
    style.WindowPadding = fan::vec2(13.0f, 10.0f);
    style.WindowRounding = 0.0f;
    style.WindowBorderSize = 1.0f;
    style.WindowMinSize = fan::vec2(32.0f, 32.0f);
    style.WindowTitleAlign = fan::vec2(0.5f, 0.5f);
    style.WindowMenuButtonPosition = gui::dir_right;
    style.ChildRounding = 3.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupRounding = 5.0f;
    style.PopupBorderSize = 1.0f;
    style.FramePadding = fan::vec2(20.0f, 8.100000381469727f);
    style.FrameRounding = 2.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = fan::vec2(3.0f, 3.0f);
    style.ItemInnerSpacing = fan::vec2(3.0f, 8.0f);
    style.CellPadding = fan::vec2(6.0f, 14.10000038146973f);
    style.IndentSpacing = 0.0f;
    style.ColumnsMinSpacing = 10.0f;
    style.ScrollbarSize = 10.0f;
    style.ScrollbarRounding = 2.0f;
    style.GrabMinSize = 12.10000038146973f;
    style.GrabRounding = 1.0f;
    style.TabRounding = 2.0f;
    style.TabBorderSize = 0.0f;
    style.TabMinWidthForCloseButton = 5.0f;
    style.ColorButtonPosition = gui::dir_right;
    style.ButtonTextAlign = fan::vec2(0.5f, 0.5f);
    style.SelectableTextAlign = fan::vec2(0.0f, 0.0f);
    style.Colors[gui::col_text]                   = fan::color(0.9803921580314636f,  0.9803921580314636f,  0.9803921580314636f,  1.0f);
    style.Colors[gui::col_text_disabled]          = fan::color(0.4980392158031464f,  0.4980392158031464f,  0.4980392158031464f,  1.0f);
    style.Colors[gui::col_window_bg]              = fan::color(0.01f,                0.01f,                0.01f,                0.99f);
    style.Colors[gui::col_child_bg]               = fan::color(0.01f,                0.01f,                0.01f,                1.0f);
    style.Colors[gui::col_popup_bg]               = fan::color(0.09411764889955521f, 0.09411764889955521f, 0.09411764889955521f, 1.0f);
    style.Colors[gui::col_border]                 = fan::color(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
    style.Colors[gui::col_border_shadow]          = fan::color(0.0f, 0.0f, 0.0f, 0.0f);
    style.Colors[gui::col_frame_bg]               = fan::color(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
    style.Colors[gui::col_frame_bg_hovered]       = fan::color(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
    style.Colors[gui::col_frame_bg_active]        = fan::color(0.0f, 0.0f, 0.0f, 0.0470588244497776f);
    style.Colors[gui::col_title_bg]               = fan::color(0.1176470592617989f,  0.1176470592617989f,  0.1176470592617989f,  1.0f);
    style.Colors[gui::col_title_bg_active]        = fan::color(0.1568627506494522f,  0.1568627506494522f,  0.1568627506494522f,  1.0f);
    style.Colors[gui::col_title_bg_collapsed]     = fan::color(0.1176470592617989f,  0.1176470592617989f,  0.1176470592617989f,  1.0f);
    style.Colors[gui::col_menu_bar_bg]            = fan::color(0.09411764889955521f, 0.09411764889955521f, 0.09411764889955521f, 1.0f);
    style.Colors[gui::col_scrollbar_bg]           = fan::color(0.0f, 0.0f, 0.0f, 0.1098039224743843f);
    style.Colors[gui::col_scrollbar_grab]         = fan::color(1.0f, 1.0f, 1.0f, 0.3921568691730499f);
    style.Colors[gui::col_scrollbar_grab_hovered] = fan::color(1.0f, 1.0f, 1.0f, 0.4705882370471954f);
    style.Colors[gui::col_scrollbar_grab_active]  = fan::color(0.0f, 0.0f, 0.0f, 0.09803921729326248f);
    style.Colors[gui::col_check_mark]             = fan::color(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[gui::col_slider_grab]            = fan::color(1.0f, 1.0f, 1.0f, 0.3921568691730499f);
    style.Colors[gui::col_slider_grab_active]     = fan::color(1.0f, 1.0f, 1.0f, 0.3137255012989044f);
    style.Colors[gui::col_button]                 = fan::color(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
    style.Colors[gui::col_button_hovered]         = fan::color(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
    style.Colors[gui::col_button_active]          = fan::color(0.0f, 0.0f, 0.0f, 0.0470588244497776f);
    style.Colors[gui::col_header]                 = fan::color(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
    style.Colors[gui::col_header_hovered]         = fan::color(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
    style.Colors[gui::col_header_active]          = fan::color(0.0f, 0.0f, 0.0f, 0.0470588244497776f);
    style.Colors[gui::col_separator]              = fan::color(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
    style.Colors[gui::col_separator_hovered]      = fan::color(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
    style.Colors[gui::col_separator_active]       = fan::color(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
    style.Colors[gui::col_resize_grip]            = fan::color(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
    style.Colors[gui::col_resize_grip_hovered]    = fan::color(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
    style.Colors[gui::col_resize_grip_active]     = fan::color(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
    style.Colors[gui::col_tab]                    = fan::color(1.0f, 1.0f, 1.0f, 0.09803921729326248f);
    style.Colors[gui::col_tab_hovered]            = fan::color(1.0f, 1.0f, 1.0f, 0.1568627506494522f);
    style.Colors[gui::col_tab_selected]           = fan::color(1.0f, 1.0f, 1.0f, 0.3137255012989044f);
    style.Colors[gui::col_tab_dimmed]             = fan::color(0.0f, 0.0f, 0.0f, 0.1568627506494522f);
    style.Colors[gui::col_tab_dimmed_selected]    = fan::color(1.0f, 1.0f, 1.0f, 0.2352941185235977f);
    style.Colors[gui::col_plot_lines]             = fan::color(1.0f, 1.0f, 1.0f, 0.3529411852359772f);
    style.Colors[gui::col_plot_lines_hovered]     = fan::color(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[gui::col_plot_histogram]         = fan::color(1.0f, 1.0f, 1.0f, 0.3529411852359772f);
    style.Colors[gui::col_plot_histogram_hovered] = fan::color(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[gui::col_table_header_bg]        = fan::color(0.1568627506494522f,  0.1568627506494522f,  0.1568627506494522f,  1.0f);
    style.Colors[gui::col_table_border_strong]    = fan::color(1.0f, 1.0f, 1.0f, 0.3137255012989044f);
    style.Colors[gui::col_table_border_light]     = fan::color(1.0f, 1.0f, 1.0f, 0.196078434586525f);
    style.Colors[gui::col_table_row_bg]           = fan::color(0.0f, 0.0f, 0.0f, 0.0f);
    style.Colors[gui::col_table_row_bg_alt]       = fan::color(1.0f, 1.0f, 1.0f, 0.01960784383118153f);
    style.Colors[gui::col_text_selected_bg]       = fan::color(0.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[gui::col_drag_drop_target]       = fan::color(0.168627455830574f,   0.2313725501298904f,  0.5372549295425415f,  1.0f);
    style.Colors[gui::col_nav_cursor]             = fan::color(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[gui::col_nav_windowing_highlight]= fan::color(1.0f, 1.0f, 1.0f, 0.699999988079071f);
    style.Colors[gui::col_nav_windowing_dim_bg]   = fan::color(0.800000011920929f,   0.800000011920929f,   0.800000011920929f,   0.2000000029802322f);
  }

  bool settings_menu_t::load() {
    std::string content;
    if (fan::io::file::read(config.config_save_path, &content) != 0) return false;
    fan::json j;
    try {
      j = fan::json::parse(content);
    }
    catch (...) {
      return false;
    }
    config.load_from_json(j);
    if (j.contains("keybinds")) {
      keybind_menu.load_from_settings_json(j);
      keybind_menu.apply_to_input_action();
    }
    return true;
  }
  void settings_menu_t::save() {
    fan::json j = config.to_json();
    keybind_menu.save_to_settings_json(j);
    fan::io::file::write(config.config_save_path, j.dump(2), std::ios_base::binary);
  }
} // namespace fan::graphics::gui

#endif

#endif
