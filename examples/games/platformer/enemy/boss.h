#pragma once

template<typename derived_t>
struct boss_t : enemy_t<derived_t> {
  using base_t = enemy_t<derived_t>;
  template<typename container_t>
  void open(container_t* bll, typename container_t::nr_t nr, const std::string& path, const std::source_location& caller_path = std::source_location::current()) {
    base_t::open(bll, nr, path, caller_path);
    base_t::hearts.clear();
  }

  struct health_bar_shapes_t {
    void init(const fan::vec2& bar_center, const fan::vec2& bar_half_world, const fan::graphics::render_view_t* rv) {
      if (initialized) return;
      background = fan::graphics::rectangle_t{{
        .render_view = rv,
        .position = fan::vec3(bar_center, 0xFFF0),
        .size = bar_half_world,
        .color = fan::colors::black.set_alpha(0.95f),
        .enable_culling = false
      }};
      delayed_bar = fan::graphics::rectangle_t{{
        .render_view = rv,
        .position = fan::vec3(bar_center, 0xFFF0 + 1),
        .size = bar_half_world,
        .color = fan::colors::white.set_alpha(0.75f),
        .enable_culling = false
      }};
      current_bar = fan::graphics::rectangle_t{{
        .render_view = rv,
        .position = fan::vec3(bar_center, 0xFFF0 + 2),
        .size = bar_half_world,
        .color = fan::colors::red.set_alpha(0.95f),
        .enable_culling = false
      }};
      initialized = true;
    }
    fan::graphics::rectangle_t background;
    fan::graphics::rectangle_t delayed_bar;
    fan::graphics::rectangle_t current_bar;
    bool initialized = false;
  };

  bool update() override {
    f32_t current_hp = base_t::body.get_health();

    if (displayed_hp == 0.f && delayed_hp == 0.f) {
      displayed_hp = current_hp;
      delayed_hp = current_hp;
    }

    if (displayed_hp != current_hp) {
      displayed_hp = current_hp;
    }

    if (delayed_hp > current_hp) {
      delayed_hp -= anim_remove_hp_s * pile->engine.delta_time;
      if (delayed_hp < current_hp) {
        delayed_hp = current_hp;
      }
    }
    else if (delayed_hp < current_hp) {
      delayed_hp = current_hp;
    }

    return base_t::base_update();
  }

  void render_health() override {
    if (!render_health_bar) {
      return;
    }

    using namespace fan::graphics;
    gui::set_next_window_pos(0);
    gui::set_next_window_size(gui::get_window_size());
    if (auto hud = gui::hud("##boss_hud")) {
      fan::vec2 viewport_size = gui::get_window_size();
      fan::vec2 bar_center(viewport_size.x * 0.5f, viewport_size.y * 0.92f);
      fan::vec2 bar_half_screen = viewport_size * fan::vec2(0.8f, 0.02f) * 0.5f;
      f32_t camera_zoom = ctx()->camera_get_zoom(ctx(), get_orthographic_render_view().camera);
      fan::vec2 bar_half_world = bar_half_screen / camera_zoom;

      fan::vec2 previous_cursor = gui::get_cursor_pos();
      fan::vec2 text_size = gui::get_text_size(name);
      fan::vec2 bar_top_left = bar_center - bar_half_screen;
      gui::set_cursor_pos(bar_top_left + fan::vec2(4.f, -text_size.y - 4.f));
      gui::text(name);

      fan::vec2 bar_world_center = screen_to_world(bar_center);
      f32_t max_hp = base_t::body.get_max_health();
      f32_t current_hp = displayed_hp;
      f32_t delayed_hp_value = delayed_hp;
      f32_t ratio = current_hp / max_hp;
      f32_t delayed_ratio = delayed_hp_value / max_hp;

      fan::vec2 current_half(bar_half_world.x * ratio, bar_half_world.y);
      fan::vec2 delayed_half(bar_half_world.x * delayed_ratio, bar_half_world.y);
      fan::vec2 delayed_center = bar_world_center.xy() - fan::vec2(bar_half_world.x - delayed_half.x, 0);
      fan::vec2 current_center = bar_world_center.xy() - fan::vec2(bar_half_world.x - current_half.x, 0);

      health_bar_shapes.init(bar_world_center, bar_half_world, &get_orthographic_render_view());

      health_bar_shapes.background.set_position(bar_world_center);
      health_bar_shapes.background.set_size(bar_half_world);

      health_bar_shapes.delayed_bar.set_position(delayed_center);
      health_bar_shapes.delayed_bar.set_size(delayed_half);

      health_bar_shapes.current_bar.set_position(current_center);
      health_bar_shapes.current_bar.set_size(current_half);

      gui::set_cursor_pos(previous_cursor);
    }
  }
  
  health_bar_shapes_t health_bar_shapes;
  std::string name;
  f32_t displayed_hp = base_t::body.get_health();
  f32_t delayed_hp = base_t::body.get_health();
  f32_t anim_remove_hp_s = 50.f;
  bool render_health_bar = false;
};