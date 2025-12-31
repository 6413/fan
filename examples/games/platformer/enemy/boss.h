#pragma once

template<typename derived_t>
struct boss_t : enemy_t<derived_t> {
  using base_t = enemy_t<derived_t>;

  std::string name;
  f32_t displayed_hp = base_t::body.get_health();
  f32_t delayed_hp = base_t::body.get_health();
  f32_t anim_remove_hp_s = 50.f;

  bool update() override {
    f32_t current_hp = base_t::body.get_health();

    if (displayed_hp == 0.f && delayed_hp == 0.f) {
      displayed_hp = current_hp;
      delayed_hp = current_hp;
    }

    displayed_hp = current_hp;

    if (delayed_hp > current_hp) {
      delayed_hp -= anim_remove_hp_s * pile->engine.delta_time;
      if (delayed_hp < current_hp) delayed_hp = current_hp;
    }
    else if (delayed_hp < current_hp) {
      delayed_hp = current_hp;
    }

    return base_t::base_update();
  }

  void render_health() override {
    using namespace fan::graphics;

    gui::set_next_window_pos(0);
    gui::set_next_window_size(gui::get_window_size());

    if (gui::hud("##boss_hud")) {
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

      auto bar_world_center = fan::vec3(screen_to_world(bar_center), 0xFFF0);

      f32_t max_hp = base_t::body.get_max_health();
      f32_t current_hp = displayed_hp;
      f32_t delayed_hp_value = delayed_hp;

      f32_t ratio = current_hp / max_hp;
      f32_t delayed_ratio = delayed_hp_value / max_hp;

      fan::vec2 current_half(bar_half_world.x * ratio, bar_half_world.y);
      fan::vec2 delayed_half(bar_half_world.x * delayed_ratio, bar_half_world.y);

      rectangle({.position=bar_world_center, .size = bar_half_world, .color = fan::colors::black.set_alpha(0.95f), .enable_culling = false});

      fan::vec2 delayed_center = bar_world_center.xy() - fan::vec2(bar_half_world.x - delayed_half.x, 0);
      fan::vec2 current_center = bar_world_center.xy() - fan::vec2(bar_half_world.x - current_half.x, 0);

      rectangle({.position=fan::vec3(delayed_center, bar_world_center.z + 1), .size= delayed_half, .color = fan::colors::white.set_alpha(0.75f), .enable_culling = false});
      rectangle({.position=fan::vec3(current_center, bar_world_center.z + 2), .size= current_half, .color = fan::colors::red.set_alpha(0.95f), .enable_culling = false});

      gui::set_cursor_pos(previous_cursor);
    }
  }
};