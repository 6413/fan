import std;
import fan;

using namespace fan::graphics;

struct resources_t {
  f64_t gold_per_tick = 5.0;
  f64_t gold = 0.0;
}resources;

struct building_t : sprite_t {
  using sprite_t::sprite_t;
  std::string name;
  fan::vec2i grid_cell;
  int cells_x = 1;
  int cells_y = 1;
};

struct building_image_t : image_t {
  using image_t::image_t;
  std::string name;
  fan::vec2 custom_scale = 1.0f;
};

struct transform_t { 
  fan::vec3 position; 
  fan::vec2 size; 
};

struct day_cycle_t {
  f32_t time_of_day = 0.25f;
  f32_t day_length = 5.f;
  f32_t speed = 1.f;
  int day_count = 0;
  bool paused = false;

  void update(f32_t dt, engine_t& engine) {
    if (paused) return;
    time_of_day += dt * speed / day_length;
    if (time_of_day >= 1.f) {
      time_of_day -= 1.f;
      day_count++;
      resources.gold += resources.gold_per_tick;
    }

    engine.get_lighting().set_target(day_cycle_ambient(time_of_day), 0.f);
  }

  bool is_night() const {
    return day_cycle_is_night(time_of_day);
  }
};

struct farm_manager_t {
  fan::json buildings_config;
  std::vector<building_image_t> buildings;
  std::list<building_t> placed_buildings;
  std::unordered_map<fan::vec2i, building_t*> building_map;
  fan::tween::tween_manager_t tweens;
  fan::vec2 tile_size;
  grid_placer_t grid;
  
  building_t placement;
  building_image_t* selected_building = nullptr;
  sprite_t grass;
  day_cycle_t day_cycle;
  grid_drag_painter_t drag_painter;

  farm_manager_t(engine_t& engine, const fan::vec2& t_size) : 
    buildings_config(fan::json::load_file("buildings.json")),
    tile_size(t_size), 
    grid(t_size),
    grass(/*engine.create_transparent_texture()*//**/"images/IMGP5511_seamless.jpg"),
    placement(0, 0, image_t::invalid()) 
  {
    load_buildings();
    spawn_initial_buildings();
    if (!buildings.empty()) {
      select_building(0);
    }
  }

  void select_building(int index) {
    selected_building = &buildings[index];
    placement.name = selected_building->name;
    placement.set_image(*selected_building);
    placement.set_size(get_transform(0, *selected_building).size);
  }

  transform_t get_transform(fan::vec2i cell, const building_image_t& img) {
    fan::vec2 size = grid.get_fit_size(img.get_size(), img.custom_scale);
    return {grid.get_placement(cell, size, img.custom_scale.x), size / 2.f};
  }

  void load_buildings() {
    std::unordered_set<std::string> loaded;
    auto load_one = [&](const std::string& dir) {
      if (!std::filesystem::exists(dir)) return;
      fan::io::iterate_directory(dir, [&](const std::string& file, bool) {
        std::string name = std::filesystem::path(file).stem().string();
        if (!loaded.insert(name).second) return;
        auto& b = buildings.emplace_back(("images/factory/" + name + ".png").c_str());
        b.name = name;
        if (buildings_config.contains(name)) {
          f32_t scale = buildings_config[name].value("scale", 1.0f);
          b.custom_scale = {scale, scale};
        }
      });
    };
    load_one("tests/factory");
    load_one("images/factory");
  }

  building_t* get_building_at(fan::vec2i cell) {
    auto it = building_map.find(cell);
    return it != building_map.end() ? it->second : nullptr;
  }

  void map_building(building_t& b) {
    for (int y = 0; y < b.cells_y; ++y)
      for (int x = 0; x < b.cells_x; ++x)
        building_map[b.grid_cell + fan::vec2i(x * tile_size.x, -y * tile_size.y)] = &b;
  }

  bool can_place(fan::vec2i cell, const building_image_t& img) {
    auto gs = grid.cells_occupied(img.custom_scale);
    if (gs.x == 1 && gs.y == 1) return get_building_at(cell) == nullptr;
    for (int y = 0; y < gs.y; ++y)
      for (int x = 0; x < gs.x; ++x)
        if (get_building_at(cell + fan::vec2i(x * tile_size.x, -y * tile_size.y))) return false;
    return true;
  }

  building_t& add_building(fan::vec2i cell, building_image_t& img) {
    auto t = get_transform(cell, img);
    auto& b = placed_buildings.emplace_back(t.position, t.size, img);
    b.name = img.name;
    b.grid_cell = cell;
    b.cells_x = grid.cells_occupied(img.custom_scale).x;
    b.cells_y = grid.cells_occupied(img.custom_scale).y;
    load_model(b);
    map_building(b);
    return b;
  }

  void place_building_with_tween(fan::vec2i cell, building_image_t* img) {
    if (!img || !can_place(cell, *img)) return;
    fan::audio::play("audio/pop1.sac");
    auto target_size = get_transform(cell, *img).size;
    add_building(cell, *img).set_size(0.f);
    tweens.add<fan::vec2>([this, cell](fan::vec2 v) {
      if (auto b = get_building_at(cell); b && b->grid_cell == cell) b->set_size(v);
    }, 0.f, target_size, 0.75f);
  }

  void place_building_instant(fan::vec2i cell, building_image_t* img) {
    if (!img || !can_place(cell, *img)) return;
    add_building(cell, *img);
  }

  void spawn_initial_buildings() {
    int x = 0;
    int y = 0;
    for (auto& img : buildings) {
      fan::vec2i cell(x * tile_size.x, 0);
      auto& b = add_building(cell, img);
      x += b.cells_x;
      if (y > 10) break;
      ++y;
    }
  }

  void render_ui() {
    static gui::grid_state_t state;
    static int selected = 0;
    fan::vec2 bounds = gui::calc_grid_bounds(buildings.size(), buildings.size(), 64.f, 8.f, state.zoom);

    {
      gui::style_scope_t bg{gui::col_window_bg, fan::color{0.08f, 0.08f, 0.1f, 0.75f}};
      gui::window_anchor_top_left(0.f);
      f32_t panel_height = 54.f;
      if (auto bar = gui::overlay_window("##topbar", {gloco()->window.get_size().x, panel_height}, 0.85f)) {
        gui::font_scope_t main_font{32.f, gui::font::bold};
        f32_t cy = (panel_height / 2.f) - gui::get_text_line_height() / 2.f + 0.5;

        gui::set_cursor_pos({12.f, cy});
        gui::image("game/resources/gold/gold_coin_01.png", gui::get_text_line_height());
        gui::same_line();
        gui::text(std::to_string(std::uint64_t(resources.gold)), {.color = fan::color{1.f, 0.84f, 0.f}, .outlined = true});
        gui::same_line();
        gui::font_scope_t small_font{24.f, gui::font::bold};
        {
          f32_t spc = gui::get_style().ItemSpacing.x;
          gui::text(
            "+" + std::to_string(std::uint64_t(resources.gold_per_tick)),
            {
              .color = fan::color{1.f, 0.84f, 0.f},
              .offset = {-spc * 0.9f, -spc * 1.4f},
              .outlined = true,
            }
          );
        }
        gui::same_line();
        gui::text(fan::color{0.9f, 0.9f, 0.95f}, "  |  Buildings: ", (int)placed_buildings.size());

        gui::same_line();
        {
          gui::font_scope_t day_font{26.f, gui::font::bold};
          gui::text(fan::color{0.9f, 0.9f, 0.95f}, "  |  Day ", day_cycle.day_count);
          gui::same_line();
          gui::text(day_cycle.paused ? fan::color{0.4f, 0.4f, 0.4f} : day_cycle.is_night() ? fan::color{0.5f, 0.5f, 0.8f} : fan::color{1.f, 0.95f, 0.6f}, day_cycle.is_night() ? "☾" : "☀");
        }
      }
    }
    
    if (auto wnd = gui::floating_toolbar{"##buildings", bounds}) {
      if (gui::single_image_selector("buildings_grid", state, buildings, selected, buildings.size(), 64.f, 8.f)) {
        select_building(selected);
      }
    }
  }

  void load_model(building_t& b) {
    std::string model_path = "models/" + b.name + ".json";
    if (std::filesystem::exists(fan::io::file::find_relative_path(model_path))) {
      auto p = b.get_position();
      static_cast<fan::graphics::shape_t&>(b) = fan::graphics::shapes_children_from_json(model_path);
      b.set_position(p);
      b.start_particles();
    }
  }

  void handle_input(engine_t& engine) {
    if (gui::want_io()) {
      return;
    }
    if (engine.is_key_clicked(fan::key_space)) {
      day_cycle.paused = !day_cycle.paused;
    }
    fan::vec2i cell = grid.get_cell(engine.get_mouse_position());
    if (engine.is_mouse_down(0)) {
      for (auto& c : drag_painter.update(engine.get_mouse_position(), tile_size)) {
        if (!get_building_at(c)) {
          place_building_instant(c, selected_building);
        }
      }
    }
    else {
      drag_painter.reset();
    }
    if (engine.is_mouse_clicked(1)) {
      if (auto b = get_building_at(cell)) {
        b->remove_all_children();
        placed_buildings.remove_if([&](const building_t& item) { return &item == b; });
      }
    }
  }

  void update_placement_cursor(engine_t& engine) {
    if (!selected_building) return;
    fan::vec2i cell = grid.get_cell(engine.get_mouse_position());
    if (can_place(cell, *selected_building)) {
      placement.set_position(get_transform(cell, *selected_building).position);
    } else {
      placement.set_position(fan::vec3(-100000.f, -100000.f, 0.f));
    }
  }

  void update(engine_t& engine, f32_t dt) {
    day_cycle.update(dt, engine);
    tweens.update(dt);
    update_placement_cursor(engine);
    update_tiling_background(grass, tile_size);
    handle_input(engine);
    render_ui();
  }
};

int main() {
  engine_t engine;
  gloco()->set_settings({
    .mode = fan::graphics::post_process_mode_e::bloom,
    .bloom_strength = 0.173f,
    .bloom_threshold = 0.4f,
    .bloom_knee = 0.057f,
    .bloom_smooth_rate = 6.431f,
    .bloom_luma_scale = 0.776f,
    .bloom_adaptation_blend = 0.203f,
    .bloom_filter_radius = 1.0f,
    .gamma = 2.2f,
    .exposure = 1.0f,
    .contrast = 1.0f,
  });
  fan::audio::play("audio/ambient1.sac", 0, true);
  interactive_camera_t ic;
  farm_manager_t farm(engine, fan::vec2{256, 256});
  engine.loop([&] (f32_t dt) {
    farm.update(engine, dt);
  });
}
