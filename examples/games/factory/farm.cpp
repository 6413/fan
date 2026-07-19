import std;
import fan;

using namespace fan::graphics;

struct building_t : sprite_t {
  using sprite_t::sprite_t;
  std::string name;
  fan::vec2i grid_cell;
  int cells_x = 1;
  int cells_y = 1;
  shape_t model;
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

struct farm_manager_t {
  fan::json buildings_config;
  std::vector<building_image_t> buildings;
  std::list<building_t> placed_buildings;
  fan::tween::tween_manager_t tweens;
  fan::vec2 tile_size;
  grid_placer_t grid;
  
  building_t placement;
  building_image_t* selected_building = nullptr;
  sprite_t grass;

  farm_manager_t(engine_t& engine, const fan::vec2& t_size) : 
    buildings_config(fan::json::load_file("buildings.json")),
    tile_size(t_size), 
    grid(t_size),
    grass(engine.create_transparent_texture()),
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
    for (auto& b : placed_buildings) {
      if (cell.x >= b.grid_cell.x && cell.x < b.grid_cell.x + b.cells_x * tile_size.x &&
          cell.y <= b.grid_cell.y && cell.y > b.grid_cell.y - b.cells_y * tile_size.y) {
        return &b;
      }
    }
    return nullptr;
  }

  bool can_place(fan::vec2i cell, const building_image_t& img) {
    auto gs = grid.cells_occupied(img.custom_scale);
    for (int y = 0; y < gs.y; ++y) {
      for (int x = 0; x < gs.x; ++x) {
        if (get_building_at(cell + fan::vec2i(x * tile_size.x, -y * tile_size.y))) {
          return false;
        }
      }
    }
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
    return b;
  }

  void spawn_initial_buildings() {
    int x = 0;
    for (auto& img : buildings) {
      fan::vec2i cell(x * tile_size.x, 0);
      auto& b = add_building(cell, img);
      x += b.cells_x;
    }
  }

  void place_building_with_tween(fan::vec2i cell, building_image_t* img) {
    if (!img || !can_place(cell, *img)) {
      return;
    }
    auto target_size = get_transform(cell, *img).size;
    add_building(cell, *img).set_size(0.f);
    tweens.add<fan::vec2>([this, cell](fan::vec2 v) {
      if (auto b = get_building_at(cell); b && b->grid_cell == cell) b->set_size(v);
    }, 0.f, target_size, 0.75f);
  }

  void render_ui() {
    static gui::grid_state_t state;
    static int selected = 0;
    fan::vec2 bounds = gui::calc_grid_bounds(buildings.size(), buildings.size(), 64.f, 8.f, state.zoom);
    if (auto wnd = gui::floating_toolbar{"##buildings", bounds}) {
      if (gui::single_image_selector("buildings_grid", state, buildings, selected, buildings.size(), 64.f, 8.f)) {
        select_building(selected);
      }
    }
  }

  void load_model(building_t& b) {
    std::string model_path = "models/" + b.name + ".json";
    if (std::filesystem::exists(fan::io::file::find_relative_path(model_path))) {
      b.model = fan::graphics::shapes_children_from_json(model_path);
      b.model.set_position(b.get_position());
      b.model.start_particles();
    }
  }

  void handle_input(engine_t& engine) {
    if (gui::want_io()) {
      return;
    }
    fan::vec2i cell = grid.get_cell(engine.get_mouse_position());
    if (engine.is_mouse_down(0)) {
      if (!get_building_at(cell)) {
        place_building_with_tween(cell, selected_building);
      }
    }
    if (engine.is_mouse_clicked(1)) {
      if (auto b = get_building_at(cell)) {
        b->model.erase();
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
    tweens.update(dt);
    update_placement_cursor(engine);
    update_tiling_background(grass, tile_size);
    handle_input(engine);
    render_ui();
  }
};

int main() {
  engine_t engine;
  interactive_camera_t ic;
  farm_manager_t farm(engine, fan::vec2{256, 256});
  engine.loop([&] (f32_t dt) {
    farm.update(engine, dt);
  });
}