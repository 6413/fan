#include fan_pch

fan::vec2ui grid_size{ 8, 8 };

struct map_t {
  fan::vec2 position;
  uint16_t wall = 0;
};

std::vector<std::vector<map_t>> grid_map;

void generate_map(fan::vec2& block_size) {
  for (int j = 0; j < grid_size.y; ++j) {
    for (int i = 0; i < grid_size.x; ++i) {
      grid_map[j][i].position = block_size / 2 + fan::vec2(i * block_size.x, j * block_size.y);
    }
  }
}

fan::vec2i generate_random_direction() {
  fan::vec2i r = fan::random::vec2i(-1, 1);
  if (r.x == 0) {
    while (r.y == 0) {
      r.y = fan::random::value_i64(-1, 1);
    }
  }
  return r;
}

void try_insert_shape(fan::vec2i& traverse_direction) {
  uint32_t max_tries = 100;
  uint32_t tries = 0;
  fan::vec2i rd;
  fan::vec2i final_direction = 0;
  do {
    rd = generate_random_direction();
    ++tries;
    if (tries >= max_tries) {
      break;
    }
    final_direction.y = traverse_direction.y + rd.y;
    final_direction.x = traverse_direction.x + rd.x;
    final_direction.x = fan::clamp(final_direction.x, 0, (int)grid_size.x - 1);
    final_direction.y = fan::clamp(final_direction.y, 0, (int)grid_size.y - 1);
  } while (grid_map[final_direction.y][final_direction.x].wall != 0);

  // aka failed
  if (traverse_direction.x + rd.x >= grid_size.x || 
      traverse_direction.x + rd.x < 0) {
    rd.x = 0;
  }
  // aka failed
  if (traverse_direction.y + rd.y >= grid_size.y ||
    traverse_direction.y + rd.y < 0) {
    rd.y = 0;
  }
  traverse_direction += rd;
}

void generate_shape(const fan::vec2& block_size, uint32_t piece_count, uint8_t shape_identifier) {
  uint32_t inserted = 0;
  fan::vec2 initial_pos = -1;
  fan::vec2i traverse_direction = 0;
  static int x = 0;

  for (int j = 0; j < grid_size.y; ++j) {
    for (int i = 0; i < grid_size.x; ++i) {
      if (grid_map[j][i].wall == 0) {
        initial_pos = grid_map[j][i].position;
        grid_map[j][i].wall = shape_identifier;
        traverse_direction = initial_pos / block_size;
        ++inserted;
        goto g_end_loop;
      }
    }
  }

  g_end_loop:;
  while (inserted < piece_count) {
    fan::vec2 old_traverse = traverse_direction;
    try_insert_shape(traverse_direction);
    if (old_traverse != traverse_direction) {
      if (grid_map[traverse_direction.y][traverse_direction.x].wall == 0) {
        grid_map[traverse_direction.y][traverse_direction.x].wall = shape_identifier;
      }
      else {
        traverse_direction = old_traverse;
      }
    }
    ++inserted;
  }
g_end_shape_gen:;
}

// i think piece count is not guaranteed to be constant (depends about grid_size)
std::vector<fan::vec2> get_character_pieces(uint32_t character) {
  std::vector<fan::vec2> positions;
  for (int j = 0; j < grid_size.y; ++j) {
    for (int i = 0; i < grid_size.x; ++i) {
      if (grid_map[j][i].wall == character) {
        positions.push_back(grid_map[j][i].position);
      }
    }
  }
  return positions;
}

void load_shape(uint32_t estimated_shape_count, fan::vec2& block_size, loco_t::vfi_t::properties_t& vfip, std::vector<fan::graphics::vfi_multiroot_t>& shapes, uint32_t character) {

  fan::color shape_color = fan::color::hsv(character * (360.f / (grid_size.multiply() / estimated_shape_count)), fan::random::value_i64(28, 100), 100);

  std::vector<fan::vec2> characters = get_character_pieces(character);
  uint32_t rotate_point_index = characters.size() / 2;

  fan::vec2 total = 0;
  for (auto& piece : characters) {
    total += piece;
  }
  total /= characters.size();


  double min_distance = std::numeric_limits<double>::max();
  fan::vec2 most_central = 0;

  for (auto& piece : characters) {
    fan::vec2 center = piece;
    double distance = std::sqrt(std::pow(center.x - total.x, 2) + std::pow(center.y - total.y, 2));

    if (distance < min_distance) {
      min_distance = distance;
      most_central = piece;
    }
  }

  for (auto& piece : characters) {
    vfip.shape.rectangle->position = piece;
    shapes[character].push_root(vfip);
    shapes[character].push_child(fan::graphics::rectangle_t{ {
        .position = piece,
        .size = block_size / 2,
        .color = shape_color,
        .rotation_point = most_central - piece
    } });
  }
}

void print_grid() {
  for (uint32_t j = 0; j < grid_size.y; ++j) {
    for (uint32_t i = 0; i < grid_size.x; ++i) {
      if (grid_map[j][i].wall) {
        fan::print_no_endline("    ", (int)grid_map[j][i].wall, "   ");
      }
      else {
        fan::print_no_endline("    0   ");
      }
    }
    fan::print("\n\n");
  }
}

void reset_grid() {
  for (uint32_t j = 0; j < grid_size.y; ++j) {
    for (uint32_t i = 0; i < grid_size.x; ++i) {
      grid_map[j][i].wall = 0;
    }
  }
}

bool is_grid_filled() {
  for (uint32_t j = 0; j < grid_size.y; ++j) {
    for (uint32_t i = 0; i < grid_size.x; ++i) {
      if (grid_map[j][i].wall == 0) {
        return false;
      }
    }
  }
  return true;
}

int main() {

  loco_t loco;

  fan::vec2 window_size = loco.window.get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, window_size.x),
    fan::vec2(0, window_size.y)
  );

  fan::vec2 playground_size = window_size / 2;

  fan::vec2 block_size = (playground_size / grid_size).floor().min();

  struct shape_t {
    std::vector<fan::vec2> shape;
    fan::color color;
  };

  // preallocate otherwise root_vfi dies
  std::vector<fan::graphics::vfi_multiroot_t> shapes;

  grid_map.resize(grid_size.y);

  for (int i = 0; i < grid_size.x; ++i) {
    grid_map[i].resize(grid_size.x);
  }

  generate_map(block_size);

  loco_t::shape_t line_grid;

  loco_t::shapes_t::line_grid_t::properties_t p;
  p.position = fan::vec3(playground_size, 0xfff);
  p.size = 0;
  p.color = fan::color::rgb(0, 128, 255);

  line_grid = p;

  gloco->shapes.line_grid.sb_set_vi(
    line_grid,
    &loco_t::shapes_t::line_grid_t::vi_t::grid_size,
    fan::vec2(fan::vec2(loco.window.get_size()) / (block_size / 2))
  );
  line_grid.set_size(loco.window.get_size());



  uint32_t character = 0;
  int max_shape_length = 12;
  while (is_grid_filled() == false) {
    generate_shape(block_size, max_shape_length, character++);
  }
  print_grid();

  // + 1 ? 
  shapes.resize(character);

  for (int i = 0; i < character; ++i) {
    shapes[i].grid_size = block_size;
    loco_t::shapes_t::vfi_t::properties_t vfip;
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle->position.z = i;
    vfip.shape.rectangle->size = block_size / 2;

    vfip.mouse_button_cb = [&shapes, i](const auto& d) {
      if (d.button_state != fan::mouse_state::press) {
        return 0;
      }
      if (d.button != fan::mouse_middle) {
        return 0;
      }
      int id = 0;
      for (auto& child : shapes[i].children) {
        child.set_angle(child.get_angle() + fan::vec3(0, 0, fan::math::pi / 2));
        shapes[i].vfi_root[id]->set_angle(child.get_angle());
        shapes[i].vfi_root[id]->set_rotation_point(child.get_rotation_point());
        id += 1;
      }
      return 0;
    };
    load_shape(max_shape_length, block_size, vfip, shapes, i);
  }

  loco.loop([&] {
    loco.get_fps();
    //fan::print((int)loco.shapes.vfi.focus.mouse.NRI);
  });

}