#include <fan/pch.h>

fan::vec2ui grid_size{ 16, 16 };

struct map_t {
  fan::vec2 position;
  uint16_t wall = 0;
};

std::vector<std::vector<map_t>> grid_map;
fan::vec2 playground_size = 0;

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

  fan::vec2 c = gloco->window.get_size() / 2 - playground_size / 2;

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
  static f32_t depth = 0;

  static f32_t down = 0;
  fan::vec2 arrival = 150;


  bool randomize = fan::random::value_i64(0, 7) == 0;

  fan::vec3 rangle = 0;
  rangle.z = std::to_array({ 0.f, fan::math::pi / 2, fan::math::pi, fan::math::pi * 1.5f })[fan::random::value_i64(1, 3)];

  fan::vec2 offset = arrival - (c + most_central);
  offset.y += down;
  down += 10;

  for (auto& piece : characters) {
    if (randomize) {
      vfip.shape.rectangle->position = fan::vec3(c + piece + offset, depth);
      vfip.shape.rectangle->angle = fan::vec3(
        0,
        0,
        0
      );
      vfip.shape.rectangle->rotation_point = most_central - piece;

      shapes[character].push_root(vfip);
      shapes[character].push_child(fan::graphics::rectangle_t{ {
        .position = fan::vec3(c + piece + offset, depth),
        .size = block_size / 2,
        .color = shape_color,
        .angle = 0,
        .rotation_point = most_central - piece,
    } });
    }
    else {
      vfip.shape.rectangle->position = fan::vec3(c + piece, depth);
      shapes[character].push_root(vfip);
      shapes[character].push_child(fan::graphics::rectangle_t{ {
        .position = fan::vec3(c + piece, depth),
        .size = block_size / 2,
        .color = shape_color,
        .angle = 0,
        .rotation_point = most_central - piece,
    } });
    } 
    ++depth;
  }

  if (!randomize) {
    return;
  }

  int id = 0;
  for (auto& child : shapes[character].children) {
    child.set_angle(rangle);
    shapes[character].vfi_root[id]->set_angle(child.get_angle());
    shapes[character].vfi_root[id]->set_rotation_point(child.get_rotation_point());
    id += 1;
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

struct MyRectangle {
  ImVec2 min;
  ImVec2 max;

  MyRectangle(ImVec2 _min, ImVec2 _max) : min(_min), max(_max) {}
};

std::vector<MyRectangle> rectangles;
const float rectangleSize = 50.0f;
const int rectanglesPerRow = 5;  // Adjust as needed

void DrawRectangles() {
  ImGuiIO& io = ImGui::GetIO();
  ImVec2 canvasSize(io.DisplaySize.x - 20.0f, io.DisplaySize.y - 20.0f);

  ImGui::Begin("Rectangles");

  fan::vec2 window_size = gloco->window.get_size();
  fan::vec2 viewport_size = ImGui::GetContentRegionAvail();
  fan::vec2 viewport_pos = fan::vec2(ImGui::GetWindowPos() + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
  f32_t zoom = 1;
  fan::vec2 offset = viewport_size - viewport_size / zoom;
  fan::vec2 s = viewport_size;
  gloco->camera_set_ortho(
    gloco->orthographic_camera.camera,
    fan::vec2(-s.x, s.x) / zoom,
    fan::vec2(-s.y, s.y) / zoom
  );

  gloco->viewport_set(
    gloco->orthographic_camera.viewport,
    viewport_pos, viewport_size, window_size
  );


  ImVec2 viewportPos = ImGui::GetWindowPos();

  // Draw existing rectangles
  for (size_t i = 0; i < rectangles.size(); ++i) {
    ImVec2 pos = ImVec2(viewportPos.x + 20.0f + (i % rectanglesPerRow) * (rectangleSize + 5.0f),
      viewportPos.y + 20.0f + (i / rectanglesPerRow) * (rectangleSize + 5.0f));

    ImGui::GetWindowDrawList()->AddRect(pos, ImVec2(pos.x + rectangleSize, pos.y + rectangleSize),
      IM_COL32(255, 255, 0, 255));
  }

  if (ImGui::Button("Add Rectangle")) {
    // Add a new rectangle to the list
    ImVec2 pos = ImVec2(viewportPos.x + 20.0f + (rectangles.size() % rectanglesPerRow) * (rectangleSize + 5.0f),
      viewportPos.y + 20.0f + (rectangles.size() / rectanglesPerRow) * (rectangleSize + 5.0f));
    rectangles.emplace_back(pos, ImVec2(pos.x + rectangleSize, pos.y + rectangleSize));
  }

  ImGui::End();
}



int main() {

  loco_t loco{ { .window_size = 1024 } };

  fan::vec2 window_size = loco.window.get_size();

  gloco->camera_set_ortho(
    gloco->orthographic_camera.camera,
    fan::vec2(0, window_size.x),
    fan::vec2(0, window_size.y)
  );

  playground_size = window_size / 2;

  fan::vec2 block_size = 32;

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

  loco_t::shape_t grid;

  loco_t::grid_t::properties_t p;
  p.position = fan::vec3(gloco->window.get_size() / 2, 0xfff);
  p.size = block_size * grid_size / 2;
  p.color = fan::colors::black;

  grid = p;

  grid.set_grid_size(grid_size);



  uint32_t character = 0;
  int max_shape_length = 4;
  while (is_grid_filled() == false) {
    generate_shape(block_size, max_shape_length, character++);
  }
  print_grid();

  // + 1 ? 
  shapes.resize(character);

  for (int i = 0; i < character; ++i) {
    shapes[i].grid_size = block_size;
    loco_t::vfi_t::properties_t vfip;
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle->position.z = i;
    vfip.shape.rectangle->rotation_point = 0;
    vfip.shape.rectangle->angle = 0;
    vfip.shape.rectangle->size = block_size / 2;

    auto rotate = [&shapes, i]{
      int id = 0;
      for (auto& child : shapes[i].children) {
        child.set_angle(child.get_angle() + fan::vec3(0, 0, fan::math::pi / 2));
        shapes[i].vfi_root[id]->set_angle(child.get_angle());
        shapes[i].vfi_root[id]->set_rotation_point(child.get_rotation_point());
        id += 1;
      }
    };

    vfip.mouse_button_cb = [&shapes, rotate](const auto& d) {
      if (d.button_state != fan::mouse_state::press) {
        return 0;
      }
      if (d.button != fan::mouse_middle) {
        return 0;
      }
      rotate();
      return 0;
    };
    vfip.keyboard_cb = [&shapes, rotate](const auto& d) {
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }
      if (d.key != fan::key_r) {
        return 0;
      }
      rotate();
      return 0;
    };
    load_shape(max_shape_length, block_size, vfip, shapes, i);
  }

  loco.loop([&] {
   // ImGui::Begin("wnd");
    DrawRectangles();
 //   ImGui::End();
    //loco.get_fps();
  });

}