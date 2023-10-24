#include fan_pch

struct divider_t {
  enum struct direction_e {
    right,
    down
  };

  struct window_t {
    fan::vec2d position = 0.5;
    fan::vec2d size = 1;
    direction_e direction = direction_e::right;
    int root_change_iterator = -1;
    fan::vec2 count = 0;
    std::vector<window_t> child;
  };

  struct iterator_t {
    std::size_t parent = -1;
    std::size_t child = -1;
    direction_e direction = direction_e::right;
  };

  void insert_to_root(direction_e direction) {
    int idx = (int)direction;
    f64_t width_height = 1.0 / (windows.empty() ? 1 : 2);
    for (auto& i : windows) {
      i.size[idx] /= 2;
      i.position[idx] /= 2;
      for (auto& j : i.child) {
        j.size[idx] /= 2;
        j.position[idx] /= 2;
      }
    }
    windows.push_back({
      .position = {0.5, 0.5}, 
      .size = {1, 1}, 
      .direction = direction
    });
    windows.back().position[idx] = 1.0 - width_height / 2;
    windows.back().size[idx] = width_height;
    ++windows.back().count[idx];
  }

  iterator_t push_child(
    divider_t::window_t* root, 
    direction_e direction, 
    iterator_t it, 
    const fan::vec2& parent_pos, 
    const fan::vec2& parent_size
  ) {
    int idx = (int)direction;
    root->child.push_back({});

    root->child.back().position[idx] = parent_pos[idx] + parent_size[idx];
    root->child.back().position[(idx + 1) & 1] = parent_pos[(idx + 1) & 1];
    root->child.back().size = parent_size;
    root->child.back().direction = direction;

    iterator_t ret;
    ret.parent = it.parent;
    ret.child = root->child.size() - 1;
    ret.direction = direction;
    return ret;
  }

  iterator_t insert_to_parent(direction_e direction, iterator_t it) {
    divider_t::window_t* root = &windows[it.parent];

    int idx = (int)direction;

    f64_t rows_size = (1 + root->child.size());
    f64_t width = 1.0 / (rows_size + 1);
    if (!root->child.size()) {
      root->size[idx] -= (width / rows_size * root->size[idx]);
      root->position[idx] -= root->size[idx] / 2;
    }

    fan::vec2 parent_pos = root->position;
    fan::vec2 parent_size = root->size;
    for (auto& j : root->child) {
      j.size[idx] -= width / rows_size * j.size[idx];
      j.position[idx] -= j.size[idx] / 2;
    }
    if (root->child.size()) {
      parent_size[idx] = root->child.back().size[idx];
      parent_pos[idx] = root->child.back().position[idx] + parent_size[idx];
    }

    return push_child(root, direction, it, parent_pos, parent_size);
  }

  iterator_t insert_to_child(direction_e direction, iterator_t it) {
    divider_t::window_t* root = &windows[it.parent];

    int idx = (int)direction;
    f64_t count = 0;
    int root_idx = 0;
    for (int j = windows[it.parent].child.size(); j--; ) {
      if (root_idx == 0 && windows[it.parent].child[j].root_change_iterator != -1) {
        root_idx = windows[it.parent].child[j].root_change_iterator;
      }
      if (root->child[j].direction == (direction_e)((idx + 1) & 1)) {
        continue;
      }
      ++count;
    }

    f64_t rows_size = (1 + count);
    f64_t width = 1.0 / (rows_size + 1);

    if (root_idx == 0 && root->direction == direction) {
      ++count;
    }

    if (root->child.size()) {
      if (it.child == 0) {
        if (root->direction != direction) {
          root->direction = direction;
        }
      }
      if (root->child[it.child].direction != direction) {
        root_idx = it.child;
        root->child[root_idx].root_change_iterator = root_idx;
        root->child[root_idx].direction = direction;
        ++count;
      }
    }
    auto& current = root->child[root_idx];
    divider_t::window_t* prev;
    if (root_idx - 1 == -1) {
      prev = root;
    }
    else {
      prev = &root->child[root_idx - 1];
    }
    auto old_size = current.size[idx];
    count = std::max(count, 1.0) + ((idx + 1) & 1);//+ idx
    current.size[idx] = prev->size[idx] / count;
    current.position[idx] -= (old_size - current.size[idx]) / 2;
    root_idx += 1;
    for (int j = root_idx; j < root->child.size(); ++j) {
      auto& current = windows[it.parent].child[j];
      current.size[idx] = root->child[root_idx - 1].size[idx];
      current.position[idx] = root->child[j - 1].position[idx] + current.size[idx];
    }
    fan::vec2 parent_pos = root->child[it.child].position;
    fan::vec2 parent_size = root->child[it.child].size;
  
    auto ret = push_child(root, direction, it, parent_pos, parent_size);

    if (it.direction != direction) {
      root->child.back().root_change_iterator = root->child.size() - 2;
    }

    return ret;
  }

  iterator_t insert(direction_e direction, iterator_t it = iterator_t()) {
    bool is_parent = it.parent != (std::size_t)-1;
    bool is_child = it.child != (std::size_t)-1;

    if (!is_parent) {
      insert_to_root(direction);
    }
    else {
      if (!is_child) {
        return insert_to_parent(direction, it);
      }
      else {
        return insert_to_child(direction, it);
      }
    }
    iterator_t ret;
    ret.direction = direction;
    ret.parent = windows.size() - 1;
    return ret;
  }

  std::vector<window_t> windows;
};

int main() {

  divider_t d;

  loco_t loco;

  struct node_t {
    fan::graphics::rectangle_t rect;
  };

  std::vector<node_t> nodes;

  //auto it0 = d.insert(divider_t::direction_e::down);
  //auto it1 = d.insert(divider_t::direction_e::down, it0);
  //d.insert(divider_t::direction_e::down, it1);
  //d.insert(divider_t::direction_e::right, it2);
  //d.insert(divider_t::direction_e::right);
  //d.insert(divider_t::direction_e::down, it2);

  auto i0 = d.insert(divider_t::direction_e::right);
  auto ii = d.insert(divider_t::direction_e::right, i0);
  d.insert(divider_t::direction_e::down);
  auto i1 = d.insert(divider_t::direction_e::down);
  auto i2 = d.insert(divider_t::direction_e::right, i1);
  auto i3 = d.insert(divider_t::direction_e::right);
  auto i4 = d.insert(divider_t::direction_e::right, i3);
  auto i5 = d.insert(divider_t::direction_e::down, i4);
  auto i6 = d.insert(divider_t::direction_e::down, i5);
  auto i7 = d.insert(divider_t::direction_e::down, i6);
  auto i8 = d.insert(divider_t::direction_e::right, i7);
  auto i9 = d.insert(divider_t::direction_e::right, i8);
  auto i10 = d.insert(divider_t::direction_e::right, i9);
  auto i11 = d.insert(divider_t::direction_e::right, i10);

  auto push_rectangle = [&nodes](const fan::vec2& position, const fan::vec2& size) {
    static int depth = 0;
    nodes.push_back(node_t{
      .rect{{
          .position = fan::vec3(position * 2 - 1, depth++),
          .size = size,
          //.color = fan::color::hsv(fan::random::value_i64(0, 360), 100, 100),
        .color = fan::random::color() - fan::color(0, 0, 0, .2),
        .blending = true
        }}
    });
    };

  for (int j = 0; j < d.windows.size(); ++j) {
    push_rectangle(d.windows[j].position, d.windows[j].size);
    for (int i = 0; i < d.windows[j].child.size(); ++i) {
      push_rectangle(d.windows[j].child[i].position, d.windows[j].child[i].size);
    }
  }

  loco.loop([&] {

  });

  return 0;
}
