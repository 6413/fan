#include fan_pch

struct divider_t {
  enum struct direction_e {
    right,
    down
  };

  struct window_t {
    fan::vec2d position = 0.5;
    fan::vec2d size = 1;
    int vert_or_hori = 0;
    int root_change_iterator = -1;
    std::vector<window_t> child;
  };

  struct iterator_t {
    std::size_t parent = -1;
    std::size_t child = -1;
    int vert_or_hori = 0;
  };

  iterator_t insert(direction_e direction, iterator_t it = iterator_t()) {
    bool is_parent = it.parent != (std::size_t)-1;
    bool is_child = it.child != (std::size_t)-1;
    if (direction == direction_e::right) {
      if (!is_parent) {
        f64_t width = 1.0 / (windows.empty() ? 1 : 2);
        for (auto& i : windows) {
          i.size.x /= 2;
          i.position.x /= 2;
          for (auto& j : i.child) {
            j.size.x /= 2;
            j.position.x /= 2;
          }
        }
        windows.push_back({.position = {1.0 - width / 2, 0.5}, .size = {width, 1.0}, .vert_or_hori = 0});
        ++size.x;
      }
      else {
        if (!is_child) {
          divider_t::window_t* root = &windows[it.parent];

          f64_t rows_size = (1 + root->child.size());
          f64_t width = 1.0 / (rows_size + 1);
          if (!root->child.size()) {
            root->size.x -= (width / rows_size * root->size.x);
            root->position.x -= root->size.x / 2;
          }

          fan::vec2 parent_pos = root->position;
          fan::vec2 parent_size = root->size;
          for (auto& j : root->child) {
            j.size.x -= width / rows_size * j.size.x;
            j.position.x -= j.size.x / 2;
          }
          if (root->child.size()) {
            parent_size.x = root->child[root->child.size() - 1].size.x;
            parent_pos.x = root->child[root->child.size() - 1].position.x + parent_size.x;
          }
          //root->child.push_back({{parent_pos.x + parent_size.x / 2 - width / 2, 0.5}, {width, parent_size.y}});
          root->child.push_back({
            .position = {parent_pos.x + parent_size.x, parent_pos.y},
            .size = {parent_size.x, parent_size.y},
            .vert_or_hori = 0
          });
          iterator_t ret;
          ret.parent = it.parent;
          ret.child = root->child.size() - 1;
          ret.vert_or_hori = 0;
          return ret;
        }
        else {
          divider_t::window_t* root = &windows[it.parent];

          f64_t count_x = 0;
          int root_y = 0;
          bool has_split = false;
          for (int j = windows[it.parent].child.size(); j--; ) {
            if (root_y == 0 && windows[it.parent].child[j].root_change_iterator != -1) {
              root_y = windows[it.parent].child[j].root_change_iterator;
              has_split = true;
            }
            if (root->child[j].vert_or_hori == 1) {
              continue;
            }
            ++count_x;
          }

          f64_t rows_size = (1 + count_x);
          f64_t width = 1.0 / (rows_size + 1);

          if (root_y == 0 && root->vert_or_hori == 0) {
            ++count_x;
          }

          if (root->child.size()) {
            if (it.child == 0) {
              if (root->vert_or_hori != (int)direction) {
                root->vert_or_hori = 0;
              }
            }
            if (root->child[it.child].vert_or_hori != (int)direction) {
              root_y = it.child;
              root->child[root_y].root_change_iterator = root_y;
              root->child[root_y].vert_or_hori = 0;
              ++count_x;
            }
          }
          auto& current = root->child[root_y];
          divider_t::window_t* prev;
          if (root_y - 1 == -1) {
            prev = root;
          }
          else {
            prev = &root->child[root_y - 1];
          }
          auto old_size = current.size.x;
          count_x = std::max(count_x, 1.0) + 1;
          current.size.x = prev->size.x / count_x;
          current.position.x -= (old_size - current.size.x) / 2;
          root_y += 1;
          for (int j = root_y; j < root->child.size(); ++j) {
            auto& current = windows[it.parent].child[j];
            current.size.x = root->child[root_y - 1].size.x;
            current.position.x = root->child[j - 1].position.x + current.size.x;
          }
          fan::vec2 parent_pos = root->child[it.child].position;
          fan::vec2 parent_size = root->child[it.child].size;
          root->child.push_back({
            .position = {parent_pos.x + parent_size.x, parent_pos.y},
            .size = {parent_size.x, parent_size.y},
            .vert_or_hori = 0
          });
          iterator_t ret;
          ret.parent = it.parent;
          ret.child = root->child.size() - 1;
          ret.vert_or_hori = 0;
          if (it.vert_or_hori != ret.vert_or_hori) {
            root->child[root->child.size() - 1].root_change_iterator = root->child.size() - 2;
          }
          return ret;
        }
      }
      iterator_t ret;
      ret.vert_or_hori = 0;
      ret.parent = windows.size() - 1;
      return ret;
    }
    if (direction == direction_e::down) {
      if (!is_parent) {

        f64_t height = 1.0 / (windows.empty() ? 1 : 2);
        for (auto& i : windows) {
          i.size.y /= 2;
          i.position.y /= 2;
          for (auto& j : i.child) {
            j.size.y /= 2;
            j.position.y /= 2;
          }
        }
        windows.push_back({
          .position = {0.5, 1.0 - height / 2},
          .size = {1.0, height},
          .vert_or_hori = 1
        });
        ++size.y;
      }
      else {
        // doesnt support erase yet, use bll to support erase

        // parent + children size
        // ?

        if (!is_child) {
          divider_t::window_t* root = &windows[it.parent];

          f64_t rows_size = (1 + root->child.size());
          f64_t width = 1.0 / (rows_size + 1);
          if (!root->child.size()) {
            root->size.y -= (width / rows_size * root->size.y);
            root->position.y -= root->size.y / 2;
          }

          fan::vec2 parent_pos = root->position;
          fan::vec2 parent_size = root->size;
          for (auto& j : root->child) {
            j.size.y -= width / rows_size * j.size.y;
            j.position.y -= j.size.y / 2;
          }
          if (root->child.size()) {
            parent_size.y = root->child[root->child.size() - 1].size.y;
            parent_pos.y = root->child[root->child.size() - 1].position.y + parent_size.y;
          }
          //root->child.push_back({{parent_pos.x + parent_size.x / 2 - width / 2, 0.5}, {width, parent_size.y}});
          root->child.push_back({
            .position = {parent_pos.x, parent_pos.y + parent_size.y},
            .size = {parent_size.x, parent_size.y},
            .vert_or_hori = 1
          });
          iterator_t ret;
          ret.parent = it.parent;
          ret.child = root->child.size() - 1;
          ret.vert_or_hori = 1;
          return ret;
        }
        else {
          divider_t::window_t* root = &windows[it.parent];

          f64_t count_y = 0;
          int root_x = 0;
          bool has_split = false;
          for (int j = windows[it.parent].child.size(); j--;) {
            if (j == -1) {
              break;
            }
            if (root_x == 0 && windows[it.parent].child[j].root_change_iterator != -1) {
              root_x = windows[it.parent].child[j].root_change_iterator;
            }
            if (root->child[j].vert_or_hori == 0) {
              continue;
            }
            ++count_y;
          }

          f64_t columns_size = (1 + count_y);
          f64_t height = 1.0 / (columns_size + 1);

          if (root_x == 0 && root->vert_or_hori == 1) {
            ++count_y;
          }

          if (root->child.size()) {
            if (it.child == 0) {
              if (root->vert_or_hori != (int)direction) {
                root->vert_or_hori = 1;
              }
            }
            if (root->child[it.child].vert_or_hori != (int)direction) {
              root_x = it.child;
              root->child[root_x].root_change_iterator = root_x;
              root->child[root_x].vert_or_hori = 1;
              //--count_y;
            }
          }
          auto& current = root->child[root_x];
          divider_t::window_t* prev;
          if (root_x - 1 == -1) {
            prev = root;
          }
          else {
            prev = &root->child[root_x - 1];
          }
          auto old_size = current.size.y;
          count_y = std::max(count_y, 1.0);
          current.size.y = prev->size.y / count_y;
          current.position.y -= (old_size - current.size.y) / 2;
          root_x += 1;
          for (int j = root_x; j < root->child.size(); ++j) {
            auto& current = windows[it.parent].child[j];
            current.size.y = root->child[root_x - 1].size.y;
            current.position.y = root->child[j - 1].position.y + current.size.y;
          }
          fan::vec2 parent_pos = root->child[it.child].position;
          fan::vec2 parent_size = root->child[it.child].size;
          root->child.push_back({
            .position = {parent_pos.x, parent_pos.y + parent_size.y},
            .size = {parent_size.x, parent_size.y},
            .vert_or_hori = 1
          });
          iterator_t ret;
          ret.parent = it.parent;
          ret.child = root->child.size() - 1;
          ret.vert_or_hori = 1;
          if (it.vert_or_hori != ret.vert_or_hori) {
            root->child[root->child.size() - 1].root_change_iterator = root->child.size() - 2;
          }
          return ret;
        }
      }
      iterator_t ret;
      ret.parent = windows.size() - 1;
      ret.vert_or_hori = 1;
      return ret;
    }
  }

  std::vector<window_t> windows;
  fan::vec2 size = 0;
};

int main() {

  divider_t d;

  loco_t loco;

  struct node_t {
    fan::graphics::rectangle_t rect;
  };

  std::vector<node_t> nodes;
  struct a_t {
    divider_t::direction_e dir;
    bool child = false;
  };

  std::vector<a_t> directions;

  auto i0 = d.insert(divider_t::direction_e::right);
  auto ii = d.insert(divider_t::direction_e::right, i0);
  //d.insert(divider_t::direction_e::right, ii);
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
