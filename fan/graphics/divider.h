#pragma once

#include <vector>

namespace fan {
  namespace graphics {
    struct viewport_divider_t {
      enum struct direction_e {
        right,
        down
      };

      struct window_t {
        fan::vec2d position = 0.5;
        fan::vec2d size = 1;
        direction_e direction = direction_e::right;
        int root_change_iterator = -1;
        fan::vec2ui count = 0;
        fan::vec2d root_size = 1;
        std::vector<window_t> child;
      };

      struct iterator_t {
        window_t* parent = nullptr;
        window_t* child = nullptr;
        direction_e direction = direction_e::right;
      };

      void iterate_children(viewport_divider_t::window_t& window, auto l) {
        fan::function_t<void(viewport_divider_t::window_t& node)> iterator;
        iterator = [&](viewport_divider_t::window_t& node) {
          l(node);
          for (viewport_divider_t::window_t& child : node.child) {
            iterator(child);
          }
          };
        iterator(window);
      }
      void iterate(auto l) {
        for (auto& i : windows) {
          iterate_children(i, l);
        }
      }

      iterator_t insert_to_root(direction_e direction) {
        int idx = (int)direction;
        f64_t width_height;
        if (windows.empty()) {
          width_height = 1.0;
        }
        else {
          width_height = windows.back().root_size[idx] / 2;
          //      might not be necessary^
        }
        fan::vec2d root_size = 1;

        iterate([idx](auto& node) {
          node.position[idx] /= 2;
          node.size[idx] /= 2;
          node.root_size[idx] /= 2;
        });
        if (windows.size()) {
          root_size[idx] = 0.5; // old_root?
        }
        windows.push_back({
          .position = {0.5, 0.5},
          .size = {1, 1},
          .direction = direction
        });
        if (windows.size() != 1) {
          windows.back().position[idx] = 0.75;
          windows.back().size[idx] = 0.5;
        }
        windows.back().root_size = root_size;
        ++windows.back().count[idx];
        iterator_t ret;
        ret.direction = direction;
        ret.parent = &windows.back();
        return ret;
      }

      iterator_t push_child(
        viewport_divider_t::window_t* root,
        direction_e direction,
        iterator_t& it,
        const fan::vec2d& parent_pos,
        const fan::vec2d& parent_size
      ) {
        int idx = (int)direction;
        root->child.push_back({});

        root->child.back().position[idx] = parent_pos[idx] + parent_size[idx];
        root->child.back().position[(idx + 1) & 1] = parent_pos[(idx + 1) & 1];
        root->child.back().size = parent_size;
        root->child.back().direction = direction;
        it.parent->count[idx] += 1;

        iterator_t ret;
        ret.parent = it.parent;
        ret.child = &root->child.back();
        ret.direction = direction;
        return ret;
      }

      iterator_t insert_to_child(iterator_t& it, direction_e direction) {
        viewport_divider_t::window_t* root = it.parent;
        int idx = (int)direction;

        if (root->child.size()) {
          // make new children for new direction
          if (it.child->direction != direction) {
            it.parent = &it.parent->child.back();
            it.child = nullptr;
            if (root->count.x == 0) {
              it.parent->root_size.x = root->root_size.x;
            }
            else {
              it.parent->root_size.x = root->root_size.x / root->count.x;
            }
            if (root->count.y == 0) {
              it.parent->root_size.y = root->root_size.y;
            }
            else {
              it.parent->root_size.y = root->root_size.y / root->count.y;
            }
            it.parent->count = 0;
            it.parent->count[idx] += 1;
            return insert(it, direction);
          }
        }

        int count = it.parent->count[idx];
        count = std::max(1, count) + 1;

        auto old_size = root->size[idx];
        root->size[idx] = root->root_size[idx] / count; // ?
        root->position[idx] -= (old_size - root->size[idx]) / 2;
        //root_idx += 1;

        iterator_t ret;

        if (it.child != nullptr) {
          for (int j = 0; j < it.parent->child.size(); ++j) {
            auto& c = it.parent->child[j];
            auto old_size = c.size[idx];
            c.size[idx] = root->size[idx];
            c.position[idx] = root->position[idx] + c.size[idx] * (j + 1);
          }
          fan::vec2d parent_pos = it.child->position;
          fan::vec2d parent_size = it.child->size;
          ret = push_child(root, direction, it, parent_pos, parent_size);
        }
        else {
          root->count[idx] = count - 1;
          ret = push_child(root, direction, it, root->position, root->size);
        }

        return ret;
      }

      iterator_t insert(iterator_t& it, direction_e direction) {
        return insert_to_child(it, direction);
      }
      iterator_t insert(direction_e direction) {
        return insert_to_root(direction);
      }

      std::vector<window_t> windows;
    };
  }
}