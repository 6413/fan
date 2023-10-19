#pragma once

namespace fan {
  namespace trees {
    struct quad_tree_t {
      fan::vec2 position;
      fan::vec2 boundary;
      // how often to split
      uint32_t capacity;
      std::vector<fan::vec2> points;

      quad_tree_t* north_west = nullptr;
      quad_tree_t* north_east = nullptr;
      quad_tree_t* south_west = nullptr;
      quad_tree_t* south_east = nullptr;

      bool divided = false;

      quad_tree_t(fan::vec2 position, fan::vec2 boundary, uint32_t n) {
        this->position = position;
        this->boundary = boundary;
        this->capacity = n;
      }

      void subdivide() {
        // todo free these
        north_east = new quad_tree_t(fan::vec2(position.x + boundary.x / 2, position.y - boundary.y / 2), boundary / 2, capacity);
        north_west = new quad_tree_t(fan::vec2(position.x - boundary.x / 2, position.y - boundary.y / 2), boundary / 2, capacity);
        south_east = new quad_tree_t(fan::vec2(position.x + boundary.x / 2, position.y + boundary.y / 2), boundary / 2, capacity);
        south_west = new quad_tree_t(fan::vec2(position.x - boundary.x / 2, position.y + boundary.y / 2), boundary / 2, capacity);
        divided = true;
      }

      constexpr bool contains(const fan::vec2& point, const fan::vec2& position, const fan::vec2& size) {
        return
          point.x > position.x - size.x &&
          point.x < position.x + size.x &&
          point.y > position.y - size.y &&
          point.y < position.y + size.y;
      }

      void insert(const fan::vec2& point, fan::vec2& inserted_position, fan::vec2& inserted_size) {
        if (!contains(point, position, boundary)) {
          return;
        }

        if (points.size() < capacity && !divided) {
          inserted_position = position;
          inserted_size = boundary;
          points.push_back(point);
        }
        else {
          if (!divided) {
            subdivide();
            for (auto& i : points) {
              insert(i, inserted_position, inserted_size);
            }
            points.clear();
          }
          north_west->insert(point, inserted_position, inserted_size);
          north_east->insert(point, inserted_position, inserted_size);
          south_west->insert(point, inserted_position, inserted_size);
          south_east->insert(point, inserted_position, inserted_size);
        }
      }
    };

    struct split_tree_t {
      fan::vec2 position;
      fan::vec2 boundary;

      int direction = -1;
      int split_side = -1;

      uint32_t capacity;
      std::vector<fan::vec2> points;

      split_tree_t* vertical[2]{ nullptr };
      split_tree_t* horizontal[2]{ nullptr };

      bool divided = false;

      void open(fan::vec2 position, fan::vec2 boundary, uint32_t n) {
        this->position = position;
        this->boundary = boundary;
        this->capacity = n;
      }

      void subdivide(int side) {
        // todo free these
        if (side == 0) {
          horizontal[0] = new split_tree_t(fan::vec2(position.x, position.y - boundary.y / 2), fan::vec2(boundary.x, boundary.y / 2), capacity);
          horizontal[1] = new split_tree_t(fan::vec2(position.x, position.y + boundary.y / 2), fan::vec2(boundary.x, boundary.y / 2), capacity);
        }
        else {
          vertical[0] = new split_tree_t(fan::vec2(position.x - boundary.x / 2, position.y), fan::vec2(boundary.x / 2, boundary.y), capacity);
          vertical[1] = new split_tree_t(fan::vec2(position.x + boundary.x / 2, position.y), fan::vec2(boundary.x / 2, boundary.y), capacity);
        }
        divided = true;
      }

      int size = 0;

      struct path_t {
        uint32_t depth;
        int dir;
        int split_side;
      };
      // split side which side you want to get returned
      std::vector<path_t> insert(std::vector<path_t>& path, int direction, int split_side, uint32_t depth = 0) {

        if (!divided) {
          this->direction = direction;
          this->split_side = split_side;
          subdivide(direction);
          path.push_back({ path_t{depth, direction, -1} });
          return path;
        }
        else if (path.size() <= depth) {
          subdivide(split_side);
          divided = true;
          this->split_side = split_side;
          path.push_back({ path_t{depth, direction, -1} });
          return path;
        }
        else if (!path.empty()) {
          if (path[depth].dir == 0) {
            if (path[depth].split_side == (uint32_t)-1) {
              path[depth].split_side = split_side;
              horizontal[split_side]->insert(path, direction, split_side, depth + 1);
            }
            else {
              horizontal[path[depth].split_side]->insert(path, direction, split_side, depth + 1);
            }
          }
          else if (path[depth].dir == 1) {
            if (path[depth].split_side == (uint32_t)-1) {
              path[depth].split_side = split_side;
              vertical[split_side]->insert(path, direction, split_side, depth + 1);
            }
            else {
              vertical[path[depth].split_side]->insert(path, direction, split_side, depth + 1);
            }
          }
          return path;
        }
      }
    };
  }
}