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
  }
}