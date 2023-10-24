#pragma once

namespace fan {
  namespace trees {
    struct quad_tree_t {
      fan::vec2 position;
      fan::vec2 boundary;

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

      enum class direction_e {
        invalid = -1,
        left,
        right,
        up,
        down
      };

      enum class split_side_e {
        invalid = -1,
        vertical,
        horizontal,
      };

      struct road_t {
        uint32_t depth;
        direction_e direction = direction_e::invalid;
        split_side_e split_side = split_side_e::invalid;
        bool root = false;
      };

      struct path_t : std::vector<road_t>{

      };

      bool valid() const {
        return depth != -1;
      }

      void open(fan::vec2 position, fan::vec2 size, uint32_t n) {
        this->position = position;
        this->size = size;
        this->capacity = n;
        this->direction = direction_e::invalid;
        this->divided = false;
        depth = 0;
      }

      std::pair<fan::vec2, fan::vec2> get_position_size(const fan::vec2& p, direction_e direction) {
        fan::vec2 new_size = size * 2;
        switch (direction){
          case direction_e::left: {
            return {fan::vec2(p.x - new_size.x, p.y), fan::vec2(size.x, size.y)};
          }
          case direction_e::right:{
            return {fan::vec2(p.x + new_size.x, p.y), fan::vec2(size.x, size.y)};
          }
          case direction_e::up: {
            return {fan::vec2(p.x, p.y - new_size.y), fan::vec2(size.x, size.y)};
          }
          case direction_e::down: {
            return {fan::vec2(p.x, p.y + new_size.y), fan::vec2(size.x, size.y)};
          }
        }
      }

      void subdivide(direction_e new_direction) {
        if (new_direction != direction_e::invalid && !directions.empty()) {
          fan::throw_error();
        }
        else {
          directions.resize(4);
        }
        for (int i = 0; i < node_count; ++i) {
          directions[i].size = size;
        }
        divided = true;
        if (new_direction == direction_e::invalid) {
          return;
        }
        auto pns = get_position_size(position, new_direction);
        directions[(int)new_direction].open(pns.first, pns.second, capacity);
      }

      path_t insert_impl(path_t& path, direction_e direction, split_side_e split_side, bool root, uint32_t depth_ = 0) {

        if (!divided) {
          subdivide(direction);
          if (direction == direction_e::invalid) {
            return {};
          }
          directions[(int)direction].direction = direction;
          directions[(int)direction].depth = depth_;
          // always insert to center
          auto psn = get_position_size(fan::vec2(0, 0), direction);
          directions[(int)direction].position = psn.first;
          directions[(int)direction].size = psn.second;
          path.push_back({ road_t{depth_, direction, split_side, root} });
          return path;
        }
        // follow path
        else if (depth_ < path.size()) {
          return directions[(int)path[depth_].direction].insert_impl(path, direction, split_side, root, depth_ + 1);
        }
        // insert to current node
        else {
          int idirection = (int)direction;
          // insert to center
          auto psn = get_position_size(fan::vec2(0, 0), direction);
          if (!directions[idirection].valid()) {
            directions[idirection].open(psn.first, psn.second, capacity);
          }
          directions[idirection].direction = direction;
          directions[idirection].depth = depth_;
          path.push_back({road_t{depth_, direction, split_side, root}});
          return path;
        }
      }

      path_t insert(const path_t& path, direction_e direction, bool root, uint32_t depth_ = 0) {
        path_t p = path;
        split_side_e ss;
        switch (direction) {
          case direction_e::left: 
          case direction_e::right: {
            ss = split_tree_t::split_side_e::vertical;
            break;
          }
          case direction_e::up:
          case direction_e::down: {
            ss = split_tree_t::split_side_e::horizontal;
            break;
          }
        }
        return insert_impl(p, direction, ss, root, depth_);
      }
      int32_t depth = -1;
      fan::vec2 position;
      fan::vec2 size;

      direction_e direction;

      uint32_t capacity;

      std::vector<fan::vec2> points;

      static constexpr uint8_t node_count = 4;

      std::vector<split_tree_t> directions;

      bool divided;
    };
  }
}