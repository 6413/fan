#error
// legacy - no astar.hpp
#include <fan/pch.h>

static constexpr fan::vec2ui map_size(32, 32);

#include _FAN_PATH(physics/AStar.hpp)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {

    fan::vec2 window_size = loco.window.get_size();
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);

    generator.setWorldSize({(int)map_size.x, (int)map_size.y});
    generator.setHeuristic(AStar::Heuristic::euclidean);
    generator.setDiagonalMovement(false);

    wall_image.load(&loco, "images/wall.webp");
    path_image.load(&loco, "images/path.webp");

    src_image.load(&loco, "images/src.webp");
    dst_image.load(&loco, "images/dst.webp");

    static auto clear_path = [](pile_t* pile) {
      if (pile->coordinate_list.empty()) {
        return;
      }
      for (uint32_t i = 0; i < pile->coordinate_list.size(); ++i) {
        pile->loco.sprite.erase(&pile->cid[map_size.multiply() + i]);
      }
      pile->coordinate_list.clear();
    };

    static auto make_path = [&]() {
      clear_path(this);

      if (path_src_dst[0].x == -1 || path_src_dst[0].y == -1) {
        return;
      }

      if (path_src_dst[1].x == -1 || path_src_dst[1].y == -1) {
        return;
      }

      coordinate_list = generator.findPath(*(AStar::Vec2i*)path_src_dst[0].data(), *(AStar::Vec2i*)path_src_dst[1].data());

      fan::vec2 window_size = loco.window.get_size();

      loco_t::shapes_t::sprite_t::properties_t p;

      p.camera = &camera;
      p.viewport = &viewport;
      p.image = &path_image;
  
      auto real_size = p.size = window_size / map_size / window_size;

      for (uint32_t i = coordinate_list.size(); i--;) {
      
        if (i == coordinate_list.size() - 1) {
          p.image = &dst_image;
          p.size = real_size;
        }
        else if (i == 0) {
          p.image = &src_image;
          p.size = real_size;
        }
        else {
          p.image = &path_image;
          p.size = real_size / 2;
        }
        p.position = fan::vec2(-1 + real_size.x * coordinate_list[i].x * 2 + real_size.x, -1 + real_size.y * coordinate_list[i].y * 2 + real_size.y);
        loco.sprite.push_back(
          &cid[
            map_size.multiply() + i
          ], 
          p
        );
      }
    };

    loco.window.add_buttons_callback([this, window_size](const fan::window_t::mouse_buttons_cb_data_t& d) {
      if (d.state != fan::mouse_state::press) {
        return;
      }
      if (d.button == fan::mouse_left) {
        fan::vec2 mp = d.window->get_mouse_position();
        if (mp.x < 0 || mp.y < 0) {
          return;
        }
        fan::vec2ui possible = (mp / window_size * map_size).floor();
        if (map[possible.y][possible.x]) {
          return;
        }
        static int counter = 0;
        path_src_dst[counter = (counter + 1) % 2] = possible;
        make_path();
      }
      if (d.button == fan::mouse_right) {
      fan::vec2 mp = d.window->get_mouse_position();
        if (mp.x < 0 || mp.y < 0) {
          return;
        }
        fan::vec2ui possible = (mp / window_size * map_size).floor();
        if (possible.x >= map_size.x || possible.y >= map_size.y) {
          return;
        }
        if (map[possible.y][possible.x]) {
          loco.sprite.erase(&cid[possible.y * map_size.x + possible.x]);
          generator.removeCollision({ (int)possible.x, (int)possible.y });
          map[possible.y][possible.x] = false;
        }
        else {
          loco_t::shapes_t::sprite_t::properties_t p;

          p.camera = &camera;
          p.viewport = &viewport;

          p.image = &path_image;
          p.size = window_size / map_size / window_size ;
          p.image = &wall_image;
          p.position = fan::vec2(-1 + p.size.x * possible.x * 2 + p.size.x, -1 + p.size.y * possible.y * 2 + p.size.y);
          loco.sprite.push_back(
            &cid[
              possible.y * map_size.x + possible.x
            ], 
            p
          );
          generator.addCollision({ (int)possible.x, (int)possible.y });
          map[possible.y][possible.x] = true;
        }
      }
      make_path();
    });

    loco.window.add_keys_callback([&](const fan::window_t::keyboard_keys_cb_data_t& d) {
      if (d.state != fan::keyboard_state::release) {
        return;
      }
      if (d.key == fan::key_v) {
        static bool vsync_toggle = 0;
        vsync_toggle = !vsync_toggle;
        loco.get_context()->set_vsync(loco.get_window(), vsync_toggle);
        fan::print(std::string("vsync:") + (vsync_toggle ? "on" : "off"));
        return;
      }
    });
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cid[map_size.multiply() * 5];
  AStar::Generator generator;
  AStar::CoordinateList coordinate_list;
  loco_t::image_t wall_image;

  using map_t = std::array<std::array<bool, map_size.x>, map_size.y>;

  map_t map{};

  fan::vec2i path_src_dst[2] = {
    fan::vec2i(-1, -1),
    fan::vec2i(-1, -1)
  };

  loco_t::image_t path_image;
  loco_t::image_t src_image;
  loco_t::image_t dst_image;
};

static constexpr const char* map_filename = "map";

pile_t::map_t parse_map() {
  fan::string str;
  fan::io::file::read(map_filename, &str);
  pile_t::map_t map{};
  std::fstream f(map_filename, std::ios_base::in);

  uint32_t i = 0;
  while (i < map_size.multiply() && f >> map[i / map_size.y][i % map_size.x]) {
    f.ignore(100, ',');
    i++;
  }
  return map;
}

void clear_map(pile_t* pile) {
  for (uint32_t i = 0; i < map_size.multiply(); ++i) {
    pile->loco.sprite.erase(&pile->cid[i]);
  }
}

void generate_map(pile_t* pile) {
  fan::vec2 window_size = pile->loco.window.get_size();

  loco_t::shapes_t::sprite_t::properties_t p;

  p.camera = &pile->camera;
  p.viewport = &pile->viewport;

  p.image = &pile->wall_image;
  p.size = window_size / map_size / window_size ;
  for (uint32_t y = 0; y < map_size.y; ++y) {
    for (uint32_t x = 0; x < map_size.x; ++x) {
      if (pile->map[y][x]) {
        p.position = fan::vec2(-1 + p.size.x * x * 2 + p.size.x, -1 + p.size.y * y * 2 + p.size.y);
        pile->loco.sprite.push_back(&pile->cid[y * map_size.x + x], p);
        pile->generator.addCollision({ (int)x, (int)y });
      }
    }
  }

  pile->loco.set_vsync(false);
}

int main() {
  pile_t* pile = new pile_t;

  pile->map = parse_map();
  generate_map(pile);

  pile->loco.loop([&] {
    pile->loco.get_fps(); 
  });

  return 0;
}