#include <fan/pch.h>

#include _FAN_PATH(graphics/gui/tilemap_editor/renderer0.h)
#include <bitset>

#include _FAN_PATH(graphics/entity.h)

struct entities_t;

struct entities_ptr_t {
  entities_t* entities = nullptr;
  entities_t* operator->() {
    return entities;
  }
}entites_ptr;

struct entities_t {

  fan::graphics::EntityList_t entity_list;

  struct EntityIdentify_t {
    enum {
      zombie
    };
  };

  #define EntityStructBegin \
    struct CONCAT2(set_EntityName,_t){ \
      using lstd_current_type = CONCAT2(set_EntityName,_t); \
      struct EntityData_t; \
      EntityData_t* ged(fan::graphics::EntityID_t EntityID){ \
        auto Entity = entites_ptr->entity_list.Get(EntityID); \
        return (EntityData_t *)Entity->UserPTR; \
      }
  #define EntityMakeStatic(name, ...) \
    static void name(lstd_preprocessor_combine_every_2(__VA_ARGS__)){ entites_ptr->set_EntityName.CONCAT2(___,name)(lstd_preprocessor_ignore_first_of_every_2(__VA_ARGS__)); } \
    void CONCAT2(___,name)(lstd_preprocessor_combine_every_2(__VA_ARGS__))

  #include "entity/zombie/zombie.h"


  #undef EntityMakeStatic
  #undef EntityStructBegin
};

struct player_t {

  static constexpr fan::vec2 speed{ 150, 150 };
  static constexpr f32_t speed_walk = 1;
  static constexpr f32_t speed_run = 1.5;
  static constexpr f32_t animation_speed = 0.2e+9;
  static constexpr fan::vec2 visual_offset = fan::vec2(0, -10);

  enum directions_e {
    left,
    right,
    up,
    down
  };

  void handle_direction(int direction, fan::keyboard_state state) {
    if (state == fan::keyboard_state::press) {
      key_directions[direction] = true;
      gloco->shapes.sprite_sheet.start(sheet_id, direction * 2, 2);
      velocity[direction / 2] = speed[direction / 2] * speed_multiplier * ((direction % 2) == 0 ? -1 : 1);
      int temp = (direction + 2) % 4;
      velocity[temp / 2] = 0;
    }
    else if (state == fan::keyboard_state::release) {
      key_directions[direction] = false;
      for (int i = direction + 1; i < direction + 4; ++i) {
        if (key_directions[i % 4]) {
          gloco->shapes.sprite_sheet.start(sheet_id, (i % 4) * 2, 2);
          velocity[(i % 4) / 2] = speed[(i % 4) / 2] * speed_multiplier * ((i % 4) % 2 == 0 ? -1 : 1);
          velocity[((i + 2) % 4) / 2] = 0;
          break;
        }
      }
      if (key_directions.none()) {
        gloco->shapes.sprite_sheet.stop(sheet_id);
      }
    }
  };


  player_t() {

    static constexpr const char* directions[] = { "left", "right", "up", "down" };

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 2; ++j) {
        frames[i * 2 + j].load(fan::format("images/boy_{}_{}.webp", directions[i], std::to_string(j + 1)));
      }
    }

    loco_t::shapes_t::sprite_sheet_t::properties_t ssp;
    ssp.images = frames;
    ssp.count = std::size(frames);
    ssp.animation_speed = animation_speed;
    ssp.blending = true;
    ssp.position = fan::vec3(0, 0, 3);
    ssp.size = 16;
    
    sheet_id = gloco->shapes.sprite_sheet.push_back(ssp);

    collider = fan::graphics::collider_dynamic_hidden_t(ssp.position, ssp.size / 4);

    loco_t::shapes_t::light_t::properties_t lp;
    lp.position = gloco->shapes.sprite_sheet.get_position(sheet_id);
    lp.size = 256;
    lp.color = fan::color(1, 0.4, 0.4, 1);

    lighting = lp;
    
    gloco->window.add_keys_callback([&](const auto& d) {
      switch (d.key) {
        case fan::key_a: {
          handle_direction(left, d.state);
          break;
        }
        case fan::key_d: {
          handle_direction(right, d.state);
          break;
        }
        case fan::key_w: {
          handle_direction(up, d.state);
          break;
        }
        case fan::key_s: {
          handle_direction(down, d.state);
          break;
        }
        case fan::key_left_shift: {
          if (d.state == fan::keyboard_state::press) {
            speed_multiplier = speed_run;
            if (velocity.x || velocity.y) {
              velocity = fan::vec2(fan::vec2b(velocity)) * speed * speed_multiplier * velocity.sign();
            }
            gloco->shapes.sprite_sheet.get_sheet_data(sheet_id).animation_speed = animation_speed / speed_run;
          }
          else if (d.state == fan::keyboard_state::release) {
            speed_multiplier = speed_walk;
            if (velocity.x || velocity.y) {
              velocity = fan::vec2(fan::vec2b(velocity)) * speed *  speed_multiplier * velocity.sign();
            }
            gloco->shapes.sprite_sheet.get_sheet_data(sheet_id).animation_speed = animation_speed / speed_walk;
          }
          break;
        }
      }
    });
    gloco->shapes.sprite_sheet.set_image(sheet_id, &frames[0]);
  }
  void update() {
    f32_t dt = gloco->delta_time;

    if (!key_directions[left] &&
      !key_directions[right]) {
      velocity.x = 0;
    }

    if (!key_directions[up] &&
      !key_directions[down]) {
      velocity.y = 0;
    }

    collider.set_velocity(velocity);
    gloco->shapes.sprite_sheet.set_position(sheet_id, collider.get_collider_position() + visual_offset);
    lighting.set_position(gloco->shapes.sprite_sheet.get_position(sheet_id));
  }
  std::bitset<4> key_directions;
  loco_t::shapes_t::sprite_sheet_t::nr_t sheet_id;
  fan::vec2 velocity = 0;
  fan::graphics::collider_dynamic_hidden_t collider;
  loco_t::shape_t lighting;
  loco_t::image_t frames[4 * 2];
  f32_t speed_multiplier = 1;
};

f32_t zoom = 5;
void init_zoom() {
  auto& window = *gloco->get_window();
  auto update_ortho = [] {
    fan::vec2 s = gloco->window.get_size();
    gloco->default_camera->camera.set_ortho(
      fan::vec2(-s.x, s.x) / zoom,
      fan::vec2(-s.y, s.y) / zoom
    );;
  };

  update_ortho();

  window.add_buttons_callback([&](const auto& d) {
    if (d.button == fan::mouse_scroll_up) {
      zoom *= 1.2;
    }
    else if (d.button == fan::mouse_scroll_down) {
      zoom /= 1.2;
    }
    update_ortho();
  });
}

int main() {
  loco_t loco;
  loco_t::texturepack_t tp;
  tp.open_compiled("texture_packs/tilemap.ftp");

  fte_renderer_t renderer;
  renderer.open(&tp);

  auto compiled_map = renderer.compile("tilemaps/map_game0_0.fte");

  fan::vec2i render_size(16, 9);
  render_size += 5;

  fte_loader_t::properties_t p;

  p.position = fan::vec3(0, 0, 0);
  p.size = (render_size * 2) * 32;
  // add custom stuff when importing files

  init_zoom();

  f32_t size = 1;

  loco_t::shape_t* light_water = nullptr;
  fte_loader_t::fte_t::tile_t light_water_tile_data;

  renderer.id_callbacks["light_water"] = [&](fte_loader_t::tile_draw_data_t& data, fte_loader_t::fte_t::tile_t& tile) {
    std::visit([&]<typename T>(T& data) {
      if constexpr (std::is_same_v<T, loco_t::shape_t>) {
        light_water = &data;
        light_water_tile_data = tile;
      }
    }, data);
  };

  auto map_id0_t = renderer.add(&compiled_map, p);

  player_t player;
  entities_t entities;
  entites_ptr.entities = &entities;

  for (int i = 0; i < 3; ++i) {
    entities.zombie.Add(fan::vec2(32 * 2, 0));
  }
  //entities.zombie.Add(fan::vec2(32 * 2, -10));
  //entities.zombie.Add(fan::vec2(32 * 2, -20));
 // entities.zombie.Add(fan::vec2(32 * 2, 10));
 // entities.zombie.Add(fan::vec2(32 * 2, 20));

  loco.set_vsync(0);
  loco.lighting.ambient = fan::vec3(0.1);
  //loco.window.set_max_fps(3);
  f32_t total_delta = 0;

  loco.loop([&] {
    size = 0.5 + 0.5 * std::abs(sin(total_delta));

    gloco->get_fps();
    player.update();
    fan::vec2 dst = gloco->shapes.sprite_sheet.get_position(player.sheet_id);
    fan::vec2 src = gloco->default_camera->camera.get_position();
    // smooth camera
    fan::vec2 offset = (dst - src) * 4 * gloco->delta_time;
    fan::vec3 render_position = src + offset; // dst
    gloco->default_camera->camera.set_position(render_position);

    renderer.update(map_id0_t, render_position);

    if (fan_2d::collision::rectangle::point_inside_no_rotation(render_position, light_water_tile_data.position, render_size * 32 - 32)) {
      light_water->set_color(light_water_tile_data.color * size);
    }

    total_delta += gloco->delta_time;

    entities.entity_list.Step(gloco->delta_time);
  });
}