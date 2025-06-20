#include <fan/pch.h>

#include <fan/graphics/gui/tilemap_editor/renderer0.h>

struct player_t {
  static constexpr fan::vec2 speed{ 450, 450 };

  player_t() {
    visual = fan::graphics::sprite_t{ {
      .position = fan::vec3(0, 0, 10),
      .size = 32 / 2,
      .blending = true
    } };
    loco_t::light_t::properties_t lp;
    lp.position = visual.get_position();
    lp.size = 256;
    lp.color = fan::color(1, 0.4, 0.4, 1);

    lighting = lp;
  }
  void update() {
    f32_t dt = gloco->delta_time;
    f32_t multiplier = 1;
    if (gloco->window.key_pressed(fan::key_shift)) {
      multiplier = 3;
    }
    if (gloco->window.key_pressed(fan::key_d)) {
      velocity.x = speed.x * multiplier;
    }
    else if (gloco->window.key_pressed(fan::key_a)) {
      velocity.x = -speed.x * multiplier;
    }
    else {
      velocity.x = 0;
    }

    if (gloco->window.key_pressed(fan::key_w)) {
      velocity.y = -speed.y * multiplier;
    }
    else if (gloco->window.key_pressed(fan::key_s)) {
      velocity.y = speed.y * multiplier;
    }
    else {
      velocity.y = 0;
    }

    visual.set_velocity(velocity);

    visual.set_position(visual.get_collider_position());
    lighting.set_position(visual.get_position());
  }
  fan::vec2 velocity = 0;
  fan::graphics::collider_dynamic_t visual;
  loco_t::shape_t lighting;
};

f32_t zoom = 2;
void init_zoom() {
  auto& window = gloco->window;
  auto update_ortho = [] {
    fan::vec2 s = gloco->window.get_size();
    gloco->camera_set_ortho(
      gloco->orthographic_render_view.camera,
      fan::vec2(-s.x, s.x) / zoom,
      fan::vec2(-s.y, s.y) / zoom
    );
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
  loco_t::image_load_properties_t lp;
  lp.visual_output = loco_t::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = GL_NEAREST;
  lp.mag_filter = GL_NEAREST;
  loco_t::texturepack_t tp;
  //tp.open_compiled("texture_packs/tilemap.ftp", lp);
  tp.open_compiled("texture_packs/TexturePack", lp);

  fte_renderer_t renderer;
  renderer.open(&tp);

  //auto compiled_map = renderer.compile("tilemaps/map_game0_0.fte");
  auto compiled_map = renderer.compile("map_game0_1.json");
  fan::vec2i render_size(16, 9);
  render_size *= 2;
  render_size += 3;

  fte_loader_t::properties_t p;

  p.position = fan::vec3(0, 0, 0);
  p.size = (render_size * 2) * 32;
  // add custom stuff when importing files
  //p.object_add_cb = [&](fte_loader_t::fte_t::tile_t& tile) {
  //  if (tile.id == "1") {
  //    fan::print("a");
  //  }
  //};

  init_zoom();

  auto map_id0_t = renderer.add(&compiled_map, p);

  player_t player;

  fan::graphics::bcol.PreSolve_Shape_cb = [](
    bcol_t* bcol,
    const bcol_t::ShapeInfoPack_t* sip0,
    const bcol_t::ShapeInfoPack_t* sip1,
    bcol_t::Contact_Shape_t* Contact
    ) {
      // player
      auto* obj0 = bcol->GetObjectExtraData(sip0->ObjectID);
      // wall
      auto* obj1 = bcol->GetObjectExtraData(sip1->ObjectID);
      if (obj1->collider_type == fan::collider::types_e::collider_sensor) {
        bcol->Contact_Shape_DisableContact(Contact);
      }

      switch (obj1->collider_type) {
      case fan::collider::types_e::collider_static:
      case fan::collider::types_e::collider_dynamic: {
        // can access shape by obj0->shape
        break;
      }
      case fan::collider::types_e::collider_hidden: {
        break;
      }
      case fan::collider::types_e::collider_sensor: {
        fte_renderer_t::userdata_t userdata = *(fte_renderer_t::userdata_t*)obj1->userdata;
        if (gloco->window.key_state(userdata.key) == userdata.key_state) {
          static int x = 0;
          fan::print("press", x++);
        }

        //fan::print("sensor triggered");
        break;
      }
      }
    };

  loco.set_vsync(0);
  //loco.window.set_max_fps(3);
  f32_t total_delta = 0;


  loco.lighting.ambient = 0.7;

  loco.loop([&] {
    // gloco->get_fps();
    player.update();
    fan::vec2 dst = player.visual.get_position();
    fan::vec2 src = gloco->camera_get_position(gloco->orthographic_render_view.camera);
    // smooth camera
    //fan::vec2 offset = (dst - src) * 4 * gloco->delta_time;
    //gloco->default_camera->camera.set_position(src + offset);
    gloco->camera_set_position(gloco->orthographic_render_view.camera, dst);
    renderer.update(map_id0_t, dst);
    });
}