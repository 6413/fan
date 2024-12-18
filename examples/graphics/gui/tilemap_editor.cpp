#include <fan/pch.h>

#include <fan/graphics/gui/tilemap_editor/editor.h>
#include <fan/graphics/gui/tilemap_editor/renderer0.h>

// editor
fan::graphics::camera_t camera0;
// program
fan::graphics::camera_t camera1;

struct player_t {
  static constexpr fan::vec2 speed{ 300, 300};

  player_t() {
    visual = fan::graphics::sprite_t{ {
      .camera = &camera1,
      .position = fan::vec3(0, 0,  10),
      .size = 32 / 2,
      .blending = true,
    } };
    loco_t::light_t::properties_t lp;
    lp.position = visual.get_position();
    lp.size = 256;
    lp.color = fan::color(1, 0.4, 0.4, 1);
    lp.camera = camera1.camera;
    lp.viewport = camera1.viewport;

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

    fan::vec2 position = visual.get_collider_position();
    visual.set_position(fan::vec3(position, floor(position.y / 64) + fte_t::shape_depths_t::max_layer_depth / 2));
    lighting.set_position(visual.get_position());
  }
  fan::vec2 velocity = 0;
  fan::graphics::collider_dynamic_t visual;
  loco_t::shape_t lighting;
};

f32_t zoom = 2;
bool hovered = false;
void init_zoom() {
  auto& window = gloco->window;
  auto update_ortho = [&] {
    fan::vec2 s = gloco->window.get_size();
    gloco->camera_set_ortho(
      camera1.camera,
      fan::vec2(-s.x, s.x) / zoom,
      fan::vec2(-s.y, s.y) / zoom
    );
  };

  update_ortho();

  window.add_buttons_callback([&](const auto& d) {
    if (!hovered) {
      return;
    }
    if (d.button == fan::mouse_scroll_up) {
      zoom *= 1.2;
    }
    else if (d.button == fan::mouse_scroll_down) {
      zoom /= 1.2;
    }
    update_ortho();
    });
}

int main(int argc, char** argv) {

  //
  loco_t loco;
  loco.window.set_windowed_fullscreen();


  camera0.camera = loco.camera_create();
  camera1.camera = loco.camera_create();
  fan::vec2 window_size = loco.window.get_size();
  loco.camera_set_ortho(
    camera0.camera,
    fan::vec2(-window_size.x, window_size.x),
    fan::vec2(-window_size.y, window_size.y)
  );
  loco.camera_set_ortho(
    camera1.camera,
    fan::vec2(-window_size.x, window_size.x),
    fan::vec2(-window_size.y, window_size.y)
  );

  camera0.viewport = loco.open_viewport(
    0,
    { 1, 1 }
  );
  camera1.viewport = loco.open_viewport(
    0,
    { 1, 1 }
  );


  init_zoom();

  fte_t fte;//
  fte_t::properties_t p;
  p.camera = &camera0;
  fte.open(p);
  fte.open_texturepack("platformer.ftp");

  std::unique_ptr<player_t> player;
  std::unique_ptr<fte_renderer_t> renderer;
  bool render_scene = false;
  std::unique_ptr<fte_renderer_t::id_t> map_id0_t;

  auto reload_scene = [&] {
    {
      renderer = std::make_unique<fte_renderer_t>();

      loco_t::image_load_properties_t lp;
      lp.visual_output = loco_t::image_sampler_address_mode::clamp_to_border;
      lp.min_filter = fan::opengl::GL_NEAREST;
      lp.mag_filter = fan::opengl::GL_NEAREST;
      renderer->open(&fte.texturepack);

      // STATIC POINTER
      static fte_loader_t::compiled_map_t compiled_map;
      compiled_map = renderer->compile(fte.file_name + "temp");
      fan::vec2i render_size(16, 9);
      render_size *= 2;
      render_size += 3;

      fte_loader_t::properties_t p;

      p.position = fan::vec3(0, 0, 0);
      p.size = (render_size * 2) * 32;

      p.camera = &camera1;
      map_id0_t = std::make_unique<fte_renderer_t::id_t>(renderer->add(&compiled_map, p));

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
              fan::print("sensor triggered");
              break;
            }
          }
        };

      loco.set_vsync(0);
      //loco.window.set_max_fps(3);
      f32_t total_delta = 0;
    }
  };

  loco.window.add_keys_callback([&](const auto& d) {
    if (d.state != fan::keyboard_state::press) {
      return;
    }
    switch (d.key) {
      case fan::key_f5: {
        render_scene = !render_scene;
        if (render_scene) {
          player = std::make_unique<player_t>();
          fte.fout(fte.file_name + "temp");
          reload_scene();
        }
        else {
          if (map_id0_t) {
            renderer.get()->clear(renderer.get()->map_list[*map_id0_t.get()]);
          }
          map_id0_t.reset();
          renderer.reset();
          player.reset();
        }
        break;
      }
    }
  });

  fte.modify_cb = [&](int mode) {
    if (map_id0_t) {
      renderer.get()->clear(renderer.get()->map_list[*map_id0_t.get()]);
    }
    map_id0_t.reset();
    renderer.reset();
    fte.fout(fte.file_name + "temp");
    reload_scene();
  };

  loco.set_vsync(0);

  loco.loop([&] {
    if (render_scene) {
      if (ImGui::Begin("Program", 0, ImGuiWindowFlags_NoBackground)) {
        fan::vec2 s = ImGui::GetContentRegionAvail();
        player->update();
        fan::vec2 dst = player->visual.get_position();
        fan::vec2 src = loco.camera_get_position(camera1.camera);
        // smooth camera
        //fan::vec2 offset = (dst - src) * 4 * gloco->delta_time;
        //gloco->default_camera->camera.set_position(src + offset);
        loco.camera_set_ortho(
          camera1.camera,
          fan::vec2(-s.x, s.x) / zoom,
          fan::vec2(-s.y, s.y) / zoom
        );

        loco.camera_set_position(
          camera1.camera,
          dst
        );
        renderer->update(*map_id0_t, dst);
        loco.set_imgui_viewport(camera1.viewport);
      }
      else {
        loco.viewport_zero(
          camera1.viewport
        );
      }
      hovered = ImGui::IsWindowHovered();
      ImGui::End();

    }
  });
  //
  return 0;
}
