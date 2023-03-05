#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

struct pile_t;

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_opengl

#define loco_window
#define loco_context

#define loco_no_inline

#define loco_rectangle
#define loco_sprite
#define loco_button
#include _FAN_PATH(graphics/loco.h)

// in stagex.h getting pile from mouse cb
// pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);

struct pile_t {
  loco_t loco;

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    fan::vec2 window_size = loco.get_window()->get_size();
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
    // keep aspect ratio
    fan::vec2 ratio = window_size / window_size.max();
    camera.set_ortho(
      &loco,
      ortho_x * ratio.x,
      ortho_y * ratio.y
    );
    viewport.set(loco.get_context(), 0, window_size, window_size);
      });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);

    // requires manual open with compiled texture pack name
  }

  loco_t::theme_t theme;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;

};

pile_t* pile = new pile_t;

#define loco_var pile->loco
#include _FAN_PATH(graphics/gui/model_maker/loader.h)

int main(int argc, char** argv) {
  if (argc < 2) {
    fan::throw_error("usage: TexturePackCompiled");
  }
  loco_t::texturepack_t tp;
  tp.open_compiled(&pile->loco, argv[1]);
  model_list_t m;

  cm_t cm;
  cm.import_from("model.fmm", &tp);

  model_list_t::properties_t p;
  p.camera = &pile->camera;
  p.viewport = &pile->viewport;

  auto model_id = m.push_model(&tp, &cm, p);
  
  uint32_t group_id = 0;

  m.iterate(model_id, group_id, [&]<typename T>(auto shape_id, const T& properties) {
    if constexpr (std::is_same_v<T, model_loader_t::mark_t>) {
      properties.group_id;
      switch (fan::get_hash(properties.id)) {
        
        case fan::get_hash("smoke_position"): {
          loco_t::rectangle_t::properties_t rp;
          rp.color = fan::colors::red;
          rp.camera = &pile->camera;
          rp.viewport = &pile->viewport;
          rp.position = properties.position;
          rp.size = 0.01;

          m.push_shape(model_id, group_id, rp);
          break;
        }
        default: {
          loco_t::sprite_t::properties_t p;
          p.camera = &pile->camera;
          p.viewport = &pile->viewport;
          p.position = properties.position;
          p.size = 0.1;
          loco_t::texturepack_t::ti_t ti;
          if (ti.qti(&tp, "tire")) {
            fan::throw_error("invalid textureapack name");
          }
          p.load_tp(&ti);
          m.push_shape(model_id, group_id, p);
          break;
        }
      }
    }
  });

  f32_t angle = 0;
  pile->loco.loop([&] {
    m.set_position(model_id, pile->loco.get_mouse_position(pile->viewport));
    m.set_angle(model_id, angle);
    angle += pile->loco.get_delta_time() * 2;
  });

}