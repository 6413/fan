#include fan_pch

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
      ortho_x * ratio.x,
      ortho_y * ratio.y
    );
    viewport.set(0, window_size, window_size);
      });
    viewport.open();
    viewport.set(0, window_size, window_size);

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

  model_list_t::cm_t cm;
  cm.import_from("entity_ship.fmm", &tp);

  model_list_t::properties_t p;
  p.camera = &pile->camera;
  p.viewport = &pile->viewport;

  auto model_id = m.push_model(&tp, &cm, p);
  
  //m.se
  
  uint32_t group_id = 0;

  m.iterate(model_id, group_id, [&]<typename T>(auto shape_id, const T& properties) {
    if constexpr (std::is_same_v<T, model_loader_t::mark_t>) {
      //static constexpr fan::string str("smoke_position");
      switch (fan::get_hash(properties.id)) {
        
        case fan::get_hash("smoke_position"): {
          typename loco_t::rectangle_t::properties_t rp;
          rp.color = fan::colors::red;
          rp.camera = &pile->camera;
          rp.viewport = &pile->viewport;
          rp.position = properties.position;
          rp.size = 0.01;

          m.push_shape(model_id, group_id, rp, properties);
          break;
        }
        default: {
          typename loco_t::sprite_t::properties_t p;
          p.camera = &pile->camera;
          p.viewport = &pile->viewport;
          p.position = properties.position;
          p.size = 0.1;
          loco_t::texturepack_t::ti_t ti;
          if (ti.qti(&tp, "tire")) {
            fan::throw_error("invalid texturepack name");
          }
          p.load_tp(&ti);
          m.push_shape(model_id, group_id, p, properties);
          break;
        }
      }
    }
  });

  f32_t angle = 0;

  m.set_size(model_id, 0.3);

  pile->loco.loop([&] {
    m.set_position(model_id, pile->loco.get_mouse_position(pile->camera, pile->viewport));
    m.set_angle(model_id, angle);
    angle += pile->loco.get_delta_time() * 2;
  });

}