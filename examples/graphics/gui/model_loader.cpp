#include <fan/pch.h>

#include <fan/graphics/gui/model_maker/loader.h>

int main() {

  loco_t loco;

  fan::vec2 window_size = loco.window.get_size();
  loco.camera_set_ortho(
    loco.orthographic_camera.camera,
    fan::vec2(0, window_size.x),
    fan::vec2(0, window_size.y)
  );

  loco_t::texturepack_t tp;
  tp.open_compiled("texture_packs/TexturePack");
  model_list_t m;

  model_list_t::cm_t cm;
  cm.import_from("ship.json", &tp);

  model_list_t::properties_t p;
  p.position = fan::vec3(fan::vec2(0), 5);

  auto model_id = m.push_model(&tp, &cm, p);
  //
  uint32_t group_id = 0;

  m.iterate(model_id, group_id, [&]<typename T>(const T& group_data) {
    fan::print(typeid(T).name());
    // mark
    if constexpr (std::is_same_v<T, loco_t::sprite_t>) {
      //static constexpr fan::string str("smoke_position");
      switch (fan::get_hash(group_data.shape.id)) {
        case fan::get_hash("0"): {

          //m.push_shape(model_id, group_id, fan::graphics::rectangle_t{{
          //    .position = fan::vec3(*(fan::vec2*)&properties.position, 3),
          //    .size = 10,
          //    .color = fan::colors::red
          //}});
          break;
        }
        default: {
          /*typename loco_t::shapes_t::sprite_t::properties_t p;
          p.position = properties.position;
          p.size = 0.1;
          loco_t::texturepack_t::ti_t ti;
          if (ti.qti(&tp, "tire")) {
            fan::throw_error("invalid texturepack name");
          }
          p.load_tp(&ti);
          m.push_shape(model_id, group_id, p, properties);
          break;*/
        }
      }
    }
  });

  //f32_t angle = 0;

  //fan::graphics::rectangle_t r{{
  //    .position = fan::vec3(window_size / 2, 3),
  //    .size = 100,
  //    .color = fan::colors::red
  //  }};


 //   m.set_size(model_id, 0.3);

  loco.loop([&] {
    
    m.set_position(model_id, loco.get_mouse_position(loco.orthographic_camera.camera, loco.orthographic_camera.viewport));
    //m.set_angle(model_id, angle);
    //m.set_angle(model_id, 1, angle * 1.5);
    //angle += loco.get_delta_time() * 2;
  });

}