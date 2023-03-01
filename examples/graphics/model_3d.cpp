// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define fan_debug fan_debug_none

#define loco_window
#define loco_context

#define loco_model_3d
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  void open() {
    loco.open(loco_t::properties_t());
    fan::graphics::open_camera(
      loco.get_context(),
      &camera,
      ortho_x,
      ortho_y
    );
    /*loco.get_window()->add_resize_callback([&](fan::window_t*, const fan::vec2i& size) {
      viewport.set(loco.get_context(), 0, size, size);
    });*/
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  fan::opengl::camera_t camera;
  fan::opengl::viewport_t viewport;
  //fan::opengl::cid_t cids[count];
};

int main() {

  pile_t pile;

  pile.open();
  loco_t::model_t::properties_t mp;
  mp.get_camera() = &pile.camera;
  mp.get_viewport() = &pile.viewport;
  mp.loaded_model = pile.loco.model.load_model("models/test.obj");
  mp.position = fan::vec3(0, 0, 0);
  mp.size = 0.3;

  fan::vec3 camera_position = 0;
  mp.camera_position = &camera_position;
  mp.rotation_vector = fan::vec3(0, 1, 0);
  
  pile.loco.model.set(mp);
  
  fan::vec2ui window_size = pile.loco.get_window()->get_size();

  static constexpr f32_t zoom = 20;
  pile.camera.set_ortho(fan::vec2(-zoom, zoom), fan::vec2(zoom, -zoom));

  auto& window = *pile.loco.get_window();

  fan::graphics::animation::frame_transform_t origin;
  origin.position = 0;
  origin.size = 0.3;
  origin.angle = -fan::math::pi / 2;
  fan::graphics::animation::strive_t striver(origin);

  fan::graphics::animation::frame_transform_t dst;
  fan::graphics::animation::strive_t::properties_t p;
  p.time_to_destination = 1e+10;

  pile.loco.get_context()->set_vsync(&window, 0);

  bool left = false;

  bool done = true;

  //window.add_keys_callback([&](fan::window_t*, uint16_t key, fan::key_state key_state) {
  //  if (key_state != fan::key_state::press) {
  //    return;
  //  }

  //  switch (key) {
  //    case fan::key_left: {
  //      //dst.position = fan::vec3(0, 15, 0);
  //      dst.angle = -fan::math::pi;
  //      dst.size = 0.3;
  //      //dst.size = 1.5;
  //      p.src = pile.loco.model.get_keyframe();
  //      p.dst = dst;
  //      dst.position = p.src.position;
  //      striver.set(p);
  //      fan::print("left");
  //      left = true;
  //      done = false;
  //      break;
  //    }
  //    case fan::key_right: {
  //      //dst.position = fan::vec3(0, -15, 0);
  //      //dst.size = 0.5;
  //      dst.size = 0.3;
  //      dst.angle = 0;
  //      p.src = pile.loco.model.get_keyframe();
  //      dst.position = p.src.position;
  //      p.dst = dst;
  //      striver.set(p);
  //      fan::print("right");
  //      left = false;
  //      done = false;
  //      break;
  //    }
  //  }
  //  
  //});

  pile.loco.loop([&] {
    if (!(done)) {
      fan::graphics::animation::frame_transform_t ft = pile.loco.model.get_keyframe();
      auto gframe = striver.process(&window, ft, &done);
      pile.loco.model.set_keyframe(gframe);
      
    }
    else {
      auto p = pile.loco.model.get_position();
      if (left) {

        p.x -= 5 * window.get_delta_time();
      }
      else {
        p.x += 5 * window.get_delta_time();
      }
      pile.loco.model.set_position(p);
    }
  });

  return 0;
}