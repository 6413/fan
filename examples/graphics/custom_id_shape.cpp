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

struct light_rectangle_t;
struct light_rectangle_fade_t;

#define loco_t_id_t_types light_rectangle_t*, light_rectangle_fade_t*

#define loco_light
#define loco_sprite
#define loco_button
#include _FAN_PATH(graphics/loco.h)

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
  }

  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};

pile_t* pile = new pile_t;

 #define sb_shape_name _light_rectangle_t
  #define sb_shader_fragment_string R"(
    #version 330

    layout (location = 1) out vec4 o_attachment1;

    in vec4 instance_color;
    in vec3 instance_position;
    in vec2 instance_size;
    in vec3 frag_position;

    void main() {
      o_attachment1 = instance_color;
    }
  )"
  #define sb_get_loco \
    loco_t* get_loco() { \
      return &pile->loco; \
    }
  #define sb_is_light
  #include _FAN_PATH(graphics/opengl/2D/objects/light.h)
  
  struct light_rectangle_t : _light_rectangle_t {
    struct properties_t : _light_rectangle_t::properties_t {
      using type_t = light_rectangle_t;
    };
    light_rectangle_t() {
      pile->loco.m_draw_queue_light.push_back([&] {
        draw();
      });
    }

    properties_t get_properties(loco_t::cid_t* cid) {
      auto p = _light_rectangle_t::get_properties(cid);
      return *(properties_t*)&p;
    }
  }light_rectangle;

  #define sb_shape_name _light_rectangle_fade_t
  #define sb_shader_fragment_string R"(
    #version 330

    layout (location = 1) out vec4 o_attachment1;

    in vec4 instance_color;
    in vec3 instance_position;
    in vec2 instance_size;
    in vec3 frag_position;

    void main() {
      o_attachment1 = instance_color;
    }
  )"
  #define sb_get_loco \
    loco_t* get_loco() { \
      return &pile->loco; \
    }
  #define sb_is_light
  #include _FAN_PATH(graphics/opengl/2D/objects/light.h)
  
  struct light_rectangle_fade_t : _light_rectangle_fade_t {
    struct properties_t : _light_rectangle_fade_t::properties_t {
      using type_t = light_rectangle_fade_t;
    };
    light_rectangle_fade_t() {
      pile->loco.m_draw_queue_light.push_back([&] {
        draw();
      });
    }

    properties_t get_properties(loco_t::cid_t* cid) {
      auto p = _light_rectangle_fade_t::get_properties(cid);
      return *(properties_t*)&p;
    }
  }light_rectangle_fade;

int main() {
  
  loco_t::sprite_t::properties_t p;

  p.size = fan::vec2(1);
  p.camera = &pile->camera;
  p.viewport = &pile->viewport;

  loco_t::image_t image;
  image.load(&pile->loco, "images/brick.webp");
  p.image = &image;
  p.position = fan::vec2(0, 0);
  loco_t::id_t id2 = p;

  light_rectangle_t::properties_t lp;
  lp.camera = &pile->camera;
  lp.viewport = &pile->viewport;
  lp.position = 0;
  lp.size = 0.5;
  lp.color = fan::color(0, 0, 1, 1) * 5;
  lp.type = 0;

  loco_t::id_t id = lp;
  id.set_position(fan::vec2());
  pile->loco.loop([&] {

	});
}