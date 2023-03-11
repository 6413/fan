// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan

//#define loco_wboit

#define loco_window
#define loco_context

//#define loco_post_process


#define loco_no_inline

#define loco_rectangle
#define loco_sprite
#define loco_letter
#define loco_button
#define loco_text_box
#define loco_light
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 5000;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      //viewport.set(loco.get_context(), 0, d.size, d.size);
      });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cids[count];
};

pile_t* pile = new pile_t;

#define loco_access &pile->loco
#include _FAN_PATH(graphics/loco_define.h)


int main() {

  loco_t::rectangle_id_t rectangle(
    fan_init_struct(
      loco_t::rectangle_id_t::properties_t,
      .position = fan::vec2(0, 0),
      .size = 0.1,
      .color = fan::colors::red,
      .camera = &pile->camera,
      .viewport = &pile->viewport
    )
  );

  loco_t::image_t image;
  image.load(&pile->loco, "images/brick.webp");

  loco_t::sprite_id_t sprite;

  sprite[{
      .position = fan::vec2(-0.5, 0),
      .size = 0.2,
      .image = &image,
      .camera = &pile->camera,
      .viewport = &pile->viewport
  }];

  loco_t::letter_id_t letter(
    fan_init_struct(
      loco_t::letter_id_t::properties_t,
      .camera = &pile->camera,
      .viewport = &pile->viewport,
      .position = fan::vec2(0.5, 0),
      .letter_id = 65,
      .font_size = 0.1
    )
  );

  loco_t::text_id_t text(
    fan_init_struct(
      loco_t::text_id_t::properties_t,
      .camera = &pile->camera,
      .viewport = &pile->viewport,
      .position = fan::vec2(0.5, 0.5),
      .text = "text",
      .font_size = 0.1
    )
  );

  loco_t::theme_t t;
  loco_t::theme_t theme = loco_t::themes::gray(0.5);
  theme.open(pile->loco.get_context());
  loco_t::button_id_t button(
    fan_init_struct(
      loco_t::button_id_t::properties_t,
      .camera = &pile->camera,
      .viewport = &pile->viewport,
      .position = fan::vec2(-0.5, 0.5),
      .size = fan::vec2(.3, .1),
      .text = "button",
      .theme = &theme,
    )
  );

  loco_t::text_box_id_t text_box(
    fan_init_struct(
      loco_t::text_box_id_t::properties_t,
      .camera = &pile->camera,
      .viewport = &pile->viewport,
      .position = fan::vec2(-0.5, -0.5),
      .size = fan::vec2(.3, .1),
      .text = "text box",
      .theme = &theme,
      )
  );

  loco_t::vfi_id_t vfi(
    fan_init_struct(
      loco_t::vfi_id_t::properties_t,
      .shape_type = loco_t::vfi_t::shape_t::rectangle,
      .shape.rectangle.position = fan::vec3(0.5, 0.5, 1),
      .shape.rectangle.size = pile->loco.text.get_text_size("text", 0.1),
      .shape.rectangle.size.x /= 2, // hitbox takes half size
      .shape.rectangle.camera = &pile->camera,
      .shape.rectangle.viewport = &pile->viewport,
      .mouse_button_cb = [](const loco_t::mouse_button_data_t& ii_d) -> int {
        fan::print("click rectangle");
        return 0;
      }
    )
  );

  loco_t::light_id_t light(
    fan_init_struct(
      loco_t::light_id_t::properties_t,
      .position = fan::vec2(0, 0),
      .size = 0.1,
      .color = fan::colors::green,
      // compress this
      .camera = &pile->camera,
      .viewport = &pile->viewport
    )
  );

  pile->loco.set_vsync(false);

  pile->loco.get_shape<loco_t::rectangle_t>();

  //pile->loco.push_shape(0, loco_t::rectangle_t::properties_t{});
  //pile->loco.erase_shape(rectangle);
  pile->loco.shape_set_position(rectangle, 0.5);
  //pile->loco.set_position(button, 0.5);

 /* loco_t::cid_t cc;
  pile->loco.rectangle.push_back(&cc, fan_init_struct(
    loco_t::rectangle_t::properties_t,
    .position = fan::vec2(0, 0),
    .size = 0.1,
    .color = fan::colors::red,
    .camera = &pile->camera,
    .viewport = &pile->viewport
  ));

  pile->loco.rectangle.erase(&cc);*/


  loco_t::id_t r2(
    fan_init_struct(
      loco_t::rectangle_t::properties_t,
      .position = fan::vec2(0, 0),
      .size = 0.1,
      .color = fan::colors::red,
      .camera = &pile->camera,
      .viewport = &pile->viewport
    )
  );

  r2 = fan_init_struct(
    loco_t::sprite_t::properties_t,
    .position = fan::vec2(0, 0),
    .size = 0.2,
    .image = &image,
    .camera = &pile->camera,
    .viewport = &pile->viewport
  );


  loco_t::id_t r3 = r2;
  
  //r3.set_position(0);

  r2.erase();

  pile->loco.loop([&] {
    //r3.set_position(pile->loco.get_mouse_position(pile->viewport));
    //fan::print(r3.get_position());
    pile->loco.get_fps();
  });

  return 0;
}