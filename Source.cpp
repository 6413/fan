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

#define loco_framebuffer


#define loco_no_inline

#define loco_rectangle
#define loco_sprite
#define loco_letter
#define loco_button
#define loco_text_box
#include _FAN_PATH(graphics/loco.h)

constexpr uint32_t count = 5000;

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    loco.open_matrices(
      &matrices,
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
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cids[count];
};

pile_t* pile = new pile_t;

#define loco_access &pile->loco
#include _FAN_PATH(graphics/loco_define.h)

struct a_t {
  loco_t::vfi_id_t vfiBaseID = loco_t::vfi_id_t(
    fan_init_struct(
      loco_t::vfi_t::properties_t,
      .mouse_button_cb = [Stage = this](const loco_t::vfi_t::mouse_button_data_t& data) {
        if (
          data.button == fan::input::mouse_scroll_up ||
          data.button == fan::input::mouse_scroll_down
          ) {
          /*f32_t a;
          if (data.button == fan::input::mouse_scroll_up) { a = -0.4; }
          if (data.button == fan::input::mouse_scroll_down) { a = +0.4; }
          Stage->WorldMatrixMultiplerVelocity += a;*/
        }
        else {
         /* if (data.button_state == fan::mouse_state::press) {
            data.flag->ignore_move_focus_check = true;
            data.vfi->set_focus_keyboard(data.vfi->get_focus_mouse());
          }
          if (data.button_state == fan::mouse_state::release) {
            data.flag->ignore_move_focus_check = false;
          }*/
        }
        return 0;
      },
      .mouse_move_cb = [](const loco_t::vfi_t::mouse_move_data_t& data) {return 0; },
        .keyboard_cb = [Stage = this](const loco_t::vfi_t::keyboard_data_t& data) {
        switch (data.key) {
        case fan::input::key_escape: {
          /* TODO
          if(data.state == fan::keyboard_state::press && !(game::pile->stage_data.sortie.ReservedFlags & 0x1)){
            game::pile->stage = game::stage_group::sortie_menu;
            // game::stage::sortie_menu::open(); TODO
          }
          else {
            game::pile->stage_data.sortie.ReservedFlags &= ~0x1;
          }
          */
          break;
        }
        case fan::input::key_left:
        case fan::input::key_right:
        case fan::input::key_up:
        case fan::input::key_down:
        case fan::input::key_a:
        case fan::input::key_d:
        case fan::input::key_w:
        case fan::input::key_s:
        {
         /* if (data.keyboard_state == fan::keyboard_state::press) {
            Stage->key_add(data.key);
          }
          else if (data.keyboard_state == fan::keyboard_state::release) {
            Stage->key_remove(data.key);
          }*/
          break;
        }
        }
        return 0;
      }

        )
  );
};

int main() {
  a_t a;
  loco_t::rectangle_id_t rectangle(
    fan_init_struct(
      loco_t::rectangle_id_t::properties_t,
      .position = fan::vec2(0, 0),
      .size = 0.1,
      .color = fan::colors::red,
      // compress this
      .matrices = &pile->matrices,
      .viewport = &pile->viewport
    )
  );

  fan_init_struct0(
    loco_t::rectangle_id_t::properties_t,
    r2,
    .position = fan::vec2(0, 0),
    .size = 0.1,
    .color = fan::colors::red,
    // compress this
    .matrices = &pile->matrices,
    .viewport = &pile->viewport
  );
  r2;

  fan_init_id_t0(
    loco_t::rectangle,
    r3,
    .position = fan::vec2(0, 0),
    .size = 0.1,
    .color = fan::colors::red,
    // compress this
    .matrices = &pile->matrices,
    .viewport = &pile->viewport
  );
  r3;

  pile->loco.set_vsync(false);

  pile->loco.loop([&] {

    pile->loco.get_fps();
  });

  return 0;
}