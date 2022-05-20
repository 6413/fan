builder_types.m_button_event.set_on_input(pile,
  [](
  fan::window_t* window,
  fan::opengl::context_t* context,
  uint32_t index,
  uint16_t key,
  fan::key_state key_state,
  fan_2d::graphics::gui::mouse_stage mouse_stage,
  void* user_ptr
) {

  if (key != fan::mouse_left) {
    return;
  }

  pile_t* pile = (pile_t*)user_ptr;

  switch (index) {

    #include <fan/graphics/gui/fgm/rectangle_sized_text_button/create.h>
    #include <fan/graphics/gui/fgm/text_renderer_clickable/create.h>
  }
});

pile->window.add_mouse_move_callback(pile,
  [](fan::window_t* window, const fan::vec2i& position, void* user_ptr) {
  pile_t* pile = (pile_t*)user_ptr;

  if (!(pile->editor_draw_types.flags & flags_t::moving)) {
    return;
  }

  switch (pile->editor_draw_types.selected_draw_type) {
    #include <fan/graphics/gui/fgm/rectangle_sized_text_button/move.h>
    #include <fan/graphics/gui/fgm/text_renderer_clickable/move.h>
  }
});

pile->window.add_keys_callback(pile,
  [](fan::window_t* window, uint16_t key, fan::key_state key_state, void* user_ptr)
{
  pile_t* pile = (pile_t*)user_ptr;

  switch (key) {
    #include <fan/graphics/gui/fgm/input/mouse_left.h>
    #include <fan/graphics/gui/fgm/input/mouse_scroll_up.h>
    #include <fan/graphics/gui/fgm/input/mouse_scroll_down.h>
    #include <fan/graphics/gui/fgm/input/key_delete.h>
  }
});