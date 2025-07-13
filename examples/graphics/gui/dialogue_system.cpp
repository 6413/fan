#include <fan/types/types.h>
#include <string>
#include <coroutine>

import fan;

#include <fan/graphics/types.h>

using namespace fan::graphics;

f32_t guide_reputation = 0.8f;
static inline std::string lore_chapter1_dialogue[] = {
  "Hello there!|",
  "Looks like you have woken up.|",
  "Hurry, there is no time to waste!|",
  "It appears that our exit gates from the cave has been locked.|",
  "I have tried searching everywhere, but I cant seem to find the master key...|",
  "Would you be willing to help searching the key?|"
};
int current_answer = 0;
static inline std::string lore_chapter1_answers[] = {
  "Yes, gladly!",
  "No."
};

fan::event::task_t chapter1_dialogue(fan::graphics::gui::dialogue_box_t& db) {
  for (int i = 0; i < std::size(lore_chapter1_dialogue); ++i) {

    co_await db.text_delayed(lore_chapter1_dialogue[i]);

    if (i + 1 != std::size(lore_chapter1_dialogue)) {
      co_await db.wait_user_input();
    }
  }
  while (db.get_button_choice() == -1) {
    db.button(lore_chapter1_answers[0], fan::vec2(0.8, 0.1), fan::vec2(128, 32));
    for (int i = 1; i < std::size(lore_chapter1_answers); ++i) {
      db.button(lore_chapter1_answers[i], fan::vec2(0.8, 0.4), fan::vec2(128, 32));
    }
    co_await db.wait_user_input();
  }
  if (db.get_button_choice() == 0) {
    guide_reputation += 0.1;
  }
  else {
    guide_reputation = 0.2;
  }
}

int main() {
  loco_t loco;

  fan::graphics::gui::dialogue_box_t dialogue_box;
  auto chapter1_task = chapter1_dialogue(dialogue_box);

  f32_t font_size = 11.f;
  
  fan_window_loop{

    if (fan::window::is_mouse_clicked(fan::mouse_scroll_up)) {
      font_size += 1.f;
    }
    else if (fan::window::is_mouse_clicked(fan::mouse_scroll_down)) {
      font_size -= 1.f;
      font_size = std::max(1.f, font_size);
    }

    // padding
    fan::vec2 window_size = gui::get_window_size();
    window_size.x /= 1.2;
    window_size.y /= 5;
    gui::push_style_color(gui::col_window_bg, fan::colors::transparent);
    gui::push_style_var(gui::style_var_window_border_size, 0.f);
    dialogue_box.font_size = font_size * 2.5;
    dialogue_box.render("Dialogue box", gui::get_font(dialogue_box.font_size), window_size, gui::get_window_size().x / 2, 32);
    gui::pop_style_var();
    gui::pop_style_color();
  };
}