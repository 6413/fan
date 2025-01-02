#include <fan/pch.h>
#include <fan/ev/ev.h>

f32_t guide_friendliness = 0.8f;
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

fan::ev::task_t chapter1_dialogue(fan::graphics::dialogue_box_t& db) {
  for (int i = 0; i < std::size(lore_chapter1_dialogue); ++i) {
    co_await db.text(lore_chapter1_dialogue[i]);
    if (i + 1 != std::size(lore_chapter1_dialogue)) {
      co_await db.wait_user_input();
    }
  } 
  while (db.get_button_choice() == -1) {
    db.button(lore_chapter1_answers[0], fan::vec2(0.8, 0.2));
    for (int i = 1; i < std::size(lore_chapter1_answers); ++i) {
      db.button(lore_chapter1_answers[i]);
    }
    co_await db.wait_user_input();
  }
  if (db.get_button_choice() == 0) {
    guide_friendliness += 0.1;
  }
  else {
    guide_friendliness = 0.2;
  }
}

int main() {
  loco_t loco;

  fan::graphics::dialogue_box_t dialogue_box;
  auto chapter1_task = chapter1_dialogue(dialogue_box);
  ImGui::GetStyle().WindowPadding = ImVec2(100.0f, 100.f);
  loco.loop([&] {
    fan::vec2 window_size = ImGui::GetWindowSize();
    window_size.x /= 1.2;
    window_size.y /= 3;
    dialogue_box.render("Dialogue box", loco.fonts[3], window_size, ImGui::GetWindowWidth() / 2, 32);
  });
}