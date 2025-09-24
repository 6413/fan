import fan;

using namespace fan::graphics;

static constexpr int dice_sides = 6;

fan_enum_string(
  choices_e,
  ones,
  twos,
  threes,
  fours,
  fives,
  sixes,
  one_pair,
  two_pairs,
  three_of_a_kind,
  four_of_a_kind,
  small_straight,
  large_straight,
  full_house,
  chance,
  yatz
);

void set_random_indices(std::array<int, 5>& indices, const std::array<bool, 5>& toggled) {
  for (auto [i, v] : fan::enumerate(indices)) {
    if (toggled[i] == true) {
      continue;
    }
    v = fan::random::value(0, 5);
  }
}

bool rolling = false;

fan::ev::task_t roll_dice(std::array<int, 5>& indices, const std::array<bool, 5>& toggled) {
  rolling = true;
  for (int rolls = 0; rolls < 5; ++rolls) {
    set_random_indices(indices, toggled);
    co_await fan::co_sleep(50);
  }
  rolling = false;
}

bool add_score(std::array<int, std::size(choices_e_strings)>& scores, choices_e choice, const std::array<int, 5>& dice_image_indices) {
  std::array<int, 5> dice = dice_image_indices;
  std::sort(std::begin(dice), std::end(dice));

  std::array<int, 6> duplicates = { 0 };
  for (int d : dice) {
    duplicates[d]++;
  }

  scores[choice] = 0;

  if (choice <= sixes) {
    scores[choice] = duplicates[choice] * (choice + 1);
    return true;
  }

  switch (choice) {
  case one_pair: {
    for (int i = 6; --i;) {
      if (duplicates[i] >= 2) {
        scores[choice] = (i + 1) * 2; 
        return true; 
      }
    }
    break;
  }
  case two_pairs: {
    int pairs = 0, sum = 0;
    for (int i = 6; --i;) {
      if (duplicates[i] >= 2) { 
        pairs++; 
        sum += (i + 1) * 2; 
        if (pairs == 2) {
          break; 
        }
      }
    }
    if (pairs == 2) {
      scores[choice] = sum; 
      return true;
    }
    break;
  }
  case three_of_a_kind: {
    for (int i = 6; --i; ) {
      if (duplicates[i] >= 3) {
        scores[choice] = (i + 1) * 3;
        return true;
      }
    }
    break;
  }
  case four_of_a_kind:
    for (int i = 6; --i; ) {
      if (duplicates[i] >= 4) {
        scores[choice] = (i + 1) * 4; 
        return true; 
      }
    }
    break;
  case small_straight: {
    if (duplicates[0] && duplicates[1] && duplicates[2] && duplicates[3] && duplicates[4]) {
      scores[choice] = 15;
      return true;
    }
    break;
  }
  case large_straight:
    if (duplicates[1] && duplicates[2] && duplicates[3] && duplicates[4] && duplicates[5]) {
      scores[choice] = 20;
      return true;
    }
    break;
  case full_house: {
    bool has3 = false, has2 = false;
    for (int i = 0; i < 6; i++) { 
      if (duplicates[i] == 3) {
        has3 = true; 
      } 
      else if (duplicates[i] == 2) {
        has2 = true;
      }
    }
    if (has3 && has2) { 
      scores[choice] = std::accumulate(dice.begin(), dice.end(), 5); 
      return true; 
    }
    break;
  }
  case chance: {
    scores[choice] = std::accumulate(dice.begin(), dice.end(), 5);
    return true;
  }
  case yatz: {
    if (std::all_of(dice.begin() + 1, dice.end(), [first = dice[0]](int d) { return d == first; })) {
      scores[choice] = 50;
      return true;
    }
    break;
  }
  }
  scores[choice] = -1;
  return false;
}

int main() {
  engine_t engine;

  fan::ev::task_t task_roll;

  static constexpr const char* image_paths[] = {
    "images/dice_1.png", "images/dice_2.png",
    "images/dice_3.png", "images/dice_4.png",
    "images/dice_5.png", "images/dice_6.png" 
  };

  image_t images[dice_sides]{};

  std::array<int, 5> dice_image_indices{};
  std::array<bool, 5> image_toggled{};
  set_random_indices(dice_image_indices, image_toggled);

  for (int i = 0; i < std::size(images); i++) {
    images[i] = engine.image_load(image_paths[i]);
  }

  int rolls = 2;

  std::array<int, std::size(choices_e_strings)> scores;
  scores.fill(-1);

  fan_window_loop{

    ImGui::Begin("main_window");
    
    ImGui::Columns(2);
    if (ImGui::BeginChild("dice")) {
      fan::vec2 dice_size = 64;
      f32_t pad = 64.f;

      ImGui::NewLine();
      ImGui::NewLine();
      ImGui::Indent(128.f);
      {
        ImGui::ToggleImageButton("button0", images[dice_image_indices[0]], dice_size, &image_toggled[0]);
        ImGui::SameLine(0, pad);
        ImGui::ToggleImageButton("button1", images[dice_image_indices[1]], dice_size, &image_toggled[1]);
        ImGui::SameLine(0, pad);
        ImGui::ToggleImageButton("button2", images[dice_image_indices[2]], dice_size, &image_toggled[2]);
      }
      ImGui::NewLine();
      {
        ImGui::Indent(128.f / 2);
        ImGui::ToggleImageButton("button3", images[dice_image_indices[3]], dice_size, &image_toggled[3]);
        ImGui::SameLine(0, pad);
        ImGui::ToggleImageButton("button4", images[dice_image_indices[4]], dice_size, &image_toggled[4]);
        ImGui::Unindent();
      }
      ImGui::Unindent();

      ImGui::NewLine();
      ImGui::NewLine();
      ImGui::NewLine();

      ImGui::Indent(128.f / 2.5);
      if (ImGui::Button(("Roll (" + std::to_string(rolls) + ")").c_str())) {
        if (rolling == false && rolls > 0) {
          task_roll = roll_dice(dice_image_indices, image_toggled);
          --rolls;
        }
      }
      ImGui::Unindent();

    }
    ImGui::EndChild();
    ImGui::NextColumn();
    if (ImGui::BeginChild("choices")) {
      for (uint32_t i = 0; i < std::size(choices_e_strings); ++i) {
        if (ImGui::Button(choices_e_strings[i]) && rolls == 0) {
          if (add_score(scores, (choices_e)i, dice_image_indices)) {
            image_toggled.fill(0);
            rolls = 3;
          }
        }
        if (scores[i] == 0) {
          ImVec2 min_pos = ImGui::GetItemRectMin();
          ImVec2 max_pos = ImGui::GetItemRectMax();
          ImVec2 text_size = ImGui::CalcTextSize(choices_e_strings[i]);

          float text_start_x = min_pos.x + (max_pos.x - min_pos.x - text_size.x) * 0.5f;
          float text_y = min_pos.y + (max_pos.y - min_pos.y) * 0.5f;

          ImDrawList* draw_list = ImGui::GetWindowDrawList();
          draw_list->AddLine(
            ImVec2(text_start_x, text_y),
            ImVec2(text_start_x + text_size.x, text_y),
            IM_COL32(200, 0, 0, 255),
            2.0f
          );
        }
        if (scores[i] == -1) {
          ImGui::SameLine();
          ImGui::PushID(i);
          if (ImGui::Button("0")) {
            scores[i] = 0;
          }
          ImGui::PopID();
        }
        if (scores[i] > 0) {
          ImGui::SameLine(); 
          ImGui::Text("%d", scores[i]);
        }
      }
    }
    ImGui::EndChild();

    ImGui::End();
  };
}