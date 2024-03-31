#include fan_pch

int main() {
  loco_t loco;

  TextEditor editor, input;

  TextEditor::LanguageDefinition lang = TextEditor::LanguageDefinition::CPlusPlus();
  // set your own known preprocessor symbols...
  static const char* ppnames[] = { "NULL" };
  // ... and their corresponding values
  static const char* ppvalues[] = {
    "#define NULL ((void*)0)",
  };

  for (int i = 0; i < sizeof(ppnames) / sizeof(ppnames[0]); ++i)
  {
    TextEditor::Identifier id;
    id.mDeclaration = ppvalues[i];
    lang.mPreprocIdentifiers.insert(std::make_pair(std::string(ppnames[i]), id));
  }

  //for (auto& i : commands.func_table) {
  //  TextEditor::Identifier id;
  //  id.mDeclaration = i.second.description;
  //  lang.mIdentifiers.insert(std::make_pair(i.first, id));
  //}

  editor.SetLanguageDefinition(lang);
  //


  auto palette = editor.GetPalette();
  fan::color bg = palette[(int)TextEditor::PaletteIndex::Background];
  bg = bg * 2;
  palette[(int)TextEditor::PaletteIndex::Background] = bg.to_u32();

  //palette[(int)TextEditor::PaletteIndex::LineNumber] = 0;
  editor.SetPalette(palette);
  editor.SetTabSize(2);
  editor.SetShowWhitespaces(false);

  fan::string str;
  fan::io::file::read(
    _FAN_PATH_QUOTE(graphics/loco.h),
    &str
  );

  editor.SetText(str);
  
  int current_font = 2;

  bool block_zoom[2]{};

  float font_scale_factor = 1.0f;
  loco.window.add_buttons_callback([&](const auto& d) {
    if (d.state != fan::mouse_state::press) {
      return;
    }
    if (loco.window.key_pressed(fan::key_left_control) == false) {
      return;
    }

    auto& io = ImGui::GetIO();
    switch (d.button) {
      case fan::mouse_scroll_up: {
        if (block_zoom[0] == true) {
          break;
        }
        font_scale_factor *= 1.1;
        block_zoom[1] = false;
        break;
      }
      case fan::mouse_scroll_down: {
        if (block_zoom[1] == true) {
          break;
        }
        font_scale_factor *= 0.9;
        block_zoom[0] = false;
        break;
      }
    }

    //ImFont* selected_font = nullptr;
    //for (int i = 0; i < std::size(loco.fonts); ++i) {
    //  if (new_font_size <= font_size * (1 << i) / 2) {
    //    selected_font = loco.fonts[i];
    //    break;
    //  }
    //}

    if (font_scale_factor > 1.5) {
      current_font++;
      if (current_font > std::size(loco.fonts) - 1) {
        current_font = std::size(loco.fonts) - 1;
        block_zoom[0] = true;
      }
      else {
        io.FontDefault = loco.fonts[current_font];
        font_scale_factor = 1;
      }
    }

    if (font_scale_factor < 0.5) {
      current_font--;
      if (current_font < 0) {
        current_font = 0;
        block_zoom[1] = true;
      }
      else {
        io.FontDefault = loco.fonts[current_font];
        font_scale_factor = 1;
      }
    }
    

    // Set the window font scale
    io.FontGlobalScale = font_scale_factor;
    // Set the selected font for ImGui
    //io.FontDefault = selected_font;
    return;
  });

  loco.loop([&] {
    loco.get_fps();
    ImGui::Begin("window");
    editor.Render("editor");

    ImGui::End();

  });
}