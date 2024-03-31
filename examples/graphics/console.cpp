#include fan_pch


int main() {
  fan::commands_t commands;

  std::vector<loco_t::shape_t> shapes;

  commands.register_command("echo", [&](const fan::commands_t::arg_t& args) {
    commands.output_cb(fan::append_args(args) + "\n");
    }).description = "prints something - usage echo [args]";

  commands.register_command("help", [&](const fan::commands_t::arg_t& args) {
    if (args.empty()) {
      std::string out;
      out += "{\n";
      for (const auto& i : commands.func_table) {
        out += "\t" + i.first + ",\n";
      }
      out += "}";
      commands.output_cb(out + "\n");
      return;
    }
    else if (args.size() == 1) {
      auto found = commands.func_table.find(args[0]);
      if (found == commands.func_table.end()) {
        commands.print_command_not_found();
        return;
      }
      commands.output_cb(found->second.description + "\n");
    }
    else {
      commands.print_invalid_arg_count();
    }
    }).description = "get info about specific command - usage help command";

  commands.register_command("list", [&](const fan::commands_t::arg_t& args) {
    std::string out;
    for (const auto& i : commands.func_table) {
      out += i.first + "\n";
    }
    commands.output_cb(out);
    }).description = "lists all commands - usage list";

  commands.register_command("alias", [&](const fan::commands_t::arg_t& args) {
    if (args.size() < 2 || args[1].empty()) {
      commands.print_invalid_arg_count();
      return;
    }
    if (commands.insert_to_command_chain(args)) {
      return;
    }
    commands.func_table[args[0]] = commands.func_table[args[1]];
    }).description = "can create alias commands - usage alias [cmd name] [cmd]";

  commands.register_command("shape", [&](const fan::commands_t::arg_t& args) {
    switch (fan::get_hash(args[0])) {
      case fan::get_hash("spawn"): {
        if (args.size() != 7) {
          commands.print_invalid_arg_count();
          return;
        }
        switch (fan::get_hash(args[1])) {
          case fan::get_hash("rectangle"): {
            loco_t::shapes_t::rectangle_t::properties_t p;
            p.position.x = std::stof(args[2]);
            p.position.y = std::stof(args[3]);

            p.size.x = std::stof(args[4]);
            p.size.y = std::stof(args[5]);
            p.color = fan::color::hex(std::stoull(args[6], nullptr, 16));
            shapes.push_back(p);
            break;
          }
        }
        break;
      }
    }
    //commands.output_cb(append_args(args));
    }).description = "can create/modify shapes - usage shape \n\tspawn shape px py sx sy 0xffffffff";

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
  palette[(int)TextEditor::PaletteIndex::Background] = bg;

  //palette[(int)TextEditor::PaletteIndex::LineNumber] = 0;
  editor.SetPalette(palette);
  editor.SetTabSize(2);
  editor.SetReadOnly(true);
  editor.SetShowWhitespaces(false);

  input = editor;

  input.SetReadOnly(false);
  input.SetShowLineNumbers(false);
  palette[(int)TextEditor::PaletteIndex::Background] = TextEditor::GetDarkPalette()[(int)TextEditor::PaletteIndex::Background];
  input.SetPalette(palette);

  //editor.SetShowLineNumbers(false);


  loco.loop([&] {
    //  create_console(commands);

    loco.get_fps();

    fan::create_console(commands, editor, input);

    });
}