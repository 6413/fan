module;

#include <fan/types/types.h>

#if defined(fan_gui)
  #include <fan/imgui/text_editor.h>
#endif

#include <regex>
#include <functional>
#include <cstring>

export module fan.console;

export import fan.types.color;

import fan.fmt;
export import fan.graphics.common_types;

#if defined(fan_gui)

export namespace fan {
  struct commands_t {
    using arg_t = std::vector<std::string>;

    struct command_t {
      std::vector<std::string> command_chain;
      std::string description;
      std::function<void(const arg_t&)> func;
    };

    std::unordered_map<std::string, fan::commands_t::command_t> func_table;

    struct command_errors_e {
      enum {
        success,
        function_not_found,
        invalid_args
      };
    };

    struct output_t {
      uint16_t highlight = fan::graphics::highlight_e::text;
      std::string text;
    };

    std::function<void(const output_t&)> output_cb = [](const auto&) {};
    std::function<void(const std::string&, const fan::color& color)> output_colored_cb = [](const std::string&, const fan::color& color) {};

    inline fan::commands_t::command_t& add(const std::string& cmd, auto func) {
      command_t command;
      command.func = func;
      command_t& obj = func_table[cmd];
      obj = command;
      return obj;
    }

    int call(const std::string& cmd) {
      std::size_t arg0_off = cmd.find(" ");
      if (arg0_off == std::string::npos) {
        arg0_off = cmd.size();
      }
      std::string arg0 = cmd.substr(0, arg0_off);
      auto found = func_table.find(arg0);
      if (found == func_table.end()) {
        commands_t::print_command_not_found(cmd);
        return command_errors_e::function_not_found;
      }
      std::string rest;
      if (arg0_off + 2 > cmd.size()) {
        rest = "";
      }
      else {
        rest = cmd.substr(arg0_off + 1);
      }
      if (found->second.command_chain.empty()) {
        found->second.func(fan::split_quoted(rest));
      }
      else {
        for (const auto& i : found->second.command_chain) {
          call(i);
        }
      }

      return command_errors_e::success;
    }

    int insert_to_command_chain(const commands_t::arg_t& args)
    {
      if (args[1].find(";") != std::string::npos) {
        auto& obj = func_table[args[0]];
        static std::regex pattern(";\\s+");
        auto removed_spaces_before_semicolon = std::regex_replace(args[1], pattern, ";");
        obj.command_chain = fan::split(removed_spaces_before_semicolon, ";");
        obj.func = [](auto) {};
        return true;
      }
      return false;
    }

    void print_invalid_arg_count() {
      output_t out;
      out.text = "invalid amount of arguments\n";
      out.highlight = fan::graphics::highlight_e::error;
      output_cb(out);
    }

    void print_command_not_found(const std::string& cmd) {
      output_t out;
      out.text = "\"" + cmd + "\"" " command not found\n";
      out.highlight = fan::graphics::highlight_e::error;
      output_cb(out);
    }


  };

  std::string append_args(const std::vector<std::string>& args, uint64_t offset = 0, uint64_t end = -1) {
    std::string ret;
    uint64_t n = end == (uint64_t)-1 ? args.size() : end > args.size() ? args.size() : end;
    for (uint64_t i = offset; i < n; ++i) {
      if (i == 0) {
        ret += args[i];
      }
      else {
        ret += " " + args[i];
      }
    }
    return ret;
  }

  struct console_t {

    #define BLL_set_SafeNext 1
    #define BLL_set_AreWeInsideStruct 1
    #define BLL_set_prefix frame_cb
    #include <fan/fan_bll_preset.h>
    #define BLL_set_Link 1
    #define BLL_set_type_node uint16_t
    #define BLL_set_NodeDataType std::function<void()>
    #include <BLL/BLL.h>

    using highlight_e = fan::graphics::highlight_e;

    void open() {
      static auto l = [&](const fan::commands_t::output_t& out) {
        editor.SetReadOnly(false);
        fan::color color = fan::graphics::highlight_color_table[out.highlight];
        editor.SetCursorPosition(TextEditor::Coordinates(editor.GetTotalLines(), 0));
        editor.MoveEnd();
        editor.InsertTextColored(out.text, color);
        editor.SetReadOnly(true);
        output_buffer.push_back(out.text);
      };
      commands.output_cb = l;

      static auto lc = [&](const std::string& text, const fan::color& color) {
        editor.SetReadOnly(false);
        editor.SetCursorPosition(TextEditor::Coordinates(editor.GetTotalLines(), 0));
        editor.MoveEnd();
        editor.InsertTextColored(text, color);
        editor.SetReadOnly(true);
        output_buffer.push_back(text);
      };
      commands.output_colored_cb = lc;

      TextEditor::LanguageDefinition lang = TextEditor::LanguageDefinition::CPlusPlus();
      // set your own known preprocessor symbols...
      static const char* ppnames[] = { "NULL" };
      // ... and their corresponding values
      static const char* ppvalues[] = {
        "#define NULL ((void*)0)",
      };

      for (std::size_t i = 0; i < sizeof(ppnames) / sizeof(ppnames[0]); ++i)
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
      //fan::color bg = fan::color::hex(palette[(int)TextEditor::PaletteIndex::Background]);
      //bg = bg * 2;
      //palette[(int)TextEditor::PaletteIndex::Background] = bg.to_u32();

      //palette[(int)TextEditor::PaletteIndex::LineNumber] = 0;
      editor.SetPalette(palette);
      editor.SetTabSize(2);
      editor.SetReadOnly(true);
      editor.SetShowWhitespaces(false);

      input = editor;

      input.SetReadOnly(false);
      //input.SetShowLineNumbers(false);
      palette[(int)TextEditor::PaletteIndex::Background] = TextEditor::GetDarkPalette()[(int)TextEditor::PaletteIndex::Background];
      input.SetPalette(palette);
    }

    void close() {
      frame_cbs.Clear();
    }

    void render() {
      possible_choices.clear();

      if (current_command.size() == 0) {
        current_command.resize(buffer_size);
      }

      {
        auto it = frame_cbs.GetNodeFirst();
        while (it != frame_cbs.dst) {
          frame_cbs.StartSafeNext(it);
          frame_cbs[it]();
          it = frame_cbs.EndSafeNext();
        }
      }

      ImGui::Begin("console", 0);

      ImGui::BeginChild("output_buffer", ImVec2(0, ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing() * 1), false);


      editor.Render("editor");

      ImGui::EndChild();

      if (current_command.size()) {

        for (const auto& i : commands.func_table) {
          const auto& command = i.first;
          std::size_t len = std::strlen(current_command.c_str());
          if (len && command.substr(0, len) == current_command.c_str()) {
            possible_choices.push_back(command.c_str());
          }
        }
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetCursorScreenPos().x, ImGui::GetCursorScreenPos().y - ImGui::GetFrameHeightWithSpacing() * possible_choices.size()));

        if (possible_choices.size()) {
          ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.24f, 0.27f, 0.28f, 0.8f));
          ImGui::BeginChild("command_hints_window",
            ImVec2(ImGui::GetWindowWidth() / 4,
              ImGui::GetFrameHeightWithSpacing() * possible_choices.size())
          );
          for (const auto& i : possible_choices) {
            ImGui::Text("%s", i.c_str());
          }
          ImGui::EndChild();
          ImGui::PopStyleColor();
        }
      }

      ImGui::BeginChild("input_text", ImVec2(0, ImGui::GetFrameHeightWithSpacing()), false);

      if (init_focus) {
        ImGui::SetNextWindowFocus();
        init_focus = false;
      }
      input.Render("input");

      current_command = input.GetText();
      current_command.pop_back();
      if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
        current_command.erase(std::remove(current_command.begin(), current_command.end(), '\n'), current_command.end());
        current_command += '\n';
    
        command_history.push_back(current_command.substr(0, current_command.size() - 1));
        output_buffer.push_back(current_command);
        if (input.IsFocused()) {
          editor.SetReadOnly(false);
          editor.InsertTextColored("> " + current_command, fan::color::hex(0x999999FF));
          editor.SetReadOnly(true);
          commands.call(current_command.substr(0, current_command.size() - 1));
        }
        history_pos = -1;
        input.SetText("");
        input.SetCursorPosition(TextEditor::Coordinates{ 0, 0});
      }
      if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, false)) {
        if (history_pos == -1) {
          if (command_history.size()) {
            history_pos = command_history.size() - 1;
          }
        }
        else {
          history_pos = (history_pos - 1) % command_history.size();
        }
        if (command_history.size() && history_pos != -1) {
          input.SetText(command_history[history_pos]);
          input.SetCursorPosition(TextEditor::Coordinates(0, command_history[history_pos].size()));
        }
      }
      if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, false)) {
        if (history_pos == -1) {
          if (command_history.size()) {
            history_pos = command_history.size() - 1;
          }
        }
        else {
          history_pos = (history_pos + 1) % command_history.size();
        }
        if (command_history.size() && history_pos != -1) {
          input.SetText(command_history[history_pos]);
          input.SetCursorPosition(TextEditor::Coordinates(0, command_history[history_pos].size()));
        }
      }
      if (ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
        if (possible_choices.size()) {
          input.SetText(possible_choices.front() + " ");
          input.SetCursorPosition(TextEditor::Coordinates(0, possible_choices.front().size() + 1));
        }
        else if (current_command == "\t") {
          input.SetText("");
        }
      }

      ImGui::EndChild();

      ImGui::End();
    }

    void print(const std::string& msg, int highlight) {
      commands_t::output_t out;
      out.text = msg;
      out.highlight = highlight;
      commands.output_cb(out);
      input.MoveEnd();
      //set_input_focus();
    }

    void println(const std::string& msg, int highlight) {
      print(msg + "\n", highlight);
    }

    void print_colored(const std::string& msg, const fan::color& color) {
      commands.output_colored_cb(msg, color);

      input.MoveEnd();
      //set_input_focus();
    }

    void println_colored(const std::string& msg, const fan::color& color) {
      print_colored(msg + "\n", color);
    }

    void erase_frame_process(frame_cb_t::nr_t& nr) {
        frame_cbs.unlrec(nr);
        nr.sic();
    }


    inline frame_cb_t::nr_t push_frame_process(auto func) {
      auto it = frame_cbs.NewNodeLast();
      frame_cbs[it] = func;
      return it;
    }

    std::vector<std::string> command_history;
    std::string current_command;
    int command_history_index = 0;
    static constexpr int buffer_size = 0xfff;
    int history_pos = -1;
    std::vector<std::string> possible_choices;
    std::vector<std::string> output_buffer;

    commands_t commands;
    TextEditor editor;
    TextEditor input;
    uint32_t transparency;
    bool init_focus = false;

    frame_cb_t frame_cbs;
  };
}
#endif