module;

#include <fan/utility.h>

#if defined(fan_gui)
  #include <fan/imgui/text_editor.h>
#endif

#include <regex>
#include <functional>
#include <cstring>

module fan.console;

import fan.types.color;
import fan.types.fstring;
import fan.utility;
import fan.graphics.common_types;

#if defined(fan_gui)

namespace fan {

  void commands_t::empty_output(const output_t&) {}

  void commands_t::empty_output_colored(const std::string&, const fan::color& color) {}

  fan::commands_t::command_t& commands_t::add(const std::string& cmd, decltype(command_t::func) func) {
    command_t command;
    command.func = func;
    command_t& obj = func_table[cmd];
    obj = command;
    return obj;
  }

  void commands_t::remove(const std::string& cmd) {
    func_table.erase(cmd);
  }

  std::vector<std::string> commands_t::split_args(const std::string& input) {
    std::vector<std::string> args;
    std::string current;
    bool in_braces = false;
    bool in_quotes = false;

    for (char c : input) {
      if (c == '"' && !in_braces) {
        in_quotes = !in_quotes;
        continue;
      }

      if (!in_quotes) {
        if (c == '{') {
          in_braces = true;
          continue;
        }
        else if (c == '}') {
          in_braces = false;
          continue;
        }
        else if ((c == ',' || std::isspace(c)) && !in_braces) {
          if (!current.empty()) {
            args.push_back(current);
            current.clear();
          }
          continue;
        }
      }

      current += c;
    }

    if (!current.empty())
      args.push_back(current);

    return args;
  }

  int commands_t::call(const std::string& cmd) {
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
      found->second.func(OFFSETLESS(this, console_t, commands), split_args(rest));
    }
    else {
      for (const auto& i : found->second.command_chain) {
        call(i);
      }
    }

    return command_errors_e::success;
  }

  int commands_t::insert_to_command_chain(const commands_t::arg_t& args)
  {
    if (args[1].find(";") != std::string::npos) {
      auto& obj = func_table[args[0]];
      static std::regex pattern(";\\s+");
      auto removed_spaces_before_semicolon = std::regex_replace(args[1], pattern, ";");
      obj.command_chain = fan::split(removed_spaces_before_semicolon, ";");
      obj.func = [](console_t*, const commands_t::arg_t&) {};
      return true;
    }
    return false;
  }

  void commands_t::print_invalid_arg_count() {
    output_t out;
    out.text = "invalid amount of arguments\n";
    out.highlight = fan::graphics::highlight_e::error;
    output_cb(out);
  }

  void commands_t::print_command_not_found(const std::string& cmd) {
    output_t out;
    out.text = "\"" + cmd + "\"" " command not found\n";
    out.highlight = fan::graphics::highlight_e::error;
    output_cb(out);
  }

  std::string append_args(const std::vector<std::string>& args, uint64_t offset, uint64_t end) {
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

  void console_t::open() {
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
    static const char* ppnames[] = { "NULL" };
    static const char* ppvalues[] = {
      "#define NULL ((void*)0)",
    };

    for (std::size_t i = 0; i < sizeof(ppnames) / sizeof(ppnames[0]); ++i)
    {
      TextEditor::Identifier id;
      id.mDeclaration = ppvalues[i];
      lang.mPreprocIdentifiers.insert(std::make_pair(std::string(ppnames[i]), id));
    }

    editor.SetLanguageDefinition(lang);

    auto palette = editor.GetPalette();
    editor.SetPalette(palette);
    editor.SetTabSize(2);
    editor.SetReadOnly(true);
    editor.SetShowWhitespaces(false);

    input = editor;
    input.SetReadOnly(false);
    palette[(int)TextEditor::PaletteIndex::Background] = TextEditor::GetDarkPalette()[(int)TextEditor::PaletteIndex::Background];
    input.SetPalette(palette);
  }

  void console_t::close() {
    frame_cbs.Clear();
  }

  void console_t::render() {
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
        editor.InsertTextColored("> " + current_command, fan::color::from_rgba(0x999999FF));
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

  void console_t::print(const std::string& msg, int highlight) {
    commands_t::output_t out;
    out.text = msg;
    out.highlight = highlight;
    commands.output_cb(out);
    input.MoveEnd();
  }

  void console_t::println(const std::string& msg, int highlight) {
    print(msg + "\n", highlight);
  }

  void console_t::print_colored(const std::string& msg, const fan::color& color) {
    commands.output_colored_cb(msg, color);
    input.MoveEnd();
  }

  void console_t::println_colored(const std::string& msg, const fan::color& color) {
    print_colored(msg + "\n", color);
  }

  void console_t::erase_frame_process(frame_cb_t::nr_t& nr) {
    frame_cbs.unlrec(nr);
    nr.sic();
  }

  console_t::frame_cb_t::nr_t console_t::push_frame_process(std::function<void()> func) {
    auto it = frame_cbs.NewNodeLast();
    frame_cbs[it] = func;
    return it;
  }

}
#endif