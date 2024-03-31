#pragma once

namespace fan {
  struct commands_t {
    using arg_t = std::vector<std::string>;

    struct command_t {
      std::vector<std::string> command_chain;
      std::string description;
      std::function<void(const arg_t&)> func;
    };

    std::unordered_map<std::string, command_t> func_table;

    struct command_errors_e {
      enum {
        success,
        function_not_found,
        invalid_args
      };
    };

    struct highlight_e {
      enum {
        text,
        error,
        success,
        info
      };
    };

    static constexpr fan::color highlight_color_table[] = {
      fan::colors::white,
      fan::colors::red,
      fan::colors::green,
      fan::colors::orange,
    };

    struct output_t {
      uint16_t highlight = highlight_e::text;
      fan::string text;
    };

    std::function<void(const output_t&)> output_cb = [](const auto&) {};

    command_t& register_command(const fan::string& cmd, auto func) {
      command_t command;
      command.func = func;
      command_t& obj = func_table[cmd];
      obj = command;
      return obj;
    }

    int call_command(const fan::string& cmd) {
      std::size_t arg0_off = cmd.find(" ");
      if (arg0_off == std::string::npos) {
        arg0_off = cmd.size();
      }
      fan::string arg0 = cmd.substr(0, arg0_off);
      auto found = func_table.find(arg0);
      if (found == func_table.end()) {
        commands_t::print_command_not_found(cmd);
        return command_errors_e::function_not_found;
      }
      fan::string rest;
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
          call_command(i);
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
      out.highlight = highlight_e::error;
      output_cb(out);
    }
    void print_command_not_found(const fan::string& cmd) {
      output_t out;
      out.text = "\"" + cmd + "\"" " command not found\n";
      out.highlight = highlight_e::error;
      output_cb(out);
    }
  };

  static std::string append_args(const std::vector<std::string>& args, uint64_t offset = 0, uint64_t end = -1) {
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

  static void create_console(commands_t& commands, TextEditor& editor, TextEditor& input) {
    static std::vector<std::string> command_history;
    static std::vector<std::string> output_buffer;
    static std::string current_command;
    static int command_history_index = 0;
    static constexpr int buffer_size = 0xfff;
    static int history_pos = -1;
    static std::vector<std::string> possible_choices;
    possible_choices.clear();

    if (current_command.size() == 0) {
      current_command.resize(buffer_size);
    }

    ImGui::Begin("console");

    ImGui::BeginChild("output_buffer", ImVec2(0, ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing() * 1), false);

    std::string output;
    for (std::size_t i = 0; i < output_buffer.size(); ++i) {
      output += output_buffer[i].c_str();
    }

    editor.Render("editor");

    //ImGui::InputTextMultiline("##input2", output.data(), output.size(), ImVec2(-1, -1), ImGuiInputTextFlags_ReadOnly);

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

    input.Render("input");

    current_command = input.GetText();
    current_command.pop_back();
    if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
      command_history.push_back(current_command.substr(0, current_command.size() - 1));
      output_buffer.push_back(current_command);
      editor.SetReadOnly(false);
      editor.InsertTextColored("> " + current_command, fan::color::hex(0x999999FF));
      editor.SetReadOnly(true);
      static auto l = [&](const fan::commands_t::output_t& out) {
        editor.SetReadOnly(false);
        fan::color color = fan::commands_t::highlight_color_table[out.highlight];
        editor.InsertTextColored(out.text, color);
        editor.SetReadOnly(true);
        output_buffer.push_back(out.text);
        };
      commands.output_cb = l;
      commands.call_command(current_command.substr(0, current_command.size() - 1));
      history_pos = -1;
      input.SetText("");
      input.InsertText("a");
      input.SetText("");
      //ImGui::SetWindowFocus("input");
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
        input.SetText(possible_choices.front());
        input.SetCursorPosition(TextEditor::Coordinates(0, possible_choices.front().size()));
      }
      else if (current_command == "\t") {
        input.SetText("");
      }
    }

    ImGui::EndChild();

    ImGui::End();
  }
}