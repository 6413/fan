module;

#if defined (FAN_WINDOW)

#include <cstdint>
  
#if defined(FAN_GUI)
  #include <fan/imgui/text_editor.h>
  #include <fan/imgui/imgui.h>
#endif

#include <fan/utility.h>

#endif

module fan.console;

#if defined (FAN_WINDOW)

import fan.memory;
import fan.types.color;
import fan.types.fstring;
import fan.utility;
import fan.graphics.common_types;
#if defined(FAN_GUI)
  import fan.graphics.gui.types;
  import fan.graphics.gui.base;
#endif

namespace fan {

  struct command_internal_t {
    std::vector<std::string> command_chain;
    std::string description;
    commands_t::cmd_cb_t func;
  };

  using cmd_table_t = std::unordered_map<std::string, command_internal_t>;

  commands_t::commands_t() {
    internal_state = new cmd_table_t();
    output_cb = [](const output_t&){};
    output_colored_cb = [](const std::string&, const fan::color&){};
  }

  commands_t::~commands_t() {
    delete static_cast<cmd_table_t*>(internal_state);
  }

  commands_t::command_proxy_t commands_t::add(const std::string& cmd, cmd_cb_t func) {
    auto& command = (*static_cast<cmd_table_t*>(internal_state))[cmd];
    command.func = func;
    return command_proxy_t(command.description);
  }

  void commands_t::remove(const std::string& cmd) {
    static_cast<cmd_table_t*>(internal_state)->erase(cmd);
  }

  std::vector<std::pair<std::string, std::string>> commands_t::get_command_list() const {
    std::vector<std::pair<std::string, std::string>> list;
    auto* table = static_cast<cmd_table_t*>(internal_state);
    for (const auto& [k, v] : *table) {
      list.push_back({k, v.description});
    }
    return list;
  }

  bool commands_t::has_command(const std::string& cmd) const {
    auto* table = static_cast<cmd_table_t*>(internal_state);
    return table->find(cmd) != table->end();
  }

  std::string commands_t::get_command_description(const std::string& cmd) const {
    auto* table = static_cast<cmd_table_t*>(internal_state);
    auto it = table->find(cmd);
    if (it != table->end()) return it->second.description;
    return "";
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
    auto* table = static_cast<cmd_table_t*>(internal_state);
    auto found = table->find(arg0);
    
    if (found == table->end()) {
      print_command_not_found(cmd);
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

  int commands_t::call_args(const std::string& cmd, const std::vector<std::string>& args) {
    auto* table = static_cast<cmd_table_t*>(internal_state);
    auto found = table->find(cmd);
    
    if (found == table->end()) {
      print_command_not_found(cmd);
      return command_errors_e::function_not_found;
    }

    if (found->second.command_chain.empty()) {
      found->second.func(OFFSETLESS(this, console_t, commands), args);
    }
    else {
      for (const auto& i : found->second.command_chain) {
        call(i);
      }
    }

    return command_errors_e::success;
  }

  int commands_t::insert_to_command_chain(const commands_t::arg_t& args) {
    if (args[1].find(';') != std::string::npos) {
      auto& obj = (*static_cast<cmd_table_t*>(internal_state))[args[0]];
      std::string cleaned = args[1];

      std::size_t pos = 0;
      while ((pos = cleaned.find(" ;", pos)) != std::string::npos) {
        cleaned.erase(pos, 1);
      }

      obj.command_chain = fan::split(cleaned, ";");
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

  std::string append_args(const std::vector<std::string>& args, std::uint64_t offset, std::uint64_t end) {
    std::string ret;
    std::uint64_t n = end == (std::uint64_t)-1 ? args.size() : end > args.size() ? args.size() : end;
    for (std::uint64_t i = offset; i < n; ++i) {
      if (i == 0) {
        ret += args[i];
      }
      else {
        ret += " " + args[i];
      }
    }
    return ret;
  }

  struct console_internal_t {
    #define BLL_set_SafeNext 1
    #define BLL_set_AreWeInsideStruct 1
    #define BLL_set_prefix frame_cb
    #include <fan/fan_bll_preset.h>
    #define BLL_set_Link 1
    #define BLL_set_type_node std::uint16_t
    #define BLL_set_NodeDataType std::function<void()>
    #include <BLL/BLL.h>

    std::vector<std::string> command_history;
    std::string current_command;
    int command_history_index = 0;
    static constexpr int buffer_size = 0xfff;
    int history_pos = -1;
    std::vector<std::string> possible_choices;
    std::vector<std::string> output_buffer;

#if defined(FAN_GUI)
    TextEditor editor;
    TextEditor input;
#endif
    frame_cb_t frame_cbs;
  };

  console_t::console_t() {
    internal_state = new console_internal_t();
  }

  console_t::~console_t() {
    delete static_cast<console_internal_t*>(internal_state);
  }

  void console_t::open() {
    auto* state = static_cast<console_internal_t*>(internal_state);
    
    auto l = [state](const fan::commands_t::output_t& out) {
      #if defined(FAN_GUI)
        state->editor.SetReadOnly(false);
        fan::color color = fan::graphics::highlight_color_table[out.highlight];
        state->editor.SetCursorPosition(TextEditor::Coordinates(state->editor.GetTotalLines(), 0));
        state->editor.MoveEnd();
        state->editor.InsertTextColored(out.text, color);
        state->editor.SetReadOnly(true);
      #endif
      state->output_buffer.push_back(out.text);
    };
    commands.output_cb = l;

    auto lc = [state](const std::string& text, const fan::color& color) {
      #if defined(FAN_GUI)
        state->editor.SetReadOnly(false);
        state->editor.SetCursorPosition(TextEditor::Coordinates(state->editor.GetTotalLines(), 0));
        state->editor.MoveEnd();
        state->editor.InsertTextColored(text, color);
        state->editor.SetReadOnly(true);
      #endif
      state->output_buffer.push_back(text);
    };
    commands.output_colored_cb = lc;

    #if defined(FAN_GUI)
      TextEditor::LanguageDefinition lang = TextEditor::LanguageDefinition::CPlusPlus();
      static const char* ppnames[] = { "NULL" };
      static const char* ppvalues[] = { "#define NULL ((void*)0)" };

      for (std::size_t i = 0; i < sizeof(ppnames) / sizeof(ppnames[0]); ++i) {
        TextEditor::Identifier id;
        id.mDeclaration = ppvalues[i];
        lang.mPreprocIdentifiers.insert(std::make_pair(std::string(ppnames[i]), id));
      }

      state->editor.SetLanguageDefinition(lang);

      auto palette = state->editor.GetPalette();
      state->editor.SetPalette(palette);
      state->editor.SetTabSize(2);
      state->editor.SetReadOnly(true);
      state->editor.SetShowWhitespaces(false);

      state->input.SetLanguageDefinition(lang);
      state->input.SetPalette(palette);
      state->input.SetTabSize(2);
      state->input.SetReadOnly(false);
      state->input.SetShowWhitespaces(false);
      palette[(int)TextEditor::PaletteIndex::Background] = TextEditor::GetDarkPalette()[(int)TextEditor::PaletteIndex::Background];
      state->input.SetPalette(palette);

      state->editor.SetRenderCursor(false);
    #endif
  }

  void console_t::close() {
    auto* state = static_cast<console_internal_t*>(internal_state);
    state->frame_cbs.Clear();
  }

#if defined(FAN_GUI)
  void console_t::render() {
    auto* state = static_cast<console_internal_t*>(internal_state);
    
    state->possible_choices.clear();

    if (state->current_command.size() == 0) {
      state->current_command.resize(console_internal_t::buffer_size);
    }

    {
      auto it = state->frame_cbs.GetNodeFirst();
      while (it != state->frame_cbs.dst) {
        state->frame_cbs.StartSafeNext(it);
        state->frame_cbs[it]();
        it = state->frame_cbs.EndSafeNext();
      }
    }

    fan::graphics::gui::begin("console", 0, fan::graphics::gui::window_flags_topmost);

    ImGui::BeginChild("output_buffer", ImVec2(0, ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing() * 1), false);
    state->editor.Render("editor");
    ImGui::EndChild();

    if (state->current_command.size()) {
      auto* table = static_cast<cmd_table_t*>(commands.internal_state);
      for (const auto& i : *table) {
        const auto& command = i.first;
        std::size_t len = std::strlen(state->current_command.c_str());
        if (len && command.substr(0, len) == state->current_command.c_str()) {
          state->possible_choices.push_back(command.c_str());
        }
      }
      ImGui::SetNextWindowPos(ImVec2(ImGui::GetCursorScreenPos().x, ImGui::GetCursorScreenPos().y - ImGui::GetFrameHeightWithSpacing() * state->possible_choices.size()));

      if (state->possible_choices.size()) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.24f, 0.27f, 0.28f, 0.8f));
        ImGui::BeginChild("command_hints_window", ImVec2(ImGui::GetWindowWidth() / 4, ImGui::GetFrameHeightWithSpacing() * state->possible_choices.size()));
        for (const auto& i : state->possible_choices) {
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
    state->input.Render("input");

    if (state->input.IsFocused()) {
      fan::graphics::gui::force_want_io_for_frame() = true;
    }

    state->current_command = state->input.GetText();
    state->current_command.pop_back();
    
    if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
      state->current_command.erase(std::remove(state->current_command.begin(), state->current_command.end(), '\n'), state->current_command.end());
      state->current_command += '\n';

      state->command_history.push_back(state->current_command.substr(0, state->current_command.size() - 1));
      state->output_buffer.push_back(state->current_command);
      if (state->input.IsFocused()) {
        state->editor.SetReadOnly(false);
        state->editor.InsertTextColored("> " + state->current_command, fan::color::from_rgba(0x999999FF));
        state->editor.SetReadOnly(true);
        commands.call(state->current_command.substr(0, state->current_command.size() - 1));
      }
      state->history_pos = -1;
      state->input.SetText("");
      state->input.SetCursorPosition(TextEditor::Coordinates{ 0, 0});
    }
    
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, false)) {
      if (state->history_pos == -1) {
        if (state->command_history.size()) {
          state->history_pos = state->command_history.size() - 1;
        }
      }
      else {
        state->history_pos = (state->history_pos - 1) % state->command_history.size();
      }
      if (state->command_history.size() && state->history_pos != -1) {
        state->input.SetText(state->command_history[state->history_pos]);
        state->input.SetCursorPosition(TextEditor::Coordinates(0, state->command_history[state->history_pos].size()));
      }
    }
    
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, false)) {
      if (state->history_pos == -1) {
        if (state->command_history.size()) {
          state->history_pos = state->command_history.size() - 1;
        }
      }
      else {
        state->history_pos = (state->history_pos + 1) % state->command_history.size();
      }
      if (state->command_history.size() && state->history_pos != -1) {
        state->input.SetText(state->command_history[state->history_pos]);
        state->input.SetCursorPosition(TextEditor::Coordinates(0, state->command_history[state->history_pos].size()));
      }
    }
    
    if (ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
      if (state->possible_choices.size()) {
        state->input.SetText(state->possible_choices.front() + " ");
        state->input.SetCursorPosition(TextEditor::Coordinates(0, state->possible_choices.front().size() + 1));
      }
      else if (state->current_command == "\t") {
        state->input.SetText("");
      }
    }

    ImGui::EndChild();
    fan::graphics::gui::end();
  }
#endif

  void console_t::print(const std::string& msg, int highlight) {
    auto* state = static_cast<console_internal_t*>(internal_state);
    commands_t::output_t out;
    out.text = msg;
    out.highlight = highlight;
    commands.output_cb(out);
  #if defined(FAN_GUI)
    state->input.MoveEnd();
  #endif
  }

  void console_t::println(const std::string& msg, int highlight) {
    print(msg + "\n", highlight);
  }

  void console_t::print_colored(const std::string& msg, const fan::color& color) {
    auto* state = static_cast<console_internal_t*>(internal_state);
    commands.output_colored_cb(msg, color);
  #if defined(FAN_GUI)
    state->input.MoveEnd();
  #endif
  }

  void console_t::println_colored(const std::string& msg, const fan::color& color) {
    print_colored(msg + "\n", color);
  }

  void console_t::erase_frame_process(frame_cb_nr_t& nr) {
    auto* state = static_cast<console_internal_t*>(internal_state);
    decltype(state->frame_cbs)::nr_t nri;
    nri.gint() = nr;
    state->frame_cbs.unlrec(nri);
    nr = static_cast<frame_cb_nr_t>(-1);
  }

  console_t::frame_cb_nr_t console_t::push_frame_process(std::function<void()> func) {
    auto* state = static_cast<console_internal_t*>(internal_state);
    auto it = state->frame_cbs.NewNodeLast();
    state->frame_cbs[it] = func;
    return it.gint();
  }

  void console_t::force_focus() {
    #if defined(FAN_GUI)
    auto* state = static_cast<console_internal_t*>(internal_state);
    state->input.InsertText("a");
    state->input.SetText("");
    init_focus = true;
    state->input.IsFocused() = false;
    #endif
  }

  void console_t::clear() {
    auto* state = static_cast<console_internal_t*>(internal_state);
    state->output_buffer.clear();
    #if defined(FAN_GUI)
      state->editor.SetText("");
    #endif
  }
}

#endif