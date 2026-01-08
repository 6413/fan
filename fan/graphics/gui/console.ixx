module;

#if defined(FAN_GUI)
  #include <fan/utility.h>
  #include <fan/imgui/text_editor.h>
#endif

#include <functional>
#include <vector>

export module fan.console;

#if defined(FAN_GUI)

export import fan.types.color;

import fan.types.fstring;
import fan.utility;
import fan.print;
export import fan.graphics.common_types;

export namespace fan {
  struct console_t;
  struct commands_t {
    using arg_t = std::vector<std::string>;

    struct command_t {
      std::vector<std::string> command_chain;
      std::string description;
      std::function<void(console_t* self, const arg_t&)> func;
    };

    std::unordered_map<std::string, fan::commands_t::command_t>& get_func_table() {
      static std::unordered_map<std::string, fan::commands_t::command_t> func_table;
      return func_table;
    }


    struct command_errors_e {
      enum {
        success,
        function_not_found,
        invalid_args
      };
    };

    struct output_t {
      std::uint16_t highlight = fan::graphics::highlight_e::text;
      std::string text;
    };

    static void empty_output(const output_t&);
    std::function<void(const output_t&)> output_cb = empty_output;

    static void empty_output_colored(const std::string&, const fan::color& color);
    std::function<void(const std::string&, const fan::color& color)> output_colored_cb = empty_output_colored;

    fan::commands_t::command_t& add(const std::string& cmd, decltype(command_t::func) func);
    void remove(const std::string& cmd);

    std::vector<std::string> split_args(const std::string& input);

    int call(const std::string& cmd);
    template <typename... Args>
    int call(const std::string& cmd, Args&&... args);

    int insert_to_command_chain(const commands_t::arg_t& args);

    void print_invalid_arg_count();
    void print_command_not_found(const std::string& cmd);
  };

  std::string append_args(const std::vector<std::string>& args, uint64_t offset = 0, uint64_t end = -1);

  struct console_t {
    #define BLL_set_SafeNext 1
    #define BLL_set_AreWeInsideStruct 1
    #define BLL_set_prefix frame_cb
    #include <fan/fan_bll_preset.h>
    #define BLL_set_Link 1
    #define BLL_set_type_node uint16_t
    #define BLL_set_NodeDataType std::function<void()>
    #include <BLL/BLL.h>

    void open();
    void close();
    void render();
    void print(const std::string& msg, int highlight);
    void println(const std::string& msg, int highlight);
    void print_colored(const std::string& msg, const fan::color& color);
    void println_colored(const std::string& msg, const fan::color& color);
    void erase_frame_process(frame_cb_t::nr_t& nr);
    frame_cb_t::nr_t push_frame_process(std::function<void()> func);

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

template <typename... Args>
int fan::commands_t::call(const std::string& cmd, Args&&... args) {
  auto found = get_func_table().find(cmd);
  if (found == get_func_table().end()) {
    commands_t::print_command_not_found(cmd);
    return command_errors_e::function_not_found;
  }

  std::vector<std::string> argv {
    fan::format_args_raw(std::forward<Args>(args))...
  };

  if (found->second.command_chain.empty()) {
    found->second.func(
      OFFSETLESS(this, fan::console_t, commands),
      argv
    );
  }
  else {
    for (const auto& i : found->second.command_chain) {
      call(i);
    }
  }

  return command_errors_e::success;
}

#endif