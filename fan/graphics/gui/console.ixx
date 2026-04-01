module;

#if defined(FAN_GUI)
  #include <fan/utility.h>
#endif
#include <cstdint>
#include <string>
#include <vector>
#include <functional>

export module fan.console;

#if defined(FAN_GUI)
import fan.types.color;
import fan.types.fstring;
import fan.formatter;
import fan.graphics.common_types;

export namespace fan {
  struct console_t;

  struct commands_t {
    using arg_t = std::vector<std::string>;
    using cmd_cb_t = std::function<void(console_t*, const arg_t&)>;

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

    std::function<void(const output_t&)> output_cb;
    std::function<void(const std::string&, const fan::color& color)> output_colored_cb;

    void* internal_state = nullptr;

    commands_t();
    ~commands_t();

    struct command_proxy_t {
      std::string& description;
      command_proxy_t(std::string& desc) : description(desc) {}
    };

    command_proxy_t add(const std::string& cmd, cmd_cb_t func);
    void remove(const std::string& cmd);

    std::vector<std::string> split_args(const std::string& input);

    int call(const std::string& cmd);
    int call_args(const std::string& cmd, const std::vector<std::string>& args);

    template <typename... Args>
    int call(const std::string& cmd, Args&&... args) {
      return call_args(cmd, {fan::format_args_raw(std::forward<Args>(args))...});
    }

    int insert_to_command_chain(const arg_t& args);

    void print_invalid_arg_count();
    void print_command_not_found(const std::string& cmd);

    std::vector<std::pair<std::string, std::string>> get_command_list() const;
    bool has_command(const std::string& cmd) const;
    std::string get_command_description(const std::string& cmd) const;
  };

  std::string append_args(const std::vector<std::string>& args, uint64_t offset = 0, uint64_t end = -1);

  struct console_t {
    using frame_cb_nr_t = uint16_t;

    void* internal_state = nullptr;
    commands_t commands;
    uint32_t transparency = 255;
    bool init_focus = false;

    console_t();
    ~console_t();

    void open();
    void close();
    void render();
    void print(const std::string& msg, int highlight);
    void println(const std::string& msg, int highlight);
    void print_colored(const std::string& msg, const fan::color& color);
    void println_colored(const std::string& msg, const fan::color& color);

    void erase_frame_process(frame_cb_nr_t& nr);
    frame_cb_nr_t push_frame_process(std::function<void()> func);

    commands_t::command_proxy_t add_cmd(const std::string& cmd, commands_t::cmd_cb_t func) {
      return commands.add(cmd, func);
    }
    void remove_cmd(const std::string& cmd) {
      commands.remove(cmd);
    }

    int call(const std::string& cmd) {
      return commands.call(cmd);
    }
    template <typename... Args>
    int call(const std::string& cmd, Args&&... args) {
      return commands.call(cmd, std::forward<Args>(args)...);
    }
    void print_command_not_found(const std::string& cmd) {
      commands.print_command_not_found(cmd);
    }
    void print_invalid_arg_count() {
      commands.print_invalid_arg_count();
    }

    void force_focus();
    void clear();
  };
}
#endif