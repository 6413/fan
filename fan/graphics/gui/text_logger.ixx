module;

export module fan.graphics.gui.text_logger;

#if defined (FAN_WINDOW)

import std;

#if defined(FAN_GUI)

import fan.types;
import fan.types.color;
import fan.types.vector;
import fan.time;
import fan.graphics.common_context;
import fan.formatter;
import fan.graphics.gui.types;

#if defined(FAN_FMT)
  import fan.fmt;
#endif

export namespace fan::graphics::gui {
  struct text_logger_t {
    struct text_t {
      std::string text;
      fan::color color;
      fan::time::timer fade_time{false};
      bool is_static;
      bool needs_formatting = false;
      std::int32_t tab_width = 0;
      std::vector<std::string> raw_columns;

      text_t(const std::string& t, const fan::color& c, bool static_msg = false);
    };

    std::vector<text_t> floating_texts;
    std::vector<text_t> static_texts;
    
    std::vector<text_t> pending_floating;
    std::vector<text_t> pending_static;
    
    std::vector<f32_t> max_column_widths;
    bool column_widths_dirty = true;

    void set_text_fade_time(f32_t seconds);
    void add_floating(const std::string& text, const fan::color& color = fan::colors::orange);
    void add_static(const std::string& text, const fan::color& color = fan::colors::orange);
    void clear_static();
    
    void queue_floating_formatted(const std::string& raw_text, std::streamsize tab_width, const fan::color& color = fan::colors::orange);
    void queue_static_formatted(const std::string& raw_text, std::streamsize tab_width, const fan::color& color = fan::colors::orange);

    void calculate_max_column_widths();
    std::string format_with_max_widths(const std::vector<std::string>& columns, std::streamsize tab_width);
    void flush_pending();

    template <typename ...Args>
    void print(const Args&... args) {
      add_floating(fan::format_args(args...) + "\n");
    }

    template <typename ...Args>
    void print(const fan::color& color, const Args&... args) {
      add_floating(fan::format_args(args...) + "\n", color);
    }

#if defined(FAN_FMT)
    template <typename... args_t>
    void printf(const std::string_view fmt, args_t&&... args) {
      add_floating(fan::format(fmt, std::forward<args_t>(args)...) + "\n");
    }

    template <typename... args_t>
    void printf(const fan::color& color, const std::string_view fmt, args_t&&... args) {
      add_floating(fan::format(fmt, std::forward<args_t>(args)...) + "\n", color);
    }
#endif

    template <typename ...Args>
    void print_static(const Args&... args) {
      add_static(fan::format_args(args...) + "\n");
    }

    template <typename ...Args>
    void print_static(const fan::color& color, const Args&... args) {
      add_static(fan::format_args(args...) + "\n", color);
    }

    void clear_static_text();

    int render_text_with_background(gui::draw_list_t* draw_list, const std::string& text, const fan::color& color, fan::vec2 position);
    void render();
  };

  template <typename ...Args>
  void print(const Args&... args) {
    ((text_logger_t*)fan::graphics::ctx().text_logger)->print(args...);
  }

  template <typename ...Args>
  void print(const fan::color& color, const Args&... args) {
    ((text_logger_t*)fan::graphics::ctx().text_logger)->print(color, args...);
  }

#if defined(FAN_FMT)
  template <typename... args_t>
  void printf(const std::string_view fmt, args_t&&... args) {
    ((text_logger_t*)fan::graphics::ctx().text_logger)->printf(fmt, std::forward<args_t>(args)...);
  }

  template <typename... args_t>
  void printf(const fan::color& color, const std::string_view fmt, args_t&&... args) {
    ((text_logger_t*)fan::graphics::ctx().text_logger)->printf(color, fmt, std::forward<args_t>(args)...);
  }
#endif

  template <typename ...args_t>
  void print_error(args_t&&... args) {
    ((text_logger_t*)fan::graphics::ctx().text_logger)->print(fan::colors::red, std::forward<args_t>(args)...);
  }

  template <typename ...args_t>
  void print_warning(args_t&&... args) {
    ((text_logger_t*)fan::graphics::ctx().text_logger)->print(fan::colors::yellow, std::forward<args_t>(args)...);
  }

  template <typename ...args_t>
  void print_success(args_t&&... args) {
    ((text_logger_t*)fan::graphics::ctx().text_logger)->print(fan::colors::green, std::forward<args_t>(args)...);
  }

  void set_text_fade_time(f32_t seconds);

  template <typename ...Args>
  void print_static(const Args&... args) {
    ((text_logger_t*)fan::graphics::ctx().text_logger)->print_static(args...);
  }

  template <typename ...Args>
  void print_static(const fan::color& color, const Args&... args) {
    ((text_logger_t*)fan::graphics::ctx().text_logger)->print_static(color, args...);
  }

  void clear_static_text();
}

export namespace fan {
  template <typename ...Args>
  void gprint(const Args&... args) {
    fan::graphics::gui::print(args...);
  }
}

#endif

#endif