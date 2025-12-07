module;

#if defined(fan_gui)
#include <string>
#include <vector>
#include <sstream>
#include <ranges>
#endif

export module fan.graphics.gui.text_logger;

#if defined(fan_gui)
import fan.graphics.common_context;
import fan.types.color;
import fan.types.vector;
import fan.fmt;
import fan.time;
import fan.math;
import fan.graphics.gui.base;

export namespace fan::graphics::gui {
  struct text_logger_t {
    struct text_t {
      std::string text;
      fan::color color;
      fan::time::timer fade_time;
      bool is_static;
      bool needs_formatting = false;
      std::streamsize tab_width = 0;
      std::vector<std::string> raw_columns;

      text_t(const std::string& t, const fan::color& c, bool static_msg = false)
        : text(t), color(c), is_static(static_msg) {
        if (!is_static) {
          fade_time = fan::time::timer(4e+9, true);
        }
      }
    };

    std::vector<text_t> floating_texts;
    std::vector<text_t> static_texts;
    
    std::vector<text_t> pending_floating;
    std::vector<text_t> pending_static;
    
    std::vector<f32_t> max_column_widths;
    bool column_widths_dirty = true;

    void set_text_fade_time(f32_t seconds) {
      for (auto& msg : floating_texts) {
        if (!msg.is_static) {
          msg.fade_time = fan::time::timer(seconds * 1e+9, false);
        }
      }
    }
    void add_floating(const std::string& text, const fan::color& color = fan::colors::orange) {
      floating_texts.emplace_back(text, color, false);
    }
    void add_static(const std::string& text, const fan::color& color = fan::colors::orange) {
      static_texts.emplace_back(text, color, true);
    }
    void clear_static() {
      static_texts.clear();
    }
    
    void queue_floating_formatted(const std::string& raw_text, std::streamsize tab_width, const fan::color& color = fan::colors::orange) {
      text_t entry(raw_text, color, false);
      entry.needs_formatting = true;
      entry.tab_width = tab_width;
      
      std::istringstream iss(raw_text);
      std::string token;
      while (iss >> token) {
        entry.raw_columns.push_back(token);
      }
      
      pending_floating.push_back(entry);
      column_widths_dirty = true;
    }
    
    void queue_static_formatted(const std::string& raw_text, std::streamsize tab_width, const fan::color& color = fan::colors::orange) {
      text_t entry(raw_text, color, true);
      entry.needs_formatting = true;
      entry.tab_width = tab_width;
      
      std::istringstream iss(raw_text);
      std::string token;
      while (iss >> token) {
        entry.raw_columns.push_back(token);
      }
      
      pending_static.push_back(entry);
      column_widths_dirty = true;
    }

    void calculate_max_column_widths() {
      if (!column_widths_dirty) {
        return;
      }
      max_column_widths.clear();
      
      auto analyze = [&](const std::vector<text_t>& entries) {
        for (const auto& entry : entries) {
          if (!entry.needs_formatting) {
            continue;
          }
          while (max_column_widths.size() < entry.raw_columns.size()) {
            max_column_widths.push_back(0);
          }
          for (size_t i = 0; i < entry.raw_columns.size(); ++i) {
            fan::vec2 text_size = gui::calc_text_size(entry.raw_columns[i]);
            max_column_widths[i] = std::max(max_column_widths[i], text_size.x);
          }
        }
      };
      
      analyze(pending_floating);
      analyze(pending_static);
      column_widths_dirty = false;
    }
    /*
    std::string format_with_max_widths(const std::vector<std::string>& columns, std::streamsize tab_width) {
      std::ostringstream oss;
      fan::vec2 space_size = ImGui::CalcTextSize(" ");
      f32_t min_column_width = tab_width;

      for (size_t i = 0; i < columns.size(); ++i) {
        bool is_neg = false;
        size_t colon_pos = columns[i].find(':');
        if (colon_pos != std::string::npos && colon_pos + 1 < columns[i].size()) {
          is_neg = columns[i][colon_pos + 1] == '-';
        }

        if (i > 0) {
          oss << (is_neg ? "" : " ");
        }

        oss << columns[i];

        if (i < columns.size() - 1 && i < max_column_widths.size()) {
          fan::vec2 text_size = ImGui::CalcTextSize(columns[i].c_str());
          f32_t target_width = std::max(max_column_widths[i], min_column_width);
          f32_t padding_needed = target_width - text_size.x;
          if (padding_needed > 0) {
            int spaces = (int)(padding_needed / space_size.x);
            for (int j = 0; j < spaces; ++j) oss << " ";
          }
          oss << " ";
        }
      }
      return oss.str();
    }
    */
    // hardcoded
    std::string format_with_max_widths(const std::vector<std::string>& columns, std::streamsize tab_width) {
      std::ostringstream oss;
      fan::vec2 space_size = gui::calc_text_size(" ");
      f32_t min_column_width = tab_width;

      for (size_t i = 0; i < columns.size(); ++i) {
        std::string s = columns[i];

        size_t num_pos = std::string::npos;
        for (size_t p = 0; p < s.size(); ++p) {
          char c = s[p];
          if (c == '-' || (c >= '0' && c <= '9') || c == '.') { num_pos = p; break; }
        }
        if (num_pos != std::string::npos && s[num_pos] != '-') {
          s.insert(num_pos, " ");
        }

        if (i > 0) oss << " ";
        oss << s;

        if (i < columns.size() - 1 && i < max_column_widths.size()) {
          fan::vec2 text_size = gui::calc_text_size(s);
          f32_t target_width = std::max(max_column_widths[i], min_column_width);
          f32_t padding_needed = target_width - text_size.x;
          if (padding_needed > 0) {
            int spaces = (int)(padding_needed / space_size.x);
            for (int j = 0; j < spaces; ++j) oss << " ";
          }
          oss << " ";
        }
      }
      return oss.str();
    }


    void flush_pending() {
      calculate_max_column_widths();
      auto process = [&](std::vector<text_t>& pending, std::vector<text_t>& target) {
        for (auto& entry : pending) {
          if (entry.needs_formatting) {
            entry.text = format_with_max_widths(entry.raw_columns, entry.tab_width) + "\n";
            entry.needs_formatting = false;
          }
          target.push_back(entry);
        }
        pending.clear();
      };
      process(pending_floating, floating_texts);
      process(pending_static, static_texts);
    }

    // Text that is added (stacked) to bottom left and fades away after specified time
    //-------------------------------------Floating text-------------------------------------
    template <typename ...Args>
    void print(const Args&... args) {
      add_floating(fan::format_args(args...) + "\n");
    }
    template <typename ...Args>
    void print(const fan::color& color, const Args&... args) {
      add_floating(fan::format_args(args...) + "\n", color);
    }
    template <typename... args_t>
    void printf(std::string_view fmt, args_t&&... args) {
      add_floating(fan::format(fmt, std::forward<args_t>(args)...) + "\n");
    }
    template <typename... args_t>
    void printf(const fan::color& color, std::string_view fmt, args_t&&... args) {
      add_floating(fan::format(fmt, std::forward<args_t>(args)...) + "\n", color);
    }
    template <typename... args_t>
    void printft(std::streamsize tab_width, std::string_view fmt, args_t&&... args) {
      std::string formatted = fan::format(fmt, std::forward<args_t>(args)...);
      queue_floating_formatted(formatted, tab_width);
    }
    template <typename... args_t>
    void printft(std::streamsize tab_width, const fan::color& color, std::string_view fmt, args_t&&... args) {
      std::string formatted = fan::format(fmt, std::forward<args_t>(args)...);
      queue_floating_formatted(formatted, tab_width, color);
    }
    //-------------------------------------Floating text-------------------------------------

    // Text that is added (stacked) to bottom left and it never disappears
    //-------------------------------------Static text-------------------------------------
    template <typename ...Args>
    void print_static(const Args&... args) {
      add_static(fan::format_args(args...) + "\n");
    }
    template <typename ...Args>
    void print_static(const fan::color& color, const Args&... args) {
      add_static(fan::format_args(args...) + "\n", color);
    }
    template <typename... args_t>
    void printf_static(std::string_view fmt, args_t&&... args) {
      add_static(fan::format(fmt, std::forward<args_t>(args)...) + "\n");
    }
    template <typename... args_t>
    void printf_static(const fan::color& color, std::string_view fmt, args_t&&... args) {
      add_static(fan::format(fmt, std::forward<args_t>(args)...) + "\n", color);
    }
    template <typename... args_t>
    void printft_static(std::streamsize tab_width, std::string_view fmt, args_t&&... args) {
      std::string formatted = fan::format(fmt, std::forward<args_t>(args)...);
      queue_static_formatted(formatted, tab_width);
    }
    template <typename... args_t>
    void printft_static(std::streamsize tab_width, const fan::color& color, std::string_view fmt, args_t&&... args) {
      std::string formatted = fan::format(fmt, std::forward<args_t>(args)...);
      queue_static_formatted(formatted, tab_width, color);
    }
    void clear_static_text() {
      clear_static();
    }
    //-------------------------------------Static text-------------------------------------

    int render_text_with_background(gui::draw_list_t* draw_list, const std::string& text, const fan::color& color, fan::vec2 position) {
      std::vector<std::string> lines;
      f32_t max_width = 0;
      size_t pos = 0;
      while (pos < text.length()) {
        size_t newline_pos = text.find('\n', pos);
        if (newline_pos == std::string::npos) {
          newline_pos = text.length();
        }
        std::string line = text.substr(pos, newline_pos - pos);
        lines.push_back(line);
        if (!line.empty()) {
          fan::vec2 text_size = gui::calc_text_size(line);
          max_width = std::max(max_width, text_size.x);
        }
        pos = newline_pos + 1;
      }
      if (!lines.empty() && max_width > 0) {
        f32_t line_height = gui::get_text_line_height();
        f32_t total_height = lines.size() * line_height;
        fan::vec2 bg_min = position + fan::vec2(-5, -5);
        fan::vec2 bg_max = position + fan::vec2(max_width + 5, total_height + 5);
        draw_list->AddRectFilled(bg_min, bg_max, fan::color(0, 0, 0, (int)(244 * color.a)).get_imgui_color(), 3.0f);
        for (int i = 0; i < lines.size(); ++i) {
          if (!lines[i].empty()) {
            fan::vec2 line_pos = position + fan::vec2(0, i * line_height);
            draw_list->AddText(line_pos, color.get_imgui_color(), lines[i].c_str());
          }
        }
      }
      return lines.size();
    }
    void render() {
      flush_pending();

      auto* draw_list = gui::get_foreground_draw_list();
      fan::vec2 window_pos = gui::get_window_pos();
      fan::vec2 window_size = gui::get_window_size();
      f32_t line_height = gui::get_text_line_height();
      f32_t box_spacing = 12.0f;
      f32_t current_y = window_size.y - 40;

      for (const auto& msg : static_texts | std::ranges::views::reverse) {
        fan::vec2 position = window_pos + fan::vec2(20, current_y);
        position.y -= gui::calc_text_size(msg.text).y;
        int lines_count = render_text_with_background(draw_list, msg.text, msg.color, position);
        current_y -= lines_count * line_height + box_spacing;
      }

      for (auto it = floating_texts.rbegin(); it != floating_texts.rend();) {
        if (it->fade_time.finished()) {
          it = std::vector<decltype(floating_texts)::value_type>::reverse_iterator(
            floating_texts.erase((++it).base())
          );
          continue;
        }

        uint64_t elapsed = it->fade_time.elapsed();
        f32_t progress = (f64_t)elapsed / it->fade_time.duration();
        fan::color color = it->color;
        f32_t alpha = 1.0f;

        if (progress > 0.8f) {
          alpha = (1.0f - progress) / 0.2f;
        }

        color.a *= fan::math::clamp(alpha, 0.0f, 1.0f);

        fan::vec2 position = window_pos + fan::vec2(20, current_y);
        position.y -= gui::calc_text_size(it->text).y;

        int lines_count = render_text_with_background(draw_list, it->text, color, position);
        current_y -= lines_count * line_height + box_spacing;

        ++it;
      }
    }
  };

  // Text that is added (stacked) to bottom left and fades away after specified time
  //-------------------------------------Floating text-------------------------------------
  template <typename ...Args>
  void print(const Args&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->print(args...);
  }
  template <typename ...Args>
  void print(const fan::color& color, const Args&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->print(color, args...);
  }
  template <typename... args_t>
  void printf(std::string_view fmt, args_t&&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->printf(fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  void printf(const fan::color& color, std::string_view fmt, args_t&&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->printf(color, fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  void printft(std::streamsize tab_width, std::string_view fmt, args_t&&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->printft(tab_width, fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  void printft(std::streamsize tab_width, const fan::color& color, std::string_view fmt, args_t&&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->printft(tab_width, color, fmt, std::forward<args_t>(args)...);
  }
  template <typename ...args_t>
  void print_error(args_t&&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->print(fan::colors::red, std::forward<args_t>(args)...);
  }
  template <typename ...args_t>
  void print_warning(args_t&&... args) {
    ((text_logger_t*)((text_logger_t*)fan::graphics::g_render_context_handle.text_logger))->print(fan::colors::yellow, std::forward<args_t>(args)...);
  }
  template <typename ...args_t>
  void print_success(args_t&&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->print(fan::colors::green, std::forward<args_t>(args)...);
  }
  void set_text_fade_time(f32_t seconds);
  //-------------------------------------Floating text-------------------------------------

  // Text that is added (stacked) to bottom left and it never disappears
  //-------------------------------------Static text-------------------------------------
  template <typename ...Args>
  void print_static(const Args&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->print_static(args...);
  }
  template <typename ...Args>
  void print_static(const fan::color& color, const Args&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->print_static(color, args...);
  }
  template <typename... args_t>
  void printf_static(std::string_view fmt, args_t&&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->printf_static(fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  void printf_static(const fan::color& color, std::string_view fmt, args_t&&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->printf_static(color, fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  void printft_static(std::streamsize tab_width, std::string_view fmt, args_t&&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->printft_static(tab_width, fmt, std::forward<args_t>(args)...);
  }
  template <typename... args_t>
  void printft_static(std::streamsize tab_width, const fan::color& color, std::string_view fmt, args_t&&... args) {
    ((text_logger_t*)fan::graphics::g_render_context_handle.text_logger)->printft_static(tab_width, color, fmt, std::forward<args_t>(args)...);
  }
  void clear_static_text();
  //-------------------------------------Static text-------------------------------------
}

export namespace fan {
  void gprint(const auto& ...args) {
    fan::graphics::gui::print(args...);
  }
}
#endif