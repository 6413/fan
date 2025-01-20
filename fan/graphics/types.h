#pragma once

#include <fan/types/color.h>

namespace fan {
  void printclnn(auto&&... values);
  void printcl(auto&&... values);
  void printclnnh(int highlight, auto&&... values);
  void printclh(int highlight = 0, auto&&... values);

  namespace graphics {
    struct highlight_e {
      enum {
        text,
        error,
        success,
        info,
        warning
      };
    };

    static constexpr fan::color highlight_color_table[] = {
      fan::colors::white,
      fan::colors::red,
      fan::colors::green,
      fan::colors::orange,
      fan::colors::yellow,
    };
  }
}