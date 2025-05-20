module;

#include <fan/types/pair.h>

export module fan:graphics.common_types;

import :types.color;
import :types.vector;

export namespace fan {
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

    inline constexpr fan::color highlight_color_table[] = {
      fan::colors::white,
      fan::colors::red,
      fan::colors::green,
      fan::colors::orange,
      fan::colors::yellow,
    };
  }
  using line = fan::pair_t<fan::vec2, fan::vec2>;
  using line3 = fan::pair_t<fan::vec3, fan::vec3>;
}