#pragma once

import fan.color;

namespace fan {
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