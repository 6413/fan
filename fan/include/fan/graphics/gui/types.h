#pragma once

#include <fan/types/color.h>

namespace fan_2d {
  namespace graphics {
    namespace gui {
      namespace defaults {
				inline fan::color text_color(1);
				inline fan::color text_color_place_holder = fan::color::hex(0x757575);
				inline f32_t font_size(32);
				constexpr f32_t text_renderer_outline_size = 0.6;
			}

      enum class text_position_e {
				left,
				middle
			};
			enum class button_states_e {
				clickable = 1,
				locked = 2
			};

			struct src_dst_t {
				fan::vec2 src = 0;
				fan::vec2 dst = 0;
			};
			namespace cursor_properties {
				inline fan::color color = fan::colors::white;
				// nanoseconds
				inline fan::time::nanoseconds blink_speed = 100000000;
				inline f32_t line_thickness = 1;
			}
    }
  }
}