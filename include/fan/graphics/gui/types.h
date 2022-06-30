#pragma once

#include _FAN_PATH(types/color.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {
      namespace defaults {
				inline fan::color text_color(1);
				inline fan::color text_color_place_holder = fan::color::hex(0x757575);
				inline f32_t font_size(0.1);
				constexpr f32_t text_renderer_outline_size = 0.4;
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
				inline fan::time::nanoseconds blink_speed = 1e+8;
				inline f32_t line_thickness = 0.002;
			}
    }
  }
}