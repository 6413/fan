#pragma once

#include <fan/types/color.hpp>

namespace fan_2d {
	
	namespace graphics {

		namespace gui {

			namespace defaults {

				inline fan::color text_color(1);
				inline fan::color text_color_place_holder = fan::color::hex(0x757575);
				inline f32_t font_size(32);

			}

			struct theme {

				struct {

					fan::color color;
					fan::color outline_color;
					fan::color text_color;

					fan::color hover_color;
					fan::color hover_outline_color;

					fan::color click_color;
					fan::color click_outline_color;

					f32_t outline_thickness;

				}text_button;

				struct {

					fan::color color;
					fan::color text_color;

					fan::color hover_color;

					fan::color click_color;

					fan::color check_color;

				}checkbox;

			};

			namespace themes {

				struct deep_blue : public fan_2d::graphics::gui::theme {

					deep_blue() {

						text_button.color = fan::color(0, 0, 0.3);
						text_button.outline_color = fan::color(0, 0, 0.5);
						text_button.text_color = fan_2d::graphics::gui::defaults::text_color;
						text_button.outline_thickness = 2; // px
						
						text_button.hover_color = text_button.color + 0.1;
						text_button.hover_outline_color = text_button.outline_color + 0.1;
						
						text_button.click_color = text_button.hover_color + 0.05;
						text_button.click_outline_color = text_button.click_color + 0.05;

						checkbox.color = text_button.color;
						checkbox.text_color = text_button.text_color;
						checkbox.hover_color = text_button.hover_color;
						checkbox.click_color = text_button.click_color;
						checkbox.check_color = fan::color(1, 1, 1);

					}

				};	

				struct deep_red : public fan_2d::graphics::gui::theme {

					deep_red() {

						text_button.color = fan::color(0.3, 0, 0);
						text_button.outline_color = fan::color(0.5, 0, 0);
						text_button.text_color = fan_2d::graphics::gui::defaults::text_color;
						text_button.outline_thickness = 2; // px

						text_button.hover_color = text_button.color + 0.1;
						text_button.hover_outline_color = text_button.outline_color + 0.1;

						text_button.click_color = text_button.hover_color + 0.05;
						text_button.click_outline_color = text_button.click_color + 0.05;

						checkbox.color = text_button.color;
						checkbox.text_color = text_button.text_color;
						checkbox.hover_color = text_button.hover_color;
						checkbox.click_color = text_button.click_color;
						checkbox.check_color = fan::color(0.2, 0, 0);

					}

				};

			}

		}

	}

}