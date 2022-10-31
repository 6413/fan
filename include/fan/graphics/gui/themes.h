#include _FAN_PATH(types/types.h)

#include _FAN_PATH(types/color.h)

#include _FAN_PATH(graphics/gui/types.h)

namespace fan_2d {
	namespace graphics {
		namespace gui {

			struct theme_t {

				#if defined(loco_opengl)
					using context_t = fan::opengl::context_t;
					#define ns fan::opengl
				#elif defined(loco_vulkan)
					using context_t = fan::vulkan::context_t;
					#define ns fan::vulkan
				#endif

				theme_t() = default;
				void open(auto* context){
					theme_reference = context->theme_list.NewNode();
					context->theme_list[theme_reference].theme_id = this;
				}

				void close(auto* context){
					context->theme_list.Recycle(theme_reference);
				}

				//template <typename T>
				//theme operator+(T value) const {
				//	theme t;
				//	t.button.color = button.color + value;
				//	t.button.outline_color = button.outline_color + value;
				//	t.button.text_color = button.text_color + value;
				//	t.button.text_outline_color = button.text_outline_color + value;
				//	t.button.text_outline_size = button.text_outline_size;

				//	return t;
				//}
				//template <typename T>
				//theme operator-(T value) {
				//	return operator+(-value);
				//}

				template <typename T>
				theme_t operator*(T value) const {
					theme_t t;
					t.theme_reference = theme_reference;
					t.button.color = button.color.mult_no_alpha(value);
					t.button.outline_color = button.outline_color.mult_no_alpha(value);
					t.button.text_outline_color = button.text_outline_color.mult_no_alpha(value);
					t.button.text_color = button.text_color;
					t.button.text_outline_size = button.text_outline_size;
					t.button.outline_size = button.outline_size;

					return t;
				}
				template <typename T>
				theme_t operator/(T value) const {
					theme_t t;
					t.theme_reference = theme_reference;
					t.button.color = button.color / value;
					t.button.outline_color = button.outline_color / value;
					t.button.text_outline_color = button.text_outline_color / value;
					t.button.text_color = button.text_color;
					t.button.text_outline_size = button.text_outline_size;
					t.button.outline_size = button.outline_size;

					return t;
				}

				struct button {

					button() = default;

					enum class states_e {
						outside,
						hover,
						click
					};

					fan::color color;
					fan::color outline_color;
					fan::color text_color;
					fan::color text_outline_color;
					f32_t text_outline_size;

					f32_t outline_size;

				}button;

				ns::theme_list_NodeReference_t theme_reference;
			};

			using theme_ptr_t = fan::ptr_maker_t<theme_t>;

			namespace themes {

				struct empty : public fan_2d::graphics::gui::theme_t {
					empty() {

						button.color = fan::color(0, 0, 0);
						button.outline_color = fan::color(0, 0, 0);
						button.text_color = fan_2d::graphics::gui::defaults::text_color;
						button.text_outline_color = fan::colors::black;
						button.text_outline_size = fan_2d::graphics::gui::defaults::text_renderer_outline_size;
						button.outline_size = 0.1;
					}
				};	

				struct deep_blue : public fan_2d::graphics::gui::theme_t {
					deep_blue(f32_t intensity = 1) {
						button.color = fan::color(0, 0, 0.3) * intensity;
						button.outline_color = fan::color(0, 0, 0.5) * intensity;
						button.text_color = fan_2d::graphics::gui::defaults::text_color;
						button.text_outline_color = fan::colors::black * intensity;
						button.text_outline_size = fan_2d::graphics::gui::defaults::text_renderer_outline_size;
						button.outline_size = 0.1;
					}
				};	

				struct deep_red : public fan_2d::graphics::gui::theme_t {

					deep_red(f32_t intensity = 1) {

						button.color = fan::color(0.3, 0, 0) * intensity;
						button.outline_color = fan::color(0.5, 0, 0) * intensity;
						button.text_color = fan_2d::graphics::gui::defaults::text_color;
						button.text_outline_color = fan::colors::black;
						button.text_outline_size = fan_2d::graphics::gui::defaults::text_renderer_outline_size;
						button.outline_size = 0.1;
					}
				};

				struct white : public fan_2d::graphics::gui::theme_t {
					white() {

						button.color = fan::color(0.8, 0.8, 0.8);
						button.outline_color = fan::color(0.9, 0.9, 0.9);
						button.text_color = fan_2d::graphics::gui::defaults::text_color;
						button.text_outline_color = fan::colors::black;
						button.text_outline_size = fan_2d::graphics::gui::defaults::text_renderer_outline_size;
						button.outline_size = 0.1;
					}
				};	

				struct locked : public fan_2d::graphics::gui::theme_t {
					locked() {
						button.color = fan::color(0.2, 0.2, 0.2, 0.8);
						button.outline_color = fan::color(0.3, 0.3, 0.3, 0.8);
						button.text_color = fan_2d::graphics::gui::defaults::text_color;
						button.text_outline_color = fan::colors::black;
						button.text_outline_size = fan_2d::graphics::gui::defaults::text_renderer_outline_size;
						button.outline_size = 0.1;
					}
				};	

				struct hidden : public fan_2d::graphics::gui::theme_t {
					hidden() {
						button.color = fan::color(0.0, 0.0, 0.0, 0.3);
						button.outline_color = fan::color(0.0, 0.0, 0.0, 0.3);
						button.text_color = fan_2d::graphics::gui::defaults::text_color;
						button.text_outline_color = fan::colors::black;
						button.text_outline_size = fan_2d::graphics::gui::defaults::text_renderer_outline_size;
						button.outline_size = 0.1;
					}
				};

				struct transparent : public fan_2d::graphics::gui::theme_t {
					transparent(f32_t intensity = 0.3) {
						button.color = fan::color(intensity, intensity, intensity, intensity);
						button.outline_color = fan::color(intensity + 0.2, intensity + 0.2, intensity + 0.2, intensity + 0.2);

						button.text_color = fan::colors::white;
						button.text_outline_color = fan::colors::black;
						button.text_outline_size = fan_2d::graphics::gui::defaults::text_renderer_outline_size;
					}
				};

				struct gray : public fan_2d::graphics::gui::theme_t {
					gray(f32_t intensity = 1) {

						button.color = fan::color(0.2, 0.2, 0.2, 1) * intensity;
						button.outline_color = fan::color(0.3, 0.3, 0.3, 1) * intensity;
						button.text_color = fan_2d::graphics::gui::defaults::text_color;
						button.text_outline_color = fan::colors::black;
						button.text_outline_size = fan_2d::graphics::gui::defaults::text_renderer_outline_size;
						button.outline_size = 0.1;
					}
				};

				struct custom : public fan_2d::graphics::gui::theme_t {
					struct properties_t {
						fan::color color;
						fan::color outline_color;
					};

					custom(const properties_t& properties) {
						button.color = properties.color;
						button.outline_color = properties.outline_color;
						button.text_color = fan_2d::graphics::gui::defaults::text_color;
						button.text_outline_color = fan::colors::black;
						button.text_outline_size = fan_2d::graphics::gui::defaults::text_renderer_outline_size;
						button.outline_size = 2; // px
					}
				};
			}
		}
	}
}

#undef ns