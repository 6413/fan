#pragma once

#include _FAN_PATH(types/types.h)

namespace fan_2d {
  namespace graphics {
    namespace gui {
      namespace focus {

				constexpr uint32_t no_focus = -1;

				struct properties_t {

					properties_t() {}
					properties_t(uint32_t x) : window_handle((void*)x), shape(0), i(x) {}
					properties_t(void* window_handle_, void* shape, uint32_t i) :
						window_handle(window_handle_), shape(shape), i(i) {}

					// focus window
					void* window_handle;
					// shape class, initialized with shape = this
					void* shape;
					// index of focus
					uint32_t i;
				};

				inline properties_t current_focus = { 0, 0, focus::no_focus };

				static bool operator==(const properties_t& focus, uint32_t focused) {
					return focus.i == focused;
				}

				static bool operator==(const properties_t& focus, const properties_t& focused) {
					return focus.window_handle == focused.window_handle && focus.shape == focused.shape;
				}

				static bool operator!=(const properties_t& focus, uint32_t focused) {
					return focus.i != focused;
				}

				static properties_t get_focus() {
					return current_focus;
				}

				static bool has_focus(const properties_t& focus) {
					return
						current_focus.i != fan::uninitialized &&
						current_focus.window_handle == focus.window_handle &&
						current_focus.shape == focus.shape &&
						current_focus.i == focus.i;
				}
				static bool has_no_focus(const properties_t& focus) {
					return current_focus.shape == nullptr ||
					current_focus.window_handle != focus.window_handle ||
					current_focus.i == fan_2d::graphics::gui::focus::no_focus ||
					current_focus.shape != focus.shape;
				}

				static void set_focus(const properties_t focus) {
					current_focus = focus;
				}
			}
    }
  }
}