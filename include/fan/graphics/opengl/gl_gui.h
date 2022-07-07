#pragma once

#include _FAN_PATH(types/types.h)

#include _FAN_PATH(graphics/renderer.h)

#if fan_renderer == fan_renderer_opengl

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(graphics/gui/themes.h)
#include _FAN_PATH(graphics/shared_gui.h)
#include _FAN_PATH(graphics/gui/be.h)

namespace fan_2d {
  namespace graphics {

    template <typename T>
    struct loco_t : T{

      // user defined
      template <typename type_t, typename instance_data_t>
      static void global_move_cb(type_t*, uint32_t src, uint32_t dst, instance_data_t*);

      struct properties_t {
        fan_2d::graphics::font_t* font;
      };

      #define do_x_if_exists(type_, name_) if constexpr (std::is_same<T2::type_t, \
        type_<T2::type_t::user_global_data_t, \
        T2::type_t::user_instance_data_t>>::value)

      #define do_for_all() \
        f(fan_2d::graphics::rectangle_t, rectangle); \
        f(fan_2d::graphics::sprite_t, sprite); \
        f(fan_2d::graphics::gui::rectangle_text_box_t, rectangle_text_box); \
        f(fan_2d::graphics::gui::rectangle_text_button_t, rectangle_text_button)

      #define create_shape(name_) \
        template<typename T2, typename = void> \
        struct CONCAT(CONCAT(name_, _exists), _) : std::false_type { }; \
        template<typename T2> \
        struct CONCAT(CONCAT(name_, _exists), _)<T2, std::void_t<typename T2::_name>> : std::true_type { }; \
        template<typename T2> \
        static constexpr bool (CONCAT(name_, _exists)) = CONCAT(CONCAT(name_, _exists), _)<T2>::value;


      create_shape(rectangle);
      create_shape(sprite);
      create_shape(rectangle_text_button);

      void open(const properties_t& p) {
        if constexpr (rectangle_exists<loco_t>) {
          T::rectangle.open(context);
        }
        if constexpr (rectangle_text_button_exists<loco_t>) {
          fan::print("test");
        }
        /*if constexpr (rectangle_exists<loco_t>)
          T::rectangle.open(context);*/

        //do_x_if_exists(fan_2d::graphics::sprite_t, sprite) T::sprite.open(context);
        //do_x_if_exists(fan_2d::graphics::rectangle_text_box_t, rectangle_text_box) T::rectangle_text_box.open(context);
        //do_x_if_exists(fan_2d::graphics::rectangle_text_button_t, rectangle_text_button) T::rectangle_text_button.open(context);
      }
      void close() {
        #define f(type_, name_) do_x_if_exists(type_, name_) T::name_.close(context)
       // do_for_all();
        #undef f
      }

      template <typename T2>
      uint32_t push_back(fan::opengl::context_t* context, const T2& p) {
        #define f(type_, name_) do_x_if_exists(type_, name_) return T::name_.push_back(context, p)
        do_for_all();
        #undef f
      }

      #undef do_x_if_exists

      fan::opengl::context_t context;
    };

  }
}

//
//namespace fan_2d {
//
//	namespace graphics {
//
//		namespace gui {
//
//			struct circle : public fan_2d::opengl::circle {
//
//				circle(fan::camera* camera);
//
//			};
//
//			//struct sprite : public fan_2d::opengl::sprite_t {
//
//			//	sprite(fan::camera* camera);
//			//	// scale with default is sprite size
//			//	sprite(fan::camera* camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0, f32_t transparency = 1);
//			//	sprite(fan::camera* camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size = 0, f32_t transparency = 1);
//
//			//};
//
//		}
//
//	}
//}
//
#endif