#pragma once

#if defined(loco_window)
	#include _FAN_PATH(window/window.h)
#endif
#include _FAN_PATH(physics/collision/rectangle.h)
#include _FAN_PATH(graphics/camera.h)
#include _FAN_PATH(types/masterpiece.h)
#include _FAN_PATH(graphics/webp.h)

#if defined(loco_opengl)
	#include _FAN_PATH(graphics/opengl/gl_graphics.h)
#elif defined(loco_vulkan)
	#include _FAN_PATH(graphics/vulkan/vk_graphics.h)
#endif

namespace fan {
  #define make_has_type(name) \
	  template <typename T, typename = int> \
    struct _has_##name##_t : std::false_type { }; \
	  template <typename T> \
    struct _has_##name##_t <T, decltype((void) T::name, 0)> : std::true_type { }; \
	  template <typename T> \
    concept has_##name##_t = _has_##name##_t<T>::value;

  make_has_type(camera);
  make_has_type(viewport);
  make_has_type(camera_id);
  make_has_type(texture_id);
  make_has_type(image);

	namespace graphics {

		#if defined(loco_opengl) && defined(loco_context)
			using fan::opengl::context_t;
			using fan::opengl::viewport_t;
			using fan::opengl::viewport_list_NodeReference_t;
			using fan::opengl::theme_list_NodeReference_t;
			using fan::opengl::cid_t;
			using fan::opengl::shader_t;
		#elif defined(loco_vulkan)
			using fan::vulkan::context_t;
			using fan::vulkan::viewport_t;
		#if defined(loco_window)
			using fan::vulkan::viewport_list_NodeReference_t;
			using fan::vulkan::theme_list_NodeReference_t;
		#endif
			using fan::vulkan::cid_t;
			using fan::vulkan::shader_t;
		#endif

		namespace core {
			#if defined(loco_opengl) && defined(loco_context)
				using fan::opengl::core::memory_write_queue_t;
				using fan::opengl::core::uniform_block_t;
			#elif defined(loco_vulkan)
				using fan::vulkan::core::memory_write_queue_t;
				using fan::vulkan::core::uniform_block_t;
			#endif
		}

	}
}