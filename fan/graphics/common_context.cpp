#include "common_context.h"

//-----------------------------------opengl-----------------------------------
fan::graphics::context_functions_t fan::graphics::get_gl_context_functions() {
  return context_functions_t{
    #define context_renderer gl
    #define context_get (*reinterpret_cast<fan::opengl::context_t*>(&context.gl))
    #include "common_context_functions.h"
  };
}
//-----------------------------------opengl-----------------------------------

//-----------------------------------vulkan-----------------------------------
fan::graphics::context_functions_t fan::graphics::get_vk_context_functions() {
	return context_functions_t{
    #define context_renderer vk
    #define context_get (*reinterpret_cast<fan::vulkan::context_t*>(&context.vk))
    #include "common_context_functions.h"
	};
}
//-----------------------------------vulkan-----------------------------------