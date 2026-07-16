module;

#if defined(FAN_2D)

#if defined(fan_platform_windows)
  #define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
  #define VK_USE_PLATFORM_XLIB_KHR
#endif
#include <vulkan/vulkan.h>
#include <fan/utility.h>

#endif

export module fan.graphics.vulkan.core:shader_subsystem;

#if defined(FAN_2D)

import std;
import :types;
import :uniform_block;
import fan.types;
import fan.types.vector;
import fan.types.fstring;
import fan.graphics.common_context;

export namespace fan::vulkan {
  struct context_t;

  struct shader_t {
    int projection_view[2]{ -1, -1 };
    uniform_block_t<fan::vulkan::view_projection_t, fan::vulkan::max_camera>* projection_view_block;
    VkPipelineShaderStageCreateInfo shader_stages[3]{};
    std::uint32_t compile_generation = 0;
  };

  struct shader_subsystem_t {
    context_t* ctx = nullptr;
    void init(context_t& context) { ctx = &context; }

    fan::vulkan::shader_t& shader_get(fan::graphics::shader_nr_t nr);

    static std::vector<std::uint32_t> compile_file(const std::string& source_name,
      int kind,
      const std::string& source);
    static std::vector<std::uint32_t> load_or_compile(const std::string& source_name,
      int kind,
      const std::string& source);
    fan::graphics::shader_nr_t shader_create();
    void shader_erase(fan::graphics::shader_nr_t nr, int recycle = 1);
    void shader_use(fan::graphics::shader_nr_t nr);
    VkShaderModule create_shader_module(const std::vector<std::uint32_t>& code);
    void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code);
    void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code);
    void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code);
    void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code);
    void shader_set_compute(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& compute_code);
    void shader_set_camera(fan::graphics::shader_nr_t nr, fan::graphics::camera_nr_t camera_nr);
    void shader_dispatch_compute(fan::graphics::shader_nr_t nr, std::uint32_t x, std::uint32_t y, std::uint32_t z);
    bool shader_compile(fan::graphics::shader_nr_t nr);
  };
}

#endif