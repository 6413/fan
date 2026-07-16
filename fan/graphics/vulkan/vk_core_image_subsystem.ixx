module;
#if defined(FAN_2D)

#if defined(fan_platform_windows)
  #define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
  #define VK_USE_PLATFORM_XLIB_KHR
#endif
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <fan/utility.h>

export module fan.graphics.vulkan.core:image_subsystem;
import std;

import fan.types;
import fan.types.vector;
import fan.types.color;
import fan.types.compile_time_string;
import fan.graphics.common_context;
import :image;
import fan.graphics.image_load;

export namespace fan::vulkan {
  struct context_t;

  struct image_subsystem_t {
    context_t& ctx;

    void transition_image_layout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void transition_image_layout_cmd(VkCommandBuffer cmd, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copy_buffer_to_image(VkBuffer buffer, VkImage image, VkFormat format, const fan::vec2ui& size, VkDeviceSize buffer_offset = 0);
    void copy_buffer_to_image_cmd(VkCommandBuffer cmd, VkBuffer buffer, VkImage image, VkFormat format, const fan::vec2ui& size, VkDeviceSize buffer_offset = 0);
    void create_texture_sampler(VkSampler& sampler, const fan::vulkan::image_load_properties_t& lp);

    VkFormat get_format_from_channels(int channels);

    fan::graphics::image_nr_t image_create();
    std::uint64_t image_get_handle(fan::graphics::image_nr_t nr);
    fan::vulkan::image_t& image_get(fan::graphics::image_nr_t nr);
    void image_erase(fan::graphics::image_nr_t nr, int recycle = 1);
    void image_bind(fan::graphics::image_nr_t nr);
    void image_bind(fan::graphics::image_nr_t nr, std::uint32_t unit);
    void image_bind(fan::graphics::image_t nr, uint32_t unit, std::uint32_t access, std::uint32_t format);
    void image_unbind(fan::graphics::image_nr_t nr);
    fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_nr_t nr);
    void image_set_settings(fan::graphics::image_nr_t nr, const fan::vulkan::image_load_properties_t& p);
    void image_set_settings(const fan::vulkan::image_load_properties_t& p);
    fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info, const fan::vulkan::image_load_properties_t& p);
    fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info);
    fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_, const fan::vulkan::image_load_properties_t& p);
    fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_);
    fan::graphics::image_nr_t create_missing_texture();
    fan::graphics::image_nr_t create_transparent_texture();
    fan::graphics::image_nr_t request_image_load_async(fan::str_view_t path, const fan::vulkan::image_load_properties_t& p, std::function<void(const fan::graphics::decoded_image_payload_t&)> on_gpu_uploaded);
    void process_async_image_uploads();
    fan::graphics::image_nr_t image_load(fan::str_view_t path, const fan::vulkan::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());
    fan::graphics::image_nr_t image_load(fan::str_view_t path, const std::source_location& callers_path = std::source_location::current());
    void image_unload(fan::graphics::image_nr_t nr);
    void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::vulkan::image_load_properties_t& p);
    void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info);
    void image_reload(fan::graphics::image_nr_t nr, fan::str_view_t path, const fan::vulkan::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());
    void image_reload(fan::graphics::image_nr_t nr, fan::str_view_t path, const std::source_location& callers_path = std::source_location::current());
    fan::graphics::image_nr_t image_create(const fan::color& color, const fan::vulkan::image_load_properties_t& p);
    fan::graphics::image_nr_t image_create(const fan::color& color);
    fan::graphics::image_nr_t image_create(void* data, const fan::vec2ui& size, const fan::vulkan::image_load_properties_t& p);
    std::vector<std::uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, std::uint32_t format, fan::vec2 uvp, fan::vec2 uvs);
    fan::graphics::image_nr_t image_create_from_view(VkImageView view, VkImage image, fan::vec2ui size, VkFormat format);
  };
}

#endif