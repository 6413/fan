module;

#if defined(fan_platform_windows)
  #define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
  #define VK_USE_PLATFORM_XLIB_KHR
#endif
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <fan/utility.h>

export module fan.graphics.vulkan.core:image;
import std;

import fan.types.vector;

export namespace fan::vulkan {
  struct image_format {
    static constexpr auto b8g8r8a8_unorm = VK_FORMAT_B8G8R8A8_UNORM;
    static constexpr auto r8b8g8a8_unorm = VK_FORMAT_R8G8B8A8_UNORM;
    static constexpr auto r8_unorm = VK_FORMAT_R8_UNORM;
    static constexpr auto r8_uint = VK_FORMAT_R8_UINT;
    static constexpr auto r8g8_unorm = VK_FORMAT_R8G8_UNORM;
    static constexpr auto r8g8b8_unorm = VK_FORMAT_R8G8B8_UNORM;
    static constexpr auto r8g8b8a8_srgb = VK_FORMAT_R8G8B8A8_SRGB;
    static constexpr auto d32_sfloat = VK_FORMAT_D32_SFLOAT;
    static constexpr auto b10_g11_r11_ufloat_pack32 = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
  };

  struct image_sampler_address_mode {
    static constexpr auto repeat = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    static constexpr auto mirrored_repeat = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    static constexpr auto clamp_to_edge = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    static constexpr auto clamp_to_border = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    static constexpr auto mirrored_clamp_to_edge = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
  };

  struct image_filter {
    static constexpr auto nearest = VK_FILTER_NEAREST;
    static constexpr auto linear = VK_FILTER_LINEAR;
  };

  struct image_load_properties_defaults {
    static constexpr VkSamplerAddressMode visual_output = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    static constexpr VkFormat format = image_format::r8b8g8a8_unorm;
    static constexpr VkFilter min_filter = image_filter::nearest;
    static constexpr VkFilter mag_filter = image_filter::nearest;
  };

  struct image_load_properties_t {
    VkSamplerAddressMode visual_output = image_load_properties_defaults::visual_output;
    std::uint8_t internal_format = 0;
    VkFormat format = image_load_properties_defaults::format;
    VkFilter min_filter = image_load_properties_defaults::min_filter;
    VkFilter mag_filter = image_load_properties_defaults::mag_filter;
  };

  struct primitive_topology_t {
    static constexpr std::uint32_t points = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    static constexpr std::uint32_t lines = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    static constexpr std::uint32_t line_strip = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    static constexpr std::uint32_t triangles = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    static constexpr std::uint32_t triangle_strip = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    static constexpr std::uint32_t triangle_fan = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    static constexpr std::uint32_t lines_with_adjacency = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
    static constexpr std::uint32_t line_strip_with_adjacency = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    static constexpr std::uint32_t triangles_with_adjacency = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
    static constexpr std::uint32_t triangle_strip_with_adjacency = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
  };

  struct buffer_t {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;

    operator VkBuffer() const { return buffer; }
    explicit operator bool() const { return buffer != VK_NULL_HANDLE; }
  };

  struct image_t {
    VkImage image_index = VK_NULL_HANDLE;
    VkImageView image_view = VK_NULL_HANDLE;
    VmaAllocation image_allocation = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VmaAllocation staging_allocation = VK_NULL_HANDLE;
    VkDeviceSize staging_size = 0;
    void* data = nullptr;
  #if defined(FAN_GUI)
    VkDescriptorSet gui_descriptor_set = VK_NULL_HANDLE;
    VkImageView gui_image_view = VK_NULL_HANDLE;
    VkSampler gui_sampler = VK_NULL_HANDLE;
  #endif
    bool owns_image = true;
    bool owns_image_view = true;
  };

  struct buffer_barrier_t {
    buffer_t* buffer = nullptr;
    VkAccessFlags src_access = 0;
    VkAccessFlags dst_access = 0;
    VkDeviceSize offset = 0;
    VkDeviceSize size = VK_WHOLE_SIZE;
  };
}
