module;

#if defined(fan_platform_windows)
  #define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
  #define VK_USE_PLATFORM_XLIB_KHR
#endif
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <fan/utility.h>

export module fan.graphics.vulkan.core:compute;
import std;

import :image;

export namespace fan::vulkan {
  struct context_t;

  struct compute_pipeline_t {
    struct binding_t {
      std::uint32_t binding = 0;
      VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      std::uint32_t descriptor_count = 1;
      VkShaderStageFlags stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
    };

    void open(fan::vulkan::context_t& context, const std::string& path, VkDeviceSize push_size, const std::vector<binding_t>& bindings);
    void close(fan::vulkan::context_t& context);
    void dispatch(fan::vulkan::context_t& context, VkCommandBuffer cmd, VkDescriptorSet descriptor_set, const void* push, std::uint32_t x, std::uint32_t y, std::uint32_t z) const;

    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDeviceSize push_size = 0;
  };

  struct compute_slot_ring_t {
    static constexpr std::uint32_t invalid_slot = (std::uint32_t)-1;

    struct buffer_properties_t {
      VkDeviceSize size = 0;
      VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      VkMemoryPropertyFlags memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
      VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bool map = true;
    };

    struct slot_t {
      std::vector<buffer_t> buffers;
      VkCommandBuffer command_buffer = VK_NULL_HANDLE;
      VkFence fence = VK_NULL_HANDLE;
      VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
      bool in_flight = false;
    };

    void open(fan::vulkan::context_t& context, std::uint32_t slot_count, VkDescriptorSetLayout descriptor_layout, const std::vector<buffer_properties_t>& buffer_properties);
    void close(fan::vulkan::context_t& context);
    std::uint32_t acquire() const;
    VkCommandBuffer begin(fan::vulkan::context_t& context, std::uint32_t slot_index);
    void submit(fan::vulkan::context_t& context, std::uint32_t slot_index);
    bool done(fan::vulkan::context_t& context, std::uint32_t slot_index) const;
    void set_idle(std::uint32_t slot_index);
    std::uint32_t free_slot_count() const;
    slot_t& get(std::uint32_t slot_index);
    const slot_t& get(std::uint32_t slot_index) const;

    std::vector<slot_t> slots;
    std::vector<buffer_properties_t> buffer_properties;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    std::uint32_t submit_slot = 0;
  };
}
