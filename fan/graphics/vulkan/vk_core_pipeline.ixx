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

#endif

export module fan.graphics.vulkan.core:pipeline;

#if defined(FAN_2D)

import std;

import :types;
import fan.types.vector;
import fan.graphics.common_context;

export namespace fan::vulkan {
  struct context_t;
  struct buffer_t;

  struct push_constants_t {
    std::uint32_t texture_id;
    std::uint32_t camera_id;
    std::uint32_t texture_id1 = 0;
    std::uint32_t texture_id2 = 0;
    std::uint32_t texture_id3 = 0;
    std::uint32_t pad0 = 0;
    std::uint32_t pad1 = 0;
    std::uint32_t pad2 = 0;
    fan::vec4 lighting_ambient = fan::vec4(1.f, 1.f, 1.f, 1.f);
  };

  struct descriptor_t {
    descriptor_t() = default;
    descriptor_t(const descriptor_t&) = delete;
    descriptor_t& operator=(const descriptor_t&) = delete;
    descriptor_t(descriptor_t&& other) noexcept;
    descriptor_t& operator=(descriptor_t&& other) noexcept;

    using properties_t = std::vector<fan::vulkan::write_descriptor_set_t>;
    void open(fan::vulkan::context_t& context, const properties_t& properties);
    void close(fan::vulkan::context_t& context);

    void update(
      fan::vulkan::context_t& context,
      std::uint32_t n,
      std::uint32_t begin = 0,
      std::uint32_t texture_n = max_textures,
      std::uint32_t texture_begin = 0
    );

    properties_t m_properties;
    VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptor_set[fan::vulkan::max_frames_in_flight];
    std::vector<VkDescriptorBufferInfo> m_buffer_infos;
    std::vector<VkWriteDescriptorSet> m_descriptor_writes;
  };

  struct descriptor_pool_t {
    void open(fan::vulkan::context_t& context);
    void close(fan::vulkan::context_t& context);

    operator VkDescriptorPool() const {
      return m_descriptor_pool;
    }

    VkDescriptorPool m_descriptor_pool;
  };

  struct pipeline_t {
    struct properties_t {
      std::vector<VkDescriptorSetLayout> descriptor_layouts;
      fan::graphics::shader_nr_t shader;
      std::uint32_t push_constants_size = 0;
      std::uint32_t subpass = 0;

      std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments;
      bool enable_depth_test = VK_TRUE;
      VkCompareOp depth_test_compare_op = VK_COMPARE_OP_LESS;
      VkPrimitiveTopology shape_type = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    };

    void open(fan::vulkan::context_t& context, const properties_t& p);
    void close(fan::vulkan::context_t& context);
#if defined(FAN_2D)
    fan::graphics::shader_nr_t shader_nr;
#endif

    VkPipelineLayout m_layout;
    VkShaderEXT m_shaders[2]; // [0] = Vertex, [1] = Fragment
    properties_t properties;
  };
}
#endif