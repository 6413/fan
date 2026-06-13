module;

#if defined(FAN_VULKAN)
#if defined(fan_platform_windows)
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
#define VK_USE_PLATFORM_XLIB_KHR
#endif
#if defined(FAN_GUI)
  #include <fan/imgui/imgui_impl_vulkan.h>
#endif
#define loco_window
#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#if defined(fan_platform_windows)
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <Windows.h>
  #define GLFW_EXPOSE_NATIVE_WIN32
  #define GLFW_EXPOSE_NATIVE_WGL
  #define GLFW_NATIVE_INCLUDE_NONE
#endif
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#endif

#include <fan/utility.h>

export module fan.graphics.vulkan.core;

import std;

#if defined(FAN_VULKAN)

import fan.types.matrix;
import fan.types.fstring;
import fan.types.color;
import fan.types.compile_time_string;

#if defined(loco_window)
  import fan.window;
#endif

import fan.memory;

import fan.utility;
import fan.print;
import fan.print.error;
import fan.graphics.image_load;
import fan.graphics.common_context;

#define __fan_internal_camera_list (*fan::graphics::ctx().camera_list)
#define __fan_internal_shader_list (*fan::graphics::ctx().shader_list)
#define __fan_internal_image_list (*fan::graphics::ctx().image_list)
#define __fan_internal_viewport_list (*fan::graphics::ctx().viewport_list)

#if defined(fan_compiler_msvc)
  #pragma comment(lib, "vulkan-1.lib")
  #pragma comment(lib, "shaderc_combined_mt.lib")
#endif

#define ENABLE_RAYTRACING_DEPENDENCIES

extern const std::vector<const char*> validationLayers;

extern const std::vector<const char*> deviceExtensions;


VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);

struct queue_family_indices_t {
  std::optional<std::uint32_t> graphics_family;
#if defined(loco_window)
  std::optional<std::uint32_t> present_family;
#endif
  bool is_complete() {
    return graphics_family.has_value()
    #if defined(loco_window)
      && present_family.has_value()
    #endif
      ;
  }
};

struct swap_chain_support_details_t {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

struct uniform_buffer_object_t {
  alignas(16) fan::mat4 model;
  alignas(16) fan::mat4 view;
  alignas(16) fan::mat4 proj;
};

export namespace fan {
  namespace vulkan {
    struct context_t;
    void image_create(const fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
  }
}

namespace fan::graphics::format_converter {
  VkFormat global_to_vulkan_format(std::uintptr_t format);
  VkSamplerAddressMode global_to_vulkan_address_mode(std::uintptr_t mode);
  VkFilter global_to_vulkan_filter(std::uintptr_t filter);

  std::uint32_t vulkan_to_global_format(VkFormat format);
  std::uint32_t vulkan_to_global_address_mode(VkSamplerAddressMode mode);
  std::uint32_t vulkan_to_global_filter(VkFilter filter);

  auto image_global_to_vulkan(const fan::graphics::image_load_properties_t& p);

  fan::graphics::image_load_properties_t image_vulkan_to_global(const auto& p) {
    return fan::graphics::image_load_properties_t{
      .visual_output = vulkan_to_global_address_mode(p.visual_output),
      .format = vulkan_to_global_format(p.format),
      .min_filter = vulkan_to_global_filter(p.min_filter),
      .mag_filter = vulkan_to_global_filter(p.mag_filter),
    };
  }
}

constexpr static std::uint32_t get_image_multiplier(VkFormat format);

export namespace fan {
  namespace vulkan {
    struct view_projection_t {
      fan::mat4 projection;
      fan::mat4 view;
    };

    void validate(VkResult result);

    inline constexpr std::uint16_t max_camera = 16;
    inline constexpr std::uint16_t max_textures = 1024;

    struct write_descriptor_set_t {
      // glsl layout binding
      std::uint32_t binding;
      std::uint32_t dst_binding = 0;

      // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
      // VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
      VkDescriptorType type;

      // VK_SHADER_STAGE_VERTEX_BIT
      // VK_SHADER_STAGE_FRAGMENT_BIT
      // Note: for VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER use VK_SHADER_STAGE_FRAGMENT_BIT
      VkShaderStageFlags flags;

      VkBuffer buffer = nullptr;

      std::uint64_t range;

      // for only VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
      // imageLayout can be VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
      bool use_image = false;
      std::uint32_t descriptor_count = 0;
      std::vector<VkDescriptorImageInfo> image_infos{max_textures};
    };

    inline constexpr std::uint32_t max_frames_in_flight = 1;

    inline std::uint32_t makeAccessMaskPipelineStageFlags(std::uint32_t accessMask) {
      static constexpr std::uint32_t accessPipes[] = {
        VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        VK_ACCESS_INDEX_READ_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        VK_ACCESS_UNIFORM_READ_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT
        | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT
        | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT
        | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_HOST_READ_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_HOST_WRITE_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_MEMORY_READ_BIT,
        0,
        VK_ACCESS_MEMORY_WRITE_BIT,
        0,
    #if VK_NV_device_generated_commands
        VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_NV,
        VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV,
        VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV,
        VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV,
    #endif
      };

      if (!accessMask)
      {
        return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      }

      std::uint32_t pipes = 0;

      for (std::uint32_t i = 0; i < std::size(accessPipes); i += 2)
      {
        if (accessPipes[i] & accessMask)
        {
          pipes |= accessPipes[i + 1];
        }
      }

      if (pipes == 0) {
        fan::throw_error("vulkan - invalid pipes");
      }

      return pipes;
    }

    // view and image
    struct vai_t {
      struct properties_t {
        fan::vec2 swap_chain_size;
        VkFormat format;
        VkImageUsageFlags usage_flags;
        VkImageAspectFlags aspect_flags;
      };
      void open(fan::vulkan::context_t& context, const properties_t& p);
      void close(fan::vulkan::context_t& context);

      void transition_image_layout(auto& context, VkImageLayout newLayout, VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT) {
        if (old_layout == newLayout) {
          return;
        }

        VkCommandBuffer commandBuffer = context.begin_single_time_commands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = old_layout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = aspectFlags;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
          barrier.srcAccessMask = 0;
          barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
          sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
          destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
          barrier.srcAccessMask = 0;
          barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
          sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
          destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }
        else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
          barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
          sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
          destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else {
          fan::throw_error("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
          commandBuffer,
          sourceStage, destinationStage,
          0,
          0, nullptr,
          0, nullptr,
          1, &barrier
        );

        context.end_single_time_commands(commandBuffer);

        old_layout = newLayout;
      }

      VkFormat format;

      VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;

      VkImage image;
      VkImageView image_view;
      VkDeviceMemory memory;
    };

    VkPipelineColorBlendAttachmentState get_default_color_blend() {
      VkPipelineColorBlendAttachmentState color_blend_attachment{};
      color_blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT
        ;
      color_blend_attachment.blendEnable = VK_TRUE;
      color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
      return color_blend_attachment;
    }
  }
}

export namespace fan {
  namespace vulkan {

    struct context_t {

      struct push_constants_t {
        std::uint32_t texture_id;
        std::uint32_t camera_id;
      };

      struct descriptor_t {

        using properties_t = std::vector<fan::vulkan::write_descriptor_set_t>;

        void open(fan::vulkan::context_t& context, const properties_t& properties) {
          m_properties = properties;

          std::vector<VkDescriptorSetLayoutBinding> uboLayoutBinding(properties.size());
          for (std::uint16_t i = 0; i < properties.size(); ++i) {
            uboLayoutBinding[i].binding = properties[i].binding;
            uboLayoutBinding[i].descriptorCount = m_properties[i].descriptor_count;
            if (uboLayoutBinding[i].descriptorCount == 0) {
              uboLayoutBinding[i].descriptorCount = m_properties[i].use_image ? max_textures : 1;
            }
            uboLayoutBinding[i].descriptorType = properties[i].type;
            uboLayoutBinding[i].stageFlags = properties[i].flags;
          }

          VkDescriptorSetLayoutCreateInfo layoutInfo{};
          layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
          layoutInfo.bindingCount = std::size(uboLayoutBinding);
          layoutInfo.pBindings = uboLayoutBinding.data();

          validate(vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &m_layout));

          std::array<VkDescriptorSetLayout, max_frames_in_flight> layouts;
          for (std::uint32_t i = 0; i < max_frames_in_flight; ++i) {
            layouts[i] = m_layout;
          }
          VkDescriptorSetAllocateInfo allocInfo{};
          allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
          allocInfo.descriptorPool = context.descriptor_pool.m_descriptor_pool;
          allocInfo.descriptorSetCount = layouts.size();
          allocInfo.pSetLayouts = layouts.data();

          validate(vkAllocateDescriptorSets(context.device, &allocInfo, m_descriptor_set));
        }
        void close(fan::vulkan::context_t& context) {
          vkDestroyDescriptorSetLayout(context.device, m_layout, 0);
        }


        // for buffer update, need to manually call .m_properties.common
        void update(
  fan::vulkan::context_t& context,
  std::uint32_t n,
  std::uint32_t begin = 0,
  std::uint32_t texture_n = max_textures,
  std::uint32_t texture_begin = 0
) {
  std::vector<VkDescriptorBufferInfo> buffer_infos(n);
  std::vector<VkWriteDescriptorSet> descriptor_writes(n);

  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint32_t j = begin + i;

    auto& write = descriptor_writes[i];
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptor_set[context.current_frame];
    write.dstBinding = m_properties[j].dst_binding;
    write.dstArrayElement = texture_begin;
    write.descriptorType = m_properties[j].type;

    if (m_properties[j].use_image) {
      std::uint32_t descriptor_n = texture_n;
      if (texture_n == max_textures && m_properties[j].descriptor_count != 0) {
        descriptor_n = m_properties[j].descriptor_count;
      }
      if (texture_begin + descriptor_n > m_properties[j].image_infos.size()) {
        fan::throw_error("descriptor image update out of range");
      }
      write.descriptorCount = descriptor_n;
      write.pImageInfo = m_properties[j].image_infos.data() + texture_begin;
    }
    else {
      buffer_infos[i].buffer = m_properties[j].buffer;
      buffer_infos[i].offset = 0;
      buffer_infos[i].range = m_properties[j].range;

      write.descriptorCount = 1;
      write.pBufferInfo = &buffer_infos[i];
    }
  }

  vkUpdateDescriptorSets(context.device, n, descriptor_writes.data(), 0, nullptr);
}

        properties_t m_properties;
        VkDescriptorSetLayout m_layout;
        VkDescriptorSet m_descriptor_set[fan::vulkan::max_frames_in_flight];
      };

      #include "memory.h"
      #include "uniform_block.h"
      #include "ssbo.h"

      struct descriptor_pool_t {

#define loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_image_sampler
        void open(fan::vulkan::context_t& context) {
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1024 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          #if defined(FAN_GUI)
              IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE +
          #endif
              fan::vulkan::max_textures * 32
            },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 64 },
          };
          VkDescriptorPoolCreateInfo pool_info = {};
          pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
          pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
          pool_info.maxSets = 0;
          for (VkDescriptorPoolSize& pool_size : pool_sizes)
            pool_info.maxSets += max_frames_in_flight * pool_size.descriptorCount;
          pool_info.poolSizeCount = (std::uint32_t)std::size(pool_sizes);
          pool_info.pPoolSizes = pool_sizes;
          ;
          fan::vulkan::validate(vkCreateDescriptorPool(context.device, &pool_info, nullptr, &m_descriptor_pool));
        }
        void close(fan::vulkan::context_t& context) {
          vkDestroyDescriptorPool(context.device, m_descriptor_pool, nullptr);
        }

        operator VkDescriptorPool() const {
          return m_descriptor_pool;
        }

        VkDescriptorPool m_descriptor_pool;
      }descriptor_pool;

      //-----------------------------shader-----------------------------

      struct shader_t {
        int projection_view[2]{ -1, -1 };
        fan::vulkan::context_t::uniform_block_t<fan::vulkan::view_projection_t, fan::vulkan::max_camera>* projection_view_block;
        VkPipelineShaderStageCreateInfo shader_stages[3]{};
      };

      fan::vulkan::context_t::shader_t& shader_get(fan::graphics::shader_nr_t nr);

      static std::vector<std::uint32_t> compile_file(const std::string& source_name,
        shaderc_shader_kind kind,
        const std::string& source);

      fan::graphics::shader_nr_t shader_create();

      void shader_erase(fan::graphics::shader_nr_t nr, int recycle = 1);

      void shader_use(fan::graphics::shader_nr_t nr);

      VkShaderModule create_shader_module(const std::vector<std::uint32_t>& code);

      void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code);

      void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code);

      void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code);

      void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code);

      void shader_set_compute(
        fan::graphics::shader_nr_t nr,
        const std::string_view file_path,
        const std::string& compute_code
      );

      void shader_set_camera(fan::graphics::shader_nr_t nr, fan::graphics::camera_nr_t camera_nr);

      void shader_dispatch_compute(
        fan::graphics::shader_nr_t nr,
        std::uint32_t x,
        std::uint32_t y,
        std::uint32_t z
      );

      static void parse_uniforms(std::string& shaderData, std::unordered_map<std::string, std::string>& uniform_type_table);

      bool shader_compile(fan::graphics::shader_nr_t nr);

      //-----------------------------shader-----------------------------

      //-----------------------------image-----------------------------

      struct image_format {
        static constexpr auto b8g8r8a8_unorm = VK_FORMAT_B8G8R8A8_UNORM;
        static constexpr auto r8b8g8a8_unorm = VK_FORMAT_R8G8B8A8_UNORM;
        static constexpr auto r8_unorm = VK_FORMAT_R8_UNORM;
        static constexpr auto r8_uint = VK_FORMAT_R8_UINT;
        static constexpr auto r8g8b8a8_srgb = VK_FORMAT_R8G8B8A8_SRGB;
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
        //static constexpr uint32_t internal_format = GL_RGBA;
        static constexpr VkFormat format = image_format::r8b8g8a8_unorm;
        //static constexpr uint32_t type = GL_UNSIGNED_BYTE;
        static constexpr VkFilter min_filter = image_filter::nearest;
        static constexpr VkFilter mag_filter = image_filter::nearest;
      };

      struct image_load_properties_t {
        //constexpr load_properties_t(auto a, auto b, auto c, auto d, auto e)
          //: visual_output(a), internal_format(b), format(c), type(d), filter(e) {}
        VkSamplerAddressMode visual_output = image_load_properties_defaults::visual_output;
        // unused opengl filler
        std::uint8_t internal_format = 0;
        //uintptr_t           internal_format = load_properties_defaults::internal_format;
        //uintptr_t           format = load_properties_defaults::format;
        //uintptr_t           type = load_properties_defaults::type;
        VkFormat format = image_load_properties_defaults::format;
        VkFilter           min_filter = image_load_properties_defaults::min_filter;
        VkFilter           mag_filter = image_load_properties_defaults::mag_filter;
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

      void transition_image_layout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
      void copy_buffer_to_image(
        VkBuffer buffer,
        VkImage image,
        VkFormat format,
        const fan::vec2ui& size,
        const fan::vec2ui& stride = fan::vec2ui(1)
      );



      void create_texture_sampler(VkSampler& sampler, const image_load_properties_t& lp);

      struct image_t {
        VkImage image_index = VK_NULL_HANDLE;
        VkImageView image_view = VK_NULL_HANDLE;
        VkDeviceMemory image_memory = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkBuffer staging_buffer = VK_NULL_HANDLE;
        VkDeviceMemory staging_buffer_memory = VK_NULL_HANDLE;
        void* data = nullptr;
      };

      VkFormat get_format_from_channels(int channels);
      constexpr std::uint32_t get_image_multiplier(VkFormat format) {
        switch (format) {
          case fan::vulkan::context_t::image_format::b8g8r8a8_unorm:
          {
            return 4;
          }
          case fan::vulkan::context_t::image_format::r8_unorm:
          {
            return 1; // 1?
          }
          case fan::vulkan::context_t::image_format::r8g8b8a8_srgb:
          {
            return 4;
          }
          case fan::vulkan::context_t::image_format::r8b8g8a8_unorm:
          {
            return 4;
          }
          default:
          {// removes warning
            break;
          }
        }
        fan::throw_error("failed to find format for image multiplier");
        return {};
      }

      std::vector<VkDescriptorImageInfo> image_pool; // for draw

      fan::graphics::image_nr_t image_create();

      std::uint64_t image_get_handle(fan::graphics::image_nr_t nr);

      fan::vulkan::context_t::image_t& image_get(fan::graphics::image_nr_t nr);

      void image_erase(fan::graphics::image_nr_t nr, int recycle = 1);


      void image_bind(fan::graphics::image_nr_t nr);

      void image_bind(fan::graphics::image_nr_t nr, std::uint32_t unit);

      void image_bind(
        fan::graphics::image_t nr,
        uint32_t unit,
        std::uint32_t access,
        std::uint32_t format
      );

      void image_unbind(fan::graphics::image_nr_t nr);

      fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_nr_t nr);

      void image_set_settings(fan::graphics::image_nr_t nr, const fan::vulkan::context_t::image_load_properties_t& p);

      void image_set_settings(const fan::vulkan::context_t::image_load_properties_t& p);

      fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info, const fan::vulkan::context_t::image_load_properties_t& p);

      fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info);

      fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_, const fan::vulkan::context_t::image_load_properties_t& p);

      fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_);

      fan::graphics::image_nr_t create_missing_texture();
      fan::graphics::image_nr_t create_transparent_texture();

      fan::graphics::image_nr_t image_load(fan::str_view_t path, const fan::vulkan::context_t::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());

      fan::graphics::image_nr_t image_load(fan::str_view_t path, const std::source_location& callers_path = std::source_location::current());

      void image_unload(fan::graphics::image_nr_t nr);

      void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::vulkan::context_t::image_load_properties_t& p);


      void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info);

      void image_reload(fan::graphics::image_nr_t nr, fan::str_view_t path, const fan::vulkan::context_t::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current());

      void image_reload(fan::graphics::image_nr_t nr, fan::str_view_t path, const std::source_location& callers_path = std::source_location::current());

      // creates single colored text size.x*size.y sized
      fan::graphics::image_nr_t image_create(const fan::color& color, const fan::vulkan::context_t::image_load_properties_t& p);

      fan::graphics::image_nr_t image_create(const fan::color& color);

      fan::graphics::image_nr_t image_create(void* data, const fan::vec2ui& size, const fan::vulkan::context_t::image_load_properties_t& p);

      std::vector<std::uint8_t> image_get_pixel_data(fan::graphics::image_nr_t nr, std::uint32_t format, fan::vec2 uvp, fan::vec2 uvs);

      fan::graphics::image_nr_t image_create_from_view(
        VkImageView view,
        VkImage image,
        fan::vec2ui size,
        VkFormat format
      );

      //-----------------------------image-----------------------------

      //-----------------------------camera-----------------------------

      fan::graphics::camera_nr_t camera_create();

      fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr);

      void camera_erase(fan::graphics::camera_nr_t nr);

      void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y);
      void camera_update_projection(fan::graphics::camera_nr_t nr);
      void camera_update_view(fan::graphics::camera_nr_t nr);

      fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y);

      fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr);

      void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp);

      fan::vec3 camera_get_center(fan::graphics::camera_nr_t nr);
      void camera_set_center(fan::graphics::camera_nr_t nr, const fan::vec3& cp);

      fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr);

      f32_t camera_get_zoom(fan::graphics::camera_nr_t nr);
      void camera_set_zoom(fan::graphics::camera_nr_t nr, f32_t new_zoom);

      void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size);

      void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset);

      //-----------------------------camera-----------------------------

      //-----------------------------viewport-----------------------------

      void viewport_set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);

      fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr);

      void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);

      fan::graphics::viewport_nr_t viewport_create();

      void viewport_erase(fan::graphics::viewport_nr_t nr);

      fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr);

      fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr);

      void viewport_zero(fan::graphics::viewport_nr_t nr);

      bool viewport_inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position);

      bool viewport_inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position);

      //-----------------------------viewport-----------------------------

      struct pipeline_t {

        struct properties_t {
          std::uint32_t descriptor_layout_count = 0;
          VkDescriptorSetLayout* descriptor_layout;
          fan::graphics::shader_nr_t shader;
          std::uint32_t push_constants_size = 0;
          std::uint32_t subpass = 0;

          std::uint32_t color_blend_attachment_count = 0;
          VkPipelineColorBlendAttachmentState* color_blend_attachment = 0;

          bool enable_depth_test = VK_TRUE;
          VkCompareOp depth_test_compare_op = VK_COMPARE_OP_LESS;
          VkPrimitiveTopology shape_type = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        };

        void open(fan::vulkan::context_t& context, const properties_t& p) {
          VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
          vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

          VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
          inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
          inputAssembly.topology = p.shape_type;
          inputAssembly.primitiveRestartEnable = VK_FALSE;

          VkPipelineViewportStateCreateInfo viewportState{};
          viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
          viewportState.viewportCount = 1;
          viewportState.scissorCount = 1;

          VkPipelineRasterizationStateCreateInfo rasterizer{};
          rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
          rasterizer.depthClampEnable = VK_FALSE;
          rasterizer.rasterizerDiscardEnable = VK_FALSE;
          rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
          rasterizer.lineWidth = 1.0f;
          rasterizer.cullMode = VK_CULL_MODE_NONE;
          rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
          rasterizer.depthBiasEnable = VK_FALSE;

          VkPipelineMultisampleStateCreateInfo multisampling{};
          multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
          multisampling.sampleShadingEnable = VK_FALSE;
          multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

          VkPipelineDepthStencilStateCreateInfo depthStencil{};
          depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
          depthStencil.depthTestEnable = p.enable_depth_test;
          depthStencil.depthWriteEnable = p.enable_depth_test;
          depthStencil.depthCompareOp = p.depth_test_compare_op;
          depthStencil.depthBoundsTestEnable = VK_FALSE;
          depthStencil.stencilTestEnable = VK_FALSE;

          VkPipelineColorBlendStateCreateInfo colorBlending{};
          colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
          colorBlending.logicOpEnable = VK_FALSE;
          colorBlending.logicOp = VK_LOGIC_OP_NO_OP;
          colorBlending.attachmentCount = p.color_blend_attachment_count;
          colorBlending.pAttachments = p.color_blend_attachment;
          colorBlending.blendConstants[0] = 1.0f;
          colorBlending.blendConstants[1] = 1.0f;
          colorBlending.blendConstants[2] = 1.0f;
          colorBlending.blendConstants[3] = 1.0f;

          std::vector<VkDynamicState> dynamicStates = {
              VK_DYNAMIC_STATE_VIEWPORT,
              VK_DYNAMIC_STATE_SCISSOR
          };
          VkPipelineDynamicStateCreateInfo dynamicState{};
          dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
          dynamicState.dynamicStateCount = dynamicStates.size();
          dynamicState.pDynamicStates = dynamicStates.data();

          VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
          pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
          pipelineLayoutInfo.setLayoutCount = p.descriptor_layout_count;
          pipelineLayoutInfo.pSetLayouts = p.descriptor_layout;

          VkPushConstantRange push_constant;
          push_constant.offset = 0;
          push_constant.size = p.push_constants_size;
          push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

          pipelineLayoutInfo.pPushConstantRanges = &push_constant;
          pipelineLayoutInfo.pushConstantRangeCount = p.push_constants_size == 0 ? 0 : 1;

          if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &m_layout) != VK_SUCCESS) {
            fan::throw_error("failed to create pipeline layout!");
          }

          VkGraphicsPipelineCreateInfo pipelineInfo{};
          pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
          pipelineInfo.stageCount = 2;
          pipelineInfo.pStages = context.shader_get(p.shader).shader_stages;
          pipelineInfo.pVertexInputState = &vertexInputInfo;
          pipelineInfo.pInputAssemblyState = &inputAssembly;
          pipelineInfo.pViewportState = &viewportState;
          pipelineInfo.pRasterizationState = &rasterizer;
          pipelineInfo.pMultisampleState = &multisampling;
          pipelineInfo.pDepthStencilState = &depthStencil;
          pipelineInfo.pColorBlendState = &colorBlending;
          pipelineInfo.pDynamicState = &dynamicState;
          pipelineInfo.layout = m_layout;
          pipelineInfo.renderPass = context.render_pass;
          pipelineInfo.subpass = p.subpass;
          pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

          shader_nr = p.shader;

          if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
            fan::throw_error("failed to create graphics pipeline");
          }
        }
        void close(fan::vulkan::context_t& context) {
          vkDestroyPipeline(context.device, m_pipeline, nullptr);
          vkDestroyPipelineLayout(context.device, m_layout, nullptr);
        }

        fan::graphics::shader_nr_t shader_nr;

        operator VkPipeline() const {
          return m_pipeline;
        }

        VkPipelineLayout m_layout;
        VkPipeline m_pipeline;
      };

      static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
      static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

      void open_no_window();
    #if defined(loco_window)
      void open(fan::window_t& window);
#endif

      void close_vais(std::vector<fan::vulkan::vai_t>& v);

      void destroy_vulkan_soft();

    public:
      void gui_close();

      void close();

      void cleanup_swap_chain_dependencies();

      void cleanup_swap_chain();

      void recreate_swap_chain_dependencies();

      // if swapchain changes, reque
      void update_swapchain_dependencies();

      void recreate_swap_chain(fan::window_t* window, VkResult err);

      //void recreate_swap_chain(const fan::vec2i& window_size);

      void create_instance();

      void populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& create_info);

      void setup_debug_messenger();

#if defined(loco_window)
      void create_surface(GLFWwindow* window);
#endif

      void pick_physical_device();

      void create_logical_device();


    #if defined(loco_window)
      void create_swap_chain(const fan::vec2ui& framebuffer_size);
    #endif

      VkImageView create_image_view(VkImage image, VkFormat format, VkImageAspectFlags aspect_flags);

      void create_image_views();


      void create_render_pass();


      void create_framebuffers();

      void create_command_pool();

      VkFormat find_supported_format(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

      VkFormat find_depth_format();

      bool has_stencil_component(VkFormat format);

      void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory);

      VkCommandBuffer begin_single_time_commands();

      void end_single_time_commands(VkCommandBuffer command_buffer);

      void copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size);

      std::uint32_t find_memory_type(std::uint32_t type_filter, VkMemoryPropertyFlags properties) const;

      void create_command_buffers();


      void bind_draw(
        const fan::vulkan::context_t::pipeline_t& pipeline,
        std::uint32_t descriptor_count,
        VkDescriptorSet* descriptor_sets);

      // assumes things are already bound
      void bindless_draw(
        std::uint32_t vertex_count,
        std::uint32_t instance_count,
        std::uint32_t first_instance);

      void draw(
        std::uint32_t vertex_count,
        std::uint32_t instance_count,
        std::uint32_t first_instance,
        const fan::vulkan::context_t::pipeline_t& pipeline,
        std::uint32_t descriptor_count,
        VkDescriptorSet* descriptor_sets
      );

      void create_sync_objects();

       //----------------------------------------------imgui stuff----------------------------------------------
      bool                     SwapChainRebuild = false;
      #if defined(FAN_GUI)
      ImGui_ImplVulkanH_Window MainWindowData;
      std::uint32_t                 MinImageCount = 2;

      void ImGuiSetupVulkanWindow();

      static void ImGuiFrameRender(void* ctx, VkResult next_image_khr_err, fan::color clear_color);
      //----------------------------------------------imgui stuff----------------------------------------------
#endif
      VkResult end_render();

      VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats);

      VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes);


      VkExtent2D choose_swap_extent(const fan::vec2ui& framebuffer_size, const VkSurfaceCapabilitiesKHR& capabilities);

      swap_chain_support_details_t query_swap_chain_support(VkPhysicalDevice device);

      bool is_device_suitable(VkPhysicalDevice device);

      bool check_device_extension_support(VkPhysicalDevice device);

      queue_family_indices_t find_queue_families(VkPhysicalDevice device);

      std::vector<std::string> get_required_extensions();


      bool check_validation_layer_support();

      static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData
      );

      #if defined(loco_window)
      void set_vsync(fan::window_t* window, bool flag);
      #endif

      VkInstance instance;
      VkDebugUtilsMessengerEXT debug_messenger;
      VkSurfaceKHR surface;

      VkPhysicalDevice physical_device = VK_NULL_HANDLE;
      VkDevice device = VK_NULL_HANDLE;

      VkQueue graphics_queue;
#if defined(loco_window)
      VkQueue present_queue;
#endif

      std::vector<vai_t> mainColorImageViews;
      std::vector<vai_t> postProcessedColorImageViews;
      std::vector<vai_t> depthImageViews;
      std::vector<vai_t> downscaleImageViews1;
      std::vector<vai_t> upscaleImageViews1;
      std::vector<vai_t> vai_depth;

      std::vector<VkImage> swap_chain_images;
      std::vector<VkImageView> swap_chain_image_views;

      VkSwapchainKHR swap_chain;
      VkFormat swap_chain_image_format;
      fan::vec2 swap_chain_size;
      std::vector<VkFramebuffer> swap_chain_framebuffers;
      VkPresentModeKHR present_mode;
      VkSurfaceFormatKHR surface_format;

      VkRenderPass render_pass;

      VkCommandPool command_pool;
      std::uint32_t queue_family = -1;
      std::uint32_t min_image_count = 0;
      std::uint32_t image_count = 0;
      
      std::vector<VkCommandBuffer> command_buffers;

      std::vector<std::function<void(VkCommandBuffer)>> begin_cmd_cb;

      std::vector<VkSemaphore> image_available_semaphores;
      std::vector<VkSemaphore> render_finished_semaphores;
      std::vector<VkFence> in_flight_fences;
      std::uint32_t current_frame = 0;

      fan::window_t::resize_handle_t window_resize_handle;

      bool enable_clear = true;
      bool shapes_top = false;

      bool vsync = true;
      std::uint32_t image_index;

      fan::vulkan::context_t::pipeline_t render_fullscreen_pl;

      bool command_buffer_in_use = false;
      bool supports_validation_layers = true;

      fan::vulkan::context_t::memory_write_queue_t memory_queue;
    };
  }
}
//#include "ssbo.h"

export namespace fan {
  namespace vulkan {
    namespace core {
      std::uint32_t get_draw_mode(std::uint8_t draw_mode) {
        switch (draw_mode) {
        case fan::graphics::primitive_topology_t::points:
          return fan::vulkan::context_t::primitive_topology_t::points;
        case fan::graphics::primitive_topology_t::lines:
          return fan::vulkan::context_t::primitive_topology_t::lines;
        case fan::graphics::primitive_topology_t::line_strip:
          return fan::vulkan::context_t::primitive_topology_t::line_strip;
        case fan::graphics::primitive_topology_t::triangles:
          return fan::vulkan::context_t::primitive_topology_t::triangles;
        case fan::graphics::primitive_topology_t::triangle_strip:
          return fan::vulkan::context_t::primitive_topology_t::triangle_strip;
        case fan::graphics::primitive_topology_t::triangle_fan:
          return fan::vulkan::context_t::primitive_topology_t::triangle_fan;
        case fan::graphics::primitive_topology_t::lines_with_adjacency:
          return fan::vulkan::context_t::primitive_topology_t::lines_with_adjacency;
        case fan::graphics::primitive_topology_t::line_strip_with_adjacency:
          return fan::vulkan::context_t::primitive_topology_t::line_strip_with_adjacency;
        case fan::graphics::primitive_topology_t::triangles_with_adjacency:
          return fan::vulkan::context_t::primitive_topology_t::triangles_with_adjacency;
        case fan::graphics::primitive_topology_t::triangle_strip_with_adjacency:
          return fan::vulkan::context_t::primitive_topology_t::triangle_strip_with_adjacency;
        default:
          fan::throw_error("invalid draw mode");
          return -1;
        }
      }
    }
  }
}

void fan::vulkan::image_create(const fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

auto fan::graphics::format_converter::image_global_to_vulkan(const fan::graphics::image_load_properties_t& p);


export namespace fan::graphics {
  fan::graphics::context_functions_t get_vk_context_functions();
  fan::vulkan::context_t& get_vk_context();
}

#endif
