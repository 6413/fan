module;


#include <fan/utility.h>

#if defined(fan_vulkan)

#include <optional>
#include <vector>
#include <set>


#if defined(fan_platform_windows)
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
#define VK_USE_PLATFORM_XLIB_KHR
#endif

#if defined(fan_gui)
  #include <fan/imgui/imgui_impl_vulkan.h>
#endif

#define loco_window

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>


#include <vulkan/vulkan.h>
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
export module fan.graphics.vulkan.core;

#if defined(fan_vulkan)

import fan.physics.collision.rectangle;

import fan.types.fstring;
import fan.types.color;

#if defined(loco_window)
  import fan.window;
#endif

import fan.utility;
export import fan.print;
export import fan.graphics.image_load;
export import fan.graphics.common_context;

#ifndef camera_list
  #define __fan_internal_camera_list (*(fan::graphics::camera_list_t*)fan::graphics::get_camera_list((uint8_t*)this))
#endif

#ifndef shader_list
  #define __fan_internal_shader_list (*(fan::graphics::shader_list_t*)fan::graphics::get_shader_list((uint8_t*)this))
#endif

#ifndef image_list
  #define __fan_internal_image_list (*(fan::graphics::image_list_t*)fan::graphics::get_image_list((uint8_t*)this))
#endif

#ifndef viewport_list
  #define __fan_internal_viewport_list (*(fan::graphics::viewport_list_t*)fan::graphics::get_viewport_list((uint8_t*)this))
#endif

#if defined(fan_compiler_msvc)
  #pragma comment(lib, "vulkan-1.lib")
  #pragma comment(lib, "shaderc_combined_mt.lib")
#endif


const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  }
  else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

struct queue_family_indices_t {
  std::optional<uint32_t> graphics_family;
#if defined(loco_window)
  std::optional<uint32_t> present_family;
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
  VkFormat global_to_vulkan_format(uintptr_t format) {
    if (format == image_format::b8g8r8a8_unorm) return VK_FORMAT_B8G8R8A8_UNORM;
    if (format == image_format::r8b8g8a8_unorm) return VK_FORMAT_R8G8B8A8_UNORM;
    if (format == image_format::r8_unorm) return VK_FORMAT_R8_UNORM;
    if (format == image_format::r8_uint) return VK_FORMAT_R8_UINT;
    if (format == image_format::r8g8b8a8_srgb) return VK_FORMAT_R8G8B8A8_SRGB;
    if (format == image_format::rgba_unorm) return VK_FORMAT_R8G8B8A8_UNORM;
  #if fan_debug >= fan_debug_high
    fan::throw_error("invalid format");
  #endif
    return VK_FORMAT_R8G8B8A8_UNORM;
  }
  VkSamplerAddressMode global_to_vulkan_address_mode(uintptr_t mode) {
    if (mode == image_sampler_address_mode::repeat) return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    if (mode == image_sampler_address_mode::mirrored_repeat) return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    if (mode == image_sampler_address_mode::clamp_to_edge) return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (mode == image_sampler_address_mode::clamp_to_border) return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    if (mode == image_sampler_address_mode::mirrored_clamp_to_edge) return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
  #if fan_debug >= fan_debug_high
    fan::throw_error("invalid format");
  #endif
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  }
  VkFilter global_to_vulkan_filter(uintptr_t filter) {
    if (filter == image_filter::nearest) return VK_FILTER_NEAREST;
    if (filter == image_filter::linear) return VK_FILTER_LINEAR;
  #if fan_debug >= fan_debug_high
    fan::throw_error("invalid format");
  #endif
    return VK_FILTER_NEAREST;
  }

  uint32_t vulkan_to_global_format(VkFormat format) {
    if (format == VK_FORMAT_B8G8R8A8_UNORM) return fan::graphics::image_format::b8g8r8a8_unorm;
    if (format == VK_FORMAT_R8G8B8A8_UNORM) return fan::graphics::image_format::r8b8g8a8_unorm;
    if (format == VK_FORMAT_R8_UNORM) return fan::graphics::image_format::r8_unorm;
    if (format == VK_FORMAT_R8_UINT) return fan::graphics::image_format::r8_uint;
    if (format == VK_FORMAT_R8G8B8A8_SRGB) return fan::graphics::image_format::r8g8b8a8_srgb;
#if fan_debug >= fan_debug_high
    fan::throw_error("invalid format");
#endif
    return fan::graphics::image_format::rgba_unorm;
  }
  uint32_t vulkan_to_global_address_mode(VkSamplerAddressMode mode) {
    if (mode == VK_SAMPLER_ADDRESS_MODE_REPEAT) return fan::graphics::image_sampler_address_mode::repeat;
    if (mode == VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT) return fan::graphics::image_sampler_address_mode::mirrored_repeat;
    if (mode == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE) return fan::graphics::image_sampler_address_mode::clamp_to_edge;
    if (mode == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) return fan::graphics::image_sampler_address_mode::clamp_to_border;
    if (mode == VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE) return fan::graphics::image_sampler_address_mode::mirrored_clamp_to_edge;
#if fan_debug >= fan_debug_high
    fan::throw_error("invalid format");
#endif
    return fan::graphics::image_sampler_address_mode::repeat;
  }
  uint32_t vulkan_to_global_filter(VkFilter filter) {
    if (filter == VK_FILTER_NEAREST) return fan::graphics::image_filter::nearest;
    if (filter == VK_FILTER_LINEAR) return fan::graphics::image_filter::linear;
#if fan_debug >= fan_debug_high
    fan::throw_error("invalid format");
#endif
    return fan::graphics::image_filter::nearest;
  }

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

constexpr static uint32_t get_image_multiplier(VkFormat format);

export namespace fan {
  namespace vulkan {

    void validate(VkResult result) {
      if (result != VK_SUCCESS) {
        fan::throw_error("function failed");
      }
    }

    inline constexpr uint16_t max_camera = 16;
    inline constexpr uint16_t max_textures = 0xffff;

    struct write_descriptor_set_t {
      // glsl layout binding
      uint32_t binding;
      uint32_t dst_binding = 0;

      // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
      // VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
      VkDescriptorType type;

      // VK_SHADER_STAGE_VERTEX_BIT
      // VK_SHADER_STAGE_FRAGMENT_BIT
      // Note: for VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER use VK_SHADER_STAGE_FRAGMENT_BIT
      VkShaderStageFlags flags;

      VkBuffer buffer = nullptr;

      uint64_t range;

      // for only VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
      // imageLayout can be VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
      bool use_image = false;
      std::vector<VkDescriptorImageInfo> image_infos{max_textures};
    };

    inline constexpr uint32_t max_frames_in_flight = 1;

    inline uint32_t makeAccessMaskPipelineStageFlags(uint32_t accessMask) {
      static constexpr uint32_t accessPipes[] = {
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

      uint32_t pipes = 0;

      for (uint32_t i = 0; i < std::size(accessPipes); i += 2)
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
          throw std::invalid_argument("unsupported layout transition!");
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
        uint32_t texture_id;
        uint32_t camera_id;
      };

      struct descriptor_t {

        using properties_t = std::vector<fan::vulkan::write_descriptor_set_t>;

        void open(fan::vulkan::context_t& context, const properties_t& properties) {
          m_properties = properties;

          std::vector<VkDescriptorSetLayoutBinding> uboLayoutBinding(properties.size());
          for (uint16_t i = 0; i < properties.size(); ++i) {
            uboLayoutBinding[i].binding = properties[i].binding;
            uboLayoutBinding[i].descriptorCount = 1;
            if (m_properties[i].use_image) {
              uboLayoutBinding[i].descriptorCount = max_textures;
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
          for (uint32_t i = 0; i < max_frames_in_flight; ++i) {
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
          uint32_t n,
          uint32_t begin = 0,
          uint32_t texture_n = max_textures,
          uint32_t texture_begin = 0
        ) {
          uint32_t frame = context.current_frame;

          std::vector<VkDescriptorBufferInfo> bufferInfo(n);

          std::vector<VkWriteDescriptorSet> descriptorWrites(begin + n);

          for (uint32_t j = begin; j < begin + n; ++j) {

            if (m_properties[j].buffer) {
              bufferInfo[j].buffer = m_properties[j].buffer;
              bufferInfo[j].offset = 0;
              bufferInfo[j].range = m_properties[j].range;
            }

            descriptorWrites[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[j].dstSet = m_descriptor_set[0];
            descriptorWrites[j].descriptorType = m_properties[j].type;
            descriptorWrites[j].descriptorCount = 1;
            descriptorWrites[j].pBufferInfo = &bufferInfo[j];
            descriptorWrites[j].dstBinding = m_properties[j].dst_binding;

            // FIX
            if (m_properties[j].use_image) {
              descriptorWrites[j].pImageInfo = &m_properties[j].image_infos[texture_begin];
              descriptorWrites[j].descriptorCount = texture_n;
            }
          }
          vkUpdateDescriptorSets(context.device, n, descriptorWrites.data() + begin, 0, nullptr);
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
          VkDescriptorPoolSize pool_sizes[] =
          {

            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            #if defined(fan_gui)
            IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE +
        #endif
            5 * max_frames_in_flight},
          };
          VkDescriptorPoolCreateInfo pool_info = {};
          pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
          pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
          pool_info.maxSets = 0;
          for (VkDescriptorPoolSize& pool_size : pool_sizes)
            pool_info.maxSets += max_frames_in_flight * pool_size.descriptorCount;
          pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
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
      struct view_projection_t {
        fan::mat4 projection;
        fan::mat4 view;
      };

      struct shader_t {
        int projection_view[2]{ -1, -1 };
        fan::vulkan::context_t::uniform_block_t<fan::vulkan::context_t::view_projection_t, fan::vulkan::max_camera>* projection_view_block;
        VkPipelineShaderStageCreateInfo shader_stages[2]{};
      };

      fan::vulkan::context_t::shader_t& shader_get(fan::graphics::shader_nr_t nr) {
        return *(fan::vulkan::context_t::shader_t*)__fan_internal_shader_list[nr].internal;
      }

      static std::vector<uint32_t> compile_file(const std::string& source_name,
        shaderc_shader_kind kind,
        const std::string& source) {
        shaderc::Compiler compiler;
        shaderc::CompileOptions options;

        // Like -DMY_DEFINE=1
        //options.AddMacroDefinition("MY_DEFINE", "1");
      #if fan_debug > 1
        options.SetOptimizationLevel(shaderc_optimization_level_zero);
      #else
        options.SetOptimizationLevel(shaderc_optimization_level_performance);
      #endif

        shaderc::SpvCompilationResult module =
          compiler.CompileGlslToSpv(source.c_str(), kind, source_name.c_str(), options);

        if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
          fan::throw_error(module.GetErrorMessage().c_str());
        }

        return { module.cbegin(), module.cend() };
      }

      fan::graphics::shader_nr_t shader_create() {
        fan::graphics::shader_nr_t nr = __fan_internal_shader_list.NewNode();
        __fan_internal_shader_list[nr].internal = new fan::vulkan::context_t::shader_t;
        auto& shader = shader_get(nr);
        shader.projection_view_block = new std::remove_pointer_t<decltype(shader.projection_view_block)>;
        //TODO
        shader.projection_view_block->open(*this);
        for (uint32_t i = 0; i < fan::vulkan::max_camera; ++i) {
          shader.projection_view_block->push_ram_instance(*this, {});
        }
        return nr;
      }

      void shader_erase(fan::graphics::shader_nr_t nr, int recycle = 1) {
        auto& shader = shader_get(nr);
        if (shader.shader_stages[0].module) {
          vkDestroyShaderModule(device, shader.shader_stages[0].module, nullptr);
        }
        if (shader.shader_stages[1].module) {
          vkDestroyShaderModule(device, shader.shader_stages[1].module, nullptr);
        }
        //TODO
        shader.projection_view_block->close(*this);
        delete shader.projection_view_block;
        delete static_cast<fan::vulkan::context_t::shader_t*>(__fan_internal_shader_list[nr].internal);
        if (recycle) {
          __fan_internal_shader_list.Recycle(nr);
        }
      }

      void shader_use(fan::graphics::shader_nr_t nr) {
        // TODO - required?
      }

      VkShaderModule create_shader_module(const std::vector<uint32_t>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size() * sizeof(typename std::remove_reference_t<decltype(code)>::value_type);
        createInfo.pCode = code.data();

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
          fan::throw_error("failed to create shader module!");
        }

        return shaderModule;
      }

      void shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code) {
        __fan_internal_shader_list[nr].svertex = vertex_code;
        // fan::print(
        //   "processed vertex shader:", path, "resulted in:",
        // preprocess_shader(shader_name.c_str(), shaderc_glsl_vertex_shader, shader_code);
        // );
      }

      void shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code) {
        auto& shader = shader_get(nr);
        __fan_internal_shader_list[nr].sfragment = fragment_code;
        //fan::print(
          // "processed vertex shader:", path, "resulted in:",
        //preprocess_shader(shader_name.c_str(), shaderc_glsl_fragment_shader, shader_code);
        //);
      }

      static void parse_uniforms(std::string& shaderData, std::unordered_map<std::string, std::string>& uniform_type_table) {
        size_t pos = 0;

        while ((pos = shaderData.find("uniform", pos)) != std::string::npos) {
          size_t endLine = shaderData.find(';', pos);
          if (endLine == std::string::npos) break;

          std::string line = shaderData.substr(pos, endLine - pos + 1);

          line = line.substr(7);

          size_t start = line.find_first_not_of(" \t");
          if (start == std::string::npos) {
            pos = endLine + 1;
            continue;
          }
          line = line.substr(start);

          size_t space1 = line.find_first_of(" \t");
          if (space1 == std::string::npos) {
            pos = endLine + 1;
            continue;
          }

          std::string type = line.substr(0, space1);
          line = line.substr(space1);
          line = line.substr(line.find_first_not_of(" \t"));

          size_t varEnd = line.find_first_of("=;");
          std::string name = line.substr(0, varEnd);

          name.erase(0, name.find_first_not_of(" \t"));
          name.erase(name.find_last_not_of(" \t") + 1);

          uniform_type_table[name] = type;

          pos = endLine + 1;
        }
      }

      bool shader_compile(fan::graphics::shader_nr_t nr) {
        auto& shader = shader_get(nr);
        {
          auto spirv = compile_file(/*vertex_code.c_str()*/ "some vertex file", shaderc_glsl_vertex_shader, __fan_internal_shader_list[nr].svertex);

          auto module_vertex = create_shader_module(spirv);

          VkPipelineShaderStageCreateInfo vert{};
          vert.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
          vert.stage = VK_SHADER_STAGE_VERTEX_BIT;
          vert.module = module_vertex;
          vert.pName = "main";

          shader.shader_stages[0] = vert;
        }
        {
          auto spirv = compile_file(/*shader_name.c_str()*/"some fragment file", shaderc_glsl_fragment_shader, __fan_internal_shader_list[nr].sfragment);

          auto module_fragment = create_shader_module(spirv);

          VkPipelineShaderStageCreateInfo frag{};
          frag.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
          frag.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
          frag.module = module_fragment;
          frag.pName = "main";

          shader.shader_stages[1] = frag;
        }

        std::string vertexData = __fan_internal_shader_list[nr].svertex;
        parse_uniforms(vertexData, __fan_internal_shader_list[nr].uniform_type_table);

        std::string fragmentData = __fan_internal_shader_list[nr].sfragment;
        parse_uniforms(fragmentData, __fan_internal_shader_list[nr].uniform_type_table);

        return 0;
      }

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
        static constexpr VkSamplerAddressMode visual_output = image_sampler_address_mode::repeat;
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
        uint8_t internal_format = 0;
        //uintptr_t           internal_format = load_properties_defaults::internal_format;
        //uintptr_t           format = load_properties_defaults::format;
        //uintptr_t           type = load_properties_defaults::type;
        VkFormat format = image_load_properties_defaults::format;
        VkFilter           min_filter = image_load_properties_defaults::min_filter;
        VkFilter           mag_filter = image_load_properties_defaults::mag_filter;
      };

      struct primitive_topology_t {
        static constexpr uint32_t points = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        static constexpr uint32_t lines = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        static constexpr uint32_t line_strip = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        static constexpr uint32_t triangles = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        static constexpr uint32_t triangle_strip = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        static constexpr uint32_t triangle_fan = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        static constexpr uint32_t lines_with_adjacency = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
        static constexpr uint32_t line_strip_with_adjacency = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
        static constexpr uint32_t triangles_with_adjacency = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
        static constexpr uint32_t triangle_strip_with_adjacency = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
      };

      void transition_image_layout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer command_buffer = begin_single_time_commands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
          barrier.srcAccessMask = 0;
          barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

          sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
          destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
          barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

          sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
          destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else {
          throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
          command_buffer,
          sourceStage, destinationStage,
          0,
          0, nullptr,
          0, nullptr,
          1, &barrier
        );

        end_single_time_commands(command_buffer);
      }
      void copy_buffer_to_image(VkBuffer buffer, VkImage image, VkFormat format, const fan::vec2ui& size, const fan::vec2ui& stride = 1) {
        VkCommandBuffer command_buffer = begin_single_time_commands();

        uint32_t block_width = get_image_multiplier(format);
        uint32_t block_x = (block_width - 1) / block_width;
        uint32_t block_y = (block_width - 1) / block_width;
        uint32_t block_h = std::max(1u, (size.y + block_width - 1) / block_width);
        // Flush CPU and GPU caches if not coherent mapping.
        VkDeviceSize buffer_flush_offset = block_y * stride.x;
        VkDeviceSize buffer_flush_size = block_h * stride.x;

        /*
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
        size.x,
        size.y,
        1
        };
        */

        VkBufferImageCopy region = {
            block_y * stride.x + block_x,// VkDeviceSize             bufferOffset
            size.x,                                        // uint32_t                 bufferRowLength
            0,                                              // uint32_t                 bufferImageHeight
            { 0, 0, 0, 1 },                  // VkImageSubresourceLayers imageSubresource
            { 0, 0, 0 },  // VkOffset3D               imageOffset
            { size.x, size.y, 1 }                              // VkExtent3D               imageExtent
        };

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;

        vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        end_single_time_commands(command_buffer);
      }

      void create_texture_sampler(VkSampler& sampler, const image_load_properties_t& lp) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical_device, &properties);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = lp.mag_filter;
        samplerInfo.minFilter = lp.min_filter;
        samplerInfo.addressModeU = lp.visual_output;
        samplerInfo.addressModeV = lp.visual_output;
        samplerInfo.addressModeW = lp.visual_output;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
          throw std::runtime_error("failed to create texture sampler!");
        }
      }

      struct image_t {
        VkImage image_index = 0;
        VkImageView image_view;
        VkDeviceMemory image_memory;
        VkSampler sampler;
        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;
        void* data;
      };

      std::vector<VkDescriptorImageInfo> image_pool; // for draw

      fan::graphics::image_nr_t image_create() {
        fan::graphics::image_nr_t nr = __fan_internal_image_list.NewNode();
        __fan_internal_image_list[nr].internal = new fan::vulkan::context_t::image_t;
        return nr;
      }

      uint64_t image_get_handle(fan::graphics::image_nr_t nr) {
        fan::throw_error("invalid call");
        return 0;
      }

      fan::vulkan::context_t::image_t& image_get(fan::graphics::image_nr_t nr) {
        return *(fan::vulkan::context_t::image_t*)__fan_internal_image_list[nr].internal;
      }

      void image_erase(fan::graphics::image_nr_t nr, int recycle = 1) {
        fan::vulkan::context_t::image_t& image = image_get(nr);
        vkDestroySampler(device, image.sampler, nullptr);
        vkDestroyBuffer(device, image.staging_buffer, nullptr);
        vkFreeMemory(device, image.staging_buffer_memory, nullptr);
        vkDestroyImage(device, image.image_index, 0);
        vkDestroyImageView(device, image.image_view, 0);
        vkFreeMemory(device, image.image_memory, nullptr);
        delete static_cast<fan::vulkan::context_t::image_t*>(__fan_internal_image_list[nr].internal);
        if (recycle) {
          __fan_internal_image_list.Recycle(nr);
        }
      }

      void image_bind(fan::graphics::image_nr_t nr) {

      }

      void image_unbind(fan::graphics::image_nr_t nr) {

      }

      fan::graphics::image_load_properties_t& image_get_settings(fan::graphics::image_nr_t nr) {
        return __fan_internal_image_list[nr].image_settings;
      }

      void image_set_settings(const fan::vulkan::context_t::image_load_properties_t& p) {

      }

      fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info, const fan::vulkan::context_t::image_load_properties_t& p) {
        fan::graphics::image_nr_t nr = image_create();

        fan::vulkan::context_t::image_t& image = image_get(nr);
        auto& image_data = __fan_internal_image_list[nr];
        image_data.size = image_info.size;
        __fan_internal_image_list[nr].image_path = "";

        auto image_multiplier = get_image_multiplier(p.format);

        VkDeviceSize image_size_bytes = image_info.size.multiply() * image_multiplier;

        create_buffer(
          image_size_bytes,
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          //VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          image.staging_buffer,
          image.staging_buffer_memory
        );

        vkMapMemory(device, image.staging_buffer_memory, 0, image_size_bytes, 0, &image.data);
        memcpy(image.data, image_info.data, image_size_bytes); // TODO  / 4 in yuv420p

        fan::vulkan::image_create(
          *this,
          image_info.size,
          p.format,
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          image.image_index,
          image.image_memory
        );
        image.image_view = create_image_view(image.image_index, p.format, VK_IMAGE_ASPECT_COLOR_BIT);
        create_texture_sampler(image.sampler, p);

        transition_image_layout(image.image_index, p.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copy_buffer_to_image(image.staging_buffer, image.image_index, p.format, image_info.size);
        transition_image_layout(image.image_index, p.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        return nr;
      }

      fan::graphics::image_nr_t image_load(const fan::image::info_t& image_info) {
        return image_load(image_info, fan::vulkan::context_t::image_load_properties_t());
      }

      fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_, const fan::vulkan::context_t::image_load_properties_t& p) {

        fan::image::info_t ii;
        ii.data = colors;
        ii.size = size_;
        ii.channels = 4;
        fan::graphics::image_nr_t nr = image_load(ii, p);

        image_set_settings(p);

        auto& image_data = __fan_internal_image_list[nr];
        image_data.size = size_;

        return nr;
      }

      fan::graphics::image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_) {
        return image_load(colors, size_, fan::vulkan::context_t::image_load_properties_t());
      }

      fan::graphics::image_nr_t create_missing_texture() {
        fan::vulkan::context_t::image_load_properties_t p;

        fan::vec2i image_size = fan::vec2i(2, 2);
        fan::graphics::image_nr_t nr = image_load((fan::color*)fan::image::missing_texture_pixels, image_size, p);

        auto& image_data = __fan_internal_image_list[nr];
        image_data.size = image_size;
        __fan_internal_image_list[nr].image_settings = fan::graphics::format_converter::image_vulkan_to_global(p);
        return nr;
      }
      fan::graphics::image_nr_t create_transparent_texture() {
        fan::vulkan::context_t::image_load_properties_t p;

        fan::vec2i image_size = fan::vec2i(2, 2);
        fan::graphics::image_nr_t nr = image_load((fan::color*)fan::image::transparent_texture_pixels, image_size, p);

        auto& image_data = __fan_internal_image_list[nr];
        image_data.size = image_size;

        return nr;
      }

      fan::graphics::image_nr_t image_load(const std::string& path, const fan::vulkan::context_t::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {

      #if fan_assert_if_same_path_loaded_multiple_times

        static std::unordered_map<std::string, bool> existing_images;

        if (existing_images.find(path) != existing_images.end()) {
          fan::throw_error("image already existing " + path);
        }

        existing_images[path] = 0;

      #endif

        fan::image::info_t image_info;
        if (fan::image::load(path, &image_info, callers_path)) {
          return create_missing_texture();
        }
        fan::graphics::image_nr_t nr = image_load(image_info, p);
        __fan_internal_image_list[nr].image_path = path;
        fan::image::free(&image_info);
        return nr;
      }

      fan::graphics::image_nr_t image_load(const std::string& path, const std::source_location& callers_path = std::source_location::current()) {
        return image_load(path, fan::vulkan::context_t::image_load_properties_t(), callers_path);
      }

      void image_unload(fan::graphics::image_nr_t nr) {
        image_erase(nr);
      }

      void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::vulkan::context_t::image_load_properties_t& p) {
        auto image_multiplier = get_image_multiplier(p.format);

        VkDeviceSize image_size = image_info.size.multiply() * image_multiplier;

        fan::vulkan::context_t::image_t& image = image_get(nr);
        auto& image_data = __fan_internal_image_list[nr];
        image_data.size = image_info.size;

        if (image.image_index == 0) {
          VkDeviceSize image_size_bytes = image_info.size.multiply() * image_multiplier;

          create_buffer(
            image_size_bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            //VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            image.staging_buffer,
            image.staging_buffer_memory
          );

          vkMapMemory(device, image.staging_buffer_memory, 0, image_size_bytes, 0, &image.data);
          memcpy(image.data, image_info.data, image_size_bytes); // TODO  / 4 in yuv420p

          fan::vulkan::image_create(
            *this,
            image_info.size,
            p.format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            image.image_index,
            image.image_memory
          );
          image.image_view = create_image_view(image.image_index, p.format, VK_IMAGE_ASPECT_COLOR_BIT);
          create_texture_sampler(image.sampler, p);
        }

        memcpy(image.data, image_info.data, image_size / 4);


        transition_image_layout(image.image_index, p.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copy_buffer_to_image(image.staging_buffer, image.image_index, p.format, image_info.size);
        transition_image_layout(image.image_index, p.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      }

      void image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) {
        image_reload(nr, image_info, fan::vulkan::context_t::image_load_properties_t());
      }

      void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const fan::vulkan::context_t::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
        fan::image::info_t image_info;
        if (fan::image::load(path, &image_info, callers_path)) {
          image_info.data = (void*)fan::image::missing_texture_pixels;
          image_info.size = 2;
          image_info.channels = 4;
          image_info.type = -1; // ignore free
        }
        image_reload(nr, image_info, p);
        __fan_internal_image_list[nr].image_path = path;
        fan::image::free(&image_info);
      }

      void image_reload(fan::graphics::image_nr_t nr, const std::string& path, const std::source_location& callers_path = std::source_location::current()) {
        image_reload(nr, path, fan::vulkan::context_t::image_load_properties_t(), callers_path);
      }

      // creates single colored text size.x*size.y sized
      fan::graphics::image_nr_t image_create(const fan::color& color, const fan::vulkan::context_t::image_load_properties_t& p) {

        uint8_t pixels[4];
        for (uint32_t p = 0; p < fan::color::size(); p++) {
          pixels[p] = color[p] * 255;
        }

        fan::image::info_t ii;

        ii.data = (void*)&color.r;
        ii.size = 1;
        ii.channels = 4;
        fan::graphics::image_nr_t nr = image_load(ii, p);

        image_bind(nr);

        image_set_settings(p);

        return nr;
      }

      fan::graphics::image_nr_t image_create(const fan::color& color) {
        return image_create(color, fan::vulkan::context_t::image_load_properties_t());
      }

      constexpr uint32_t get_image_multiplier(VkFormat format) {
        switch (format) {
        case fan::vulkan::context_t::image_format::b8g8r8a8_unorm: {
          return 4;
        }
        case fan::vulkan::context_t::image_format::r8_unorm: {
          return 4; // 1?
        }
        case fan::vulkan::context_t::image_format::r8g8b8a8_srgb: {
          return 4;
        }
        case fan::vulkan::context_t::image_format::r8b8g8a8_unorm: {
          return 4;
        }
        default: {// removes warning
          break;
        }
        }
        fan::throw_error("failed to find format for image multiplier");
        return {};
      }

      //-----------------------------image-----------------------------

      //-----------------------------camera-----------------------------

      fan::graphics::camera_nr_t camera_create() {
        return __fan_internal_camera_list.NewNode();
      }

      fan::graphics::context_camera_t& camera_get(fan::graphics::camera_nr_t nr) {
        return __fan_internal_camera_list[nr];
      }

      void camera_erase(fan::graphics::camera_nr_t nr) {
        __fan_internal_camera_list.Recycle(nr);
      }

      void camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
        fan::graphics::context_camera_t& camera = camera_get(nr);

        camera.coordinates.left = x.x;
        camera.coordinates.right = x.y;
        camera.coordinates.down = y.y;
        camera.coordinates.up = y.x;

        camera.m_projection = fan::math::ortho<fan::mat4>(
          camera.coordinates.left,
          camera.coordinates.right,
          camera.coordinates.up,
          camera.coordinates.down,
          -fan::graphics::znearfar / 2,
          fan::graphics::znearfar / 2
        );

        camera.m_view[3][0] = 0;
        camera.m_view[3][1] = 0;
        camera.m_view[3][2] = 0;
        camera.m_view = camera.m_view.translate(camera.position);
        fan::vec3 position = camera.m_view.get_translation();
        constexpr fan::vec3 front(0, 0, 1);

        camera.m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);

        //auto it = gloco->m_viewport_resize_callback.GetNodeFirst();

        //while (it != gloco->m_viewport_resize_callback.dst) {

        //  gloco->m_viewport_resize_callback.StartSafeNext(it);

        //  resize_cb_data_t cbd;
        //  cbd.camera = this;
        //  cbd.position = get_position();
        //  cbd.size = get_camera_size();
        //  gloco->m_viewport_resize_callback[it].data(cbd);

        //  it = gloco->m_viewport_resize_callback.EndSafeNext();
        //}
      }

      fan::graphics::camera_nr_t camera_create(const fan::vec2& x, const fan::vec2& y) {
        fan::graphics::camera_nr_t nr = camera_create();
        camera_set_ortho(nr, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
        return nr;
      }

      fan::vec3 camera_get_position(fan::graphics::camera_nr_t nr) {
        return camera_get(nr).position;
      }

      void camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
        fan::graphics::context_camera_t& camera = camera_get(nr);
        camera.position = cp;


        camera.m_view[3][0] = 0;
        camera.m_view[3][1] = 0;
        camera.m_view[3][2] = 0;
        camera.m_view = camera.m_view.translate(camera.position);
        fan::vec3 position = camera.m_view.get_translation();
        constexpr fan::vec3 front(0, 0, 1);

        camera.m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
      }

      fan::vec2 camera_get_size(fan::graphics::camera_nr_t nr) {
        fan::graphics::context_camera_t& camera = camera_get(nr);
        return fan::vec2(std::abs(camera.coordinates.right - camera.coordinates.left), std::abs(camera.coordinates.down - camera.coordinates.up));
      }

      void camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
        fan::graphics::context_camera_t& camera = camera_get(nr);

        camera.m_projection = fan::math::perspective<fan::mat4>(fan::math::radians(fov), (f32_t)window_size.x / (f32_t)window_size.y, camera.znear, camera.zfar);

        camera.update_view();

        camera.m_view = camera.get_view_matrix();

        //auto it = gloco->m_viewport_resize_callback.GetNodeFirst();

        //while (it != gloco->m_viewport_resize_callback.dst) {

        //  gloco->m_viewport_resize_callback.StartSafeNext(it);

        //  resize_cb_data_t cbd;
        //  cbd.camera = this;
        //  cbd.position = get_position();
        //  cbd.size = get_camera_size();
        //  gloco->m_viewport_resize_callback[it].data(cbd);

        //  it = gloco->m_viewport_resize_callback.EndSafeNext();
        //}
      }

      void camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
        fan::graphics::context_camera_t& camera = camera_get(nr);
        camera.rotate_camera(offset);
        camera.m_view = camera.get_view_matrix();
      }

      //-----------------------------camera-----------------------------

      //-----------------------------viewport-----------------------------

      void viewport_set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
        VkViewport viewport{};
        viewport.x = viewport_position_.x;
        viewport.y = viewport_position_.y;
        viewport.width = viewport_size_.x;
        viewport.height = viewport_size_.y;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (!command_buffer_in_use) {
          VkResult result = vkGetFenceStatus(device, in_flight_fences[current_frame]);
          if (result == VK_NOT_READY) {
            vkDeviceWaitIdle(device);
          }

          if (vkBeginCommandBuffer(command_buffers[current_frame], &beginInfo) != VK_SUCCESS) {
            fan::throw_error("failed to begin recording command buffer!");
          }
        }
        vkCmdSetViewport(command_buffers[current_frame], 0, 1, &viewport);

        if (!command_buffer_in_use) {
          if (vkEndCommandBuffer(command_buffers[current_frame]) != VK_SUCCESS) {
            fan::throw_error("failed to record command buffer!");
          }
          command_buffer_in_use = false;
        }
      }

      fan::graphics::context_viewport_t& viewport_get(fan::graphics::viewport_nr_t nr) {
        return __fan_internal_viewport_list[nr];
      }

      void viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
        fan::graphics::context_viewport_t& viewport = viewport_get(nr);
        viewport.viewport_position = viewport_position_;
        viewport.viewport_size = viewport_size_;

        viewport_set(viewport_position_, viewport_size_, window_size);
      }

      fan::graphics::viewport_nr_t viewport_create()
      {
        auto nr = __fan_internal_viewport_list.NewNode();

        viewport_set(nr, 0, 1, 0);
        return nr;
      }

      void viewport_erase(fan::graphics::viewport_nr_t nr) {
        __fan_internal_viewport_list.Recycle(nr);
      }

      fan::vec2 viewport_get_position(fan::graphics::viewport_nr_t nr) {
        return viewport_get(nr).viewport_position;
      }

      fan::vec2 viewport_get_size(fan::graphics::viewport_nr_t nr) {
        return viewport_get(nr).viewport_size;
      }

      void viewport_zero(fan::graphics::viewport_nr_t nr) {
        auto& viewport = viewport_get(nr);
        viewport.viewport_position = 0;
        viewport.viewport_size = 0;
        viewport_set(0, 0, 0); // window_size not used
      }

      bool viewport_inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
        fan::graphics::context_viewport_t& viewport = viewport_get(nr);
        return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport.viewport_position + viewport.viewport_size / 2, viewport.viewport_size / 2);
      }

      bool viewport_inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
        fan::graphics::context_viewport_t& viewport = viewport_get(nr);
        return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport.viewport_size / 2, viewport.viewport_size / 2);
      }

      //-----------------------------viewport-----------------------------

      struct pipeline_t {

        struct properties_t {
          uint32_t descriptor_layout_count = 0;
          VkDescriptorSetLayout* descriptor_layout;
          fan::graphics::shader_nr_t shader;
          uint32_t push_constants_size = 0;
          uint32_t subpass = 0;

          uint32_t color_blend_attachment_count = 0;
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
          depthStencil.depthTestEnable = VK_TRUE;//p.enable_depth_test;
          depthStencil.depthWriteEnable = VK_TRUE;
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

      void open_no_window() {
        create_instance();
        setup_debug_messenger();
        create_instance();
        setup_debug_messenger();
        pick_physical_device();
        create_logical_device();
        create_command_pool();
        create_command_buffers();
        create_sync_objects();
      }
    #if defined(loco_window)
      void open(fan::window_t& window) {
        window.add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
          SwapChainRebuild = true;
          recreate_swap_chain(d.window, VK_ERROR_OUT_OF_DATE_KHR);
          });


        create_instance();

        setup_debug_messenger();
        create_surface(window);
        pick_physical_device();
        create_logical_device();

        create_swap_chain(window.get_size());

        create_command_pool();
        create_image_views();
        create_render_pass();
        create_framebuffers();
        create_command_buffers();
        create_sync_objects();
        descriptor_pool.open(*this);
      #if defined(fan_gui)
        ImGuiSetupVulkanWindow();
      #endif

        //{
        //  VkImageMemoryBarrier barrier = {};
        //  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        //  barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Layout after the first render pass
        //  barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Layout for the second render pass
        //  barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // Access in the first pass
        //  barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // Access in the second pass
        //  barrier.image = swap_chain; // Your color attachment image
        //  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // For color attachments
        //  barrier.subresourceRange.baseMipLevel = 0;
        //  barrier.subresourceRange.levelCount = 1;
        //  barrier.subresourceRange.baseArrayLayer = 0;
        //  barrier.subresourceRange.layerCount = 1;

        //  // Insert the pipeline barrier
        //  vkCmdPipelineBarrier(commandBuffer,
        //    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // Source stage
        //    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // Destination stage
        //    0, // No dependency flags
        //    0, nullptr, // No memory barriers
        //    0, nullptr, // No buffer barriers
        //    1, &barrier); // One image barrier
        //}

      }
#endif

      void close_vais(std::vector<fan::vulkan::vai_t>& v) {
        for (auto& e : v) {
          e.close(*this);
        }
      }

      void destroy_vulkan_soft() {
        vkDeviceWaitIdle(device);
        fan::vulkan::context_t& context = *this;
        {
          fan::graphics::shader_list_t::nrtra_t nrtra;
          fan::graphics::shader_list_t::nr_t nr;
          nrtra.Open(&__fan_internal_shader_list, &nr);
          while (nrtra.Loop(&__fan_internal_shader_list, &nr)) {
            shader_erase(nr, 0);
          }
          nrtra.Close(&__fan_internal_shader_list);
        }
        {
          fan::graphics::image_list_t::nrtra_t nrtra;
          fan::graphics::image_list_t::nr_t nr;
          nrtra.Open(&__fan_internal_image_list, &nr);
          while (nrtra.Loop(&__fan_internal_image_list, &nr)) {
            image_erase(nr, 0);
          }
          nrtra.Close(&__fan_internal_image_list);
        }

        close_vais(mainColorImageViews);
        close_vais(postProcessedColorImageViews);
        close_vais(depthImageViews);
        close_vais(downscaleImageViews1);
        close_vais(upscaleImageViews1);
        close_vais(vai_depth);

        for (size_t i = 0; i < max_frames_in_flight; i++) {
          if (render_finished_semaphores.size())
            vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
          if (image_available_semaphores.size())
            vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
          if (in_flight_fences.size())
            vkDestroyFence(device, in_flight_fences[i], nullptr);
        }

        vkDestroyRenderPass(device, render_pass, nullptr);
        vkDestroyCommandPool(device, command_pool, nullptr);

      #if fan_debug >= fan_debug_high
        if (supports_validation_layers) {
          DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
        }
      #endif
      }

    public:
      void gui_close() {
        vkFreeCommandBuffers(device, command_pool, command_buffers.size(), command_buffers.data());
        cleanup_swap_chain_dependencies();
        descriptor_pool.close(*this);
        destroy_vulkan_soft();
      #if defined(fan_gui)
        ImGui_ImplVulkanH_DestroyWindow(instance, device, &MainWindowData, nullptr);
      #endif

        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
      }

      void close() {
        vkDeviceWaitIdle(device);
        cleanup_swap_chain();
        vkDestroySurfaceKHR(instance, surface, nullptr);
        destroy_vulkan_soft();
        fan::vulkan::context_t& context = *this;
        {
          fan::graphics::camera_list_t::nrtra_t nrtra;
          fan::graphics::camera_list_t::nr_t nr;
          nrtra.Open(&__fan_internal_camera_list, &nr);
          while (nrtra.Loop(&__fan_internal_camera_list, &nr)) {
            camera_erase(nr);
          }
          nrtra.Close(&__fan_internal_camera_list);
        }
        {
          fan::graphics::shader_list_t::nrtra_t nrtra;
          fan::graphics::shader_list_t::nr_t nr;
          nrtra.Open(&__fan_internal_shader_list, &nr);
          while (nrtra.Loop(&__fan_internal_shader_list, &nr)) {
            shader_erase(nr);
          }
          nrtra.Close(&__fan_internal_shader_list);
        }
        {
          fan::graphics::image_list_t::nrtra_t nrtra;
          fan::graphics::image_list_t::nr_t nr;
          nrtra.Open(&__fan_internal_image_list, &nr);
          while (nrtra.Loop(&__fan_internal_image_list, &nr)) {
            image_erase(nr);
          }
          nrtra.Close(&__fan_internal_image_list);
        }
        {
          fan::graphics::viewport_list_t::nrtra_t nrtra;
          fan::graphics::viewport_list_t::nr_t nr;
          nrtra.Open(&__fan_internal_viewport_list, &nr);
          while (nrtra.Loop(&__fan_internal_viewport_list, &nr)) {
            viewport_erase(nr);
          }
          nrtra.Close(&__fan_internal_viewport_list);
        }
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
      }

      void cleanup_swap_chain_dependencies() {
        vkDeviceWaitIdle(device);
        close_vais(mainColorImageViews);
        close_vais(postProcessedColorImageViews);
        close_vais(depthImageViews);
        close_vais(downscaleImageViews1);
        close_vais(upscaleImageViews1);
        close_vais(vai_depth);

        for (auto framebuffer : swap_chain_framebuffers) {
          vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        for (auto& i : swap_chain_image_views) {
          vkDestroyImageView(device, i, nullptr);
        }
      }

      void cleanup_swap_chain() {
        cleanup_swap_chain_dependencies();
        if (swap_chain != VK_NULL_HANDLE) {
          vkDestroySwapchainKHR(device, swap_chain, nullptr);
          swap_chain = VK_NULL_HANDLE;
        }
      }

      void recreate_swap_chain_dependencies() {
        create_image_views();
        create_framebuffers();
      }

      // if swapchain changes, reque
      void update_swapchain_dependencies() {
        uint32_t imageCount =
        #if defined(fan_gui)
          MinImageCount + 1
        #else 
          min_image_count + 1
        #endif
          ;
        vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, nullptr);
        swap_chain_images.resize(imageCount);
        mainColorImageViews.resize(imageCount);
        postProcessedColorImageViews.resize(imageCount);
        depthImageViews.resize(imageCount);
        downscaleImageViews1.resize(imageCount);
        upscaleImageViews1.resize(imageCount);
        vai_depth.resize(imageCount);

        vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, swap_chain_images.data());
        recreate_swap_chain_dependencies();
      }

      void recreate_swap_chain(fan::window_t* window, VkResult err) {
        if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR || SwapChainRebuild) {
          int fb_width, fb_height;
          glfwGetFramebufferSize(*window, &fb_width, &fb_height);
          if (fb_width > 0 && fb_height > 0 &&
          #if defined(fan_gui)
            (
            #endif
              SwapChainRebuild
            #if defined(fan_gui)
              || MainWindowData.Width != fb_width ||
              MainWindowData.Height != fb_height)
          #endif
            )
          {

          #if defined(fan_gui)
            ImGui_ImplVulkan_SetMinImageCount(MinImageCount);
            ImGui_ImplVulkanH_CreateOrResizeWindow(instance, physical_device, device, &MainWindowData, queue_family, /*g_Allocator*/nullptr, fb_width, fb_height, MinImageCount);
            current_frame = MainWindowData.FrameIndex = 0;
          #endif
            SwapChainRebuild = false;
          #if defined(fan_gui)
            swap_chain = MainWindowData.Swapchain;
          #endif
            swap_chain_size = fan::vec2(fb_width, fb_height);
            update_swapchain_dependencies();
          }
        }
        else if (err != VK_SUCCESS) {
          throw std::runtime_error("failed to present swap chain image");
        }
      }

      void recreate_swap_chain(const fan::vec2i& window_size) {
        vkDeviceWaitIdle(device);
        cleanup_swap_chain();
        create_swap_chain(window_size);
        recreate_swap_chain_dependencies();
        // need to recreate some imgui's swapchain dependencies
      #if defined(fan_gui)
        MainWindowData.Swapchain = swap_chain;
      #endif
      }

      void create_instance() {

      #if fan_debug >= fan_debug_high
        if (!check_validation_layer_support()) {
          fan::print_warning("validation layers not supported");
          supports_validation_layers = false;
        }
      #endif

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "application";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 1, 0);
        appInfo.pEngineName = "fan";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 1, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = get_required_extensions();
        createInfo.enabledExtensionCount = extensions.size();
        std::vector<char*> extension_names(extensions.size() + 1);

        for (uint32_t i = 0; i < extensions.size(); ++i) {
          extension_names[i] = new char[extensions[i].size() + 1];
          memcpy(extension_names[i], extensions[i].data(), extensions[i].size() + 1);
        }
        createInfo.ppEnabledExtensionNames = extension_names.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
      #if fan_debug >= fan_debug_high
        if (supports_validation_layers) {
          createInfo.enabledLayerCount = validationLayers.size();
          createInfo.ppEnabledLayerNames = validationLayers.data();

          populate_debug_messenger_create_info(debugCreateInfo);
          createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }

      #endif

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
          throw std::runtime_error("failed to create instance!");
        }
      }

      void populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& create_info) {
        create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        create_info.pfnUserCallback = debug_callback;
      }

      void setup_debug_messenger() {
      #if fan_debug < fan_debug_high
        return;
      #endif

        if (!supports_validation_layers) {
          return;
        }

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populate_debug_messenger_create_info(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debug_messenger) != VK_SUCCESS) {
          throw std::runtime_error("failed to set up debug messenger!");
        }
      }

#if defined(loco_window)
      void create_surface(GLFWwindow* window) {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
          throw std::runtime_error("failed to create window surface!");
        }
      }
#endif

      void pick_physical_device() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
          throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
          if (is_device_suitable(device)) {
            physical_device = device;
            break;
          }
        }

        if (physical_device == VK_NULL_HANDLE) {
          throw std::runtime_error("failed to find a suitable GPU!");
        }
      }

      void create_logical_device() {
        queue_family_indices_t indices = find_queue_families(physical_device);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
          indices.graphics_family.value(),
      #if defined(loco_window)
          indices.present_family.value()
      #endif
        };
        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
          VkDeviceQueueCreateInfo queueCreateInfo{};
          queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
          queueCreateInfo.queueFamilyIndex = queueFamily;
          queueCreateInfo.queueCount = 1;
          queueCreateInfo.pQueuePriorities = &queuePriority;
          queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceProperties2 deviceProperties{};
        deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        vkGetPhysicalDeviceProperties2(physical_device, &deviceProperties);

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;

        //deviceFeatures.vertexPipelineStoresAndAtomics = VK_TRUE;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = queueCreateInfos.size();
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        if (deviceProperties.properties.apiVersion >= VK_API_VERSION_1_2) {
          // Use Vulkan 1.2 features
          VkPhysicalDeviceVulkan12Features vulkan12Features{};
          vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
          vulkan12Features.runtimeDescriptorArray = VK_TRUE;
          vulkan12Features.descriptorIndexing = VK_TRUE;
          vulkan12Features.descriptorBindingVariableDescriptorCount = VK_TRUE;
          vulkan12Features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;

          VkPhysicalDeviceFeatures2 deviceFeatures2{};
          deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
          deviceFeatures2.features = deviceFeatures;
          deviceFeatures2.pNext = &vulkan12Features;

          createInfo.pNext = &deviceFeatures2;
          createInfo.pEnabledFeatures = nullptr;
        }
        else {
          VkPhysicalDeviceDescriptorIndexingFeaturesEXT indexingFeatures{};
          indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
          indexingFeatures.runtimeDescriptorArray = VK_TRUE;
          indexingFeatures.descriptorBindingVariableDescriptorCount = VK_TRUE;
          indexingFeatures.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;

          VkPhysicalDeviceFeatures2 deviceFeatures2{};
          deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
          deviceFeatures2.features = deviceFeatures;
          deviceFeatures2.pNext = &indexingFeatures;

          createInfo.pNext = &deviceFeatures2;
          createInfo.pEnabledFeatures = nullptr;
        }

        createInfo.enabledExtensionCount = deviceExtensions.size();
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

      #if fan_debug >= fan_debug_high
        if (supports_validation_layers) {
          createInfo.enabledLayerCount = validationLayers.size();
          createInfo.ppEnabledLayerNames = validationLayers.data();
        }
      #endif

        if (vkCreateDevice(physical_device, &createInfo, nullptr, &device) != VK_SUCCESS) {
          throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphics_family.value(), 0, &graphics_queue);
      #if defined(loco_window)
        vkGetDeviceQueue(device, indices.present_family.value(), 0, &present_queue);
      #endif
      }

    #if defined(loco_window)
      void create_swap_chain(const fan::vec2ui& framebuffer_size) {
        swap_chain_support_details_t swapChainSupport = query_swap_chain_support(physical_device);

        surface_format = choose_swap_surface_format(swapChainSupport.formats);
        present_mode = choose_swap_present_mode(swapChainSupport.present_modes);
        VkExtent2D extent = choose_swap_extent(framebuffer_size, swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        min_image_count = swapChainSupport.capabilities.minImageCount;
        image_count = imageCount;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
          imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surface_format.format;
        createInfo.imageColorSpace = surface_format.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

        queue_family_indices_t indices = find_queue_families(physical_device);
        queue_family = indices.graphics_family.value();
        uint32_t queueFamilyIndices[] = { indices.graphics_family.value(), indices.present_family.value() };

        if (indices.graphics_family != indices.present_family) {
          createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
          createInfo.queueFamilyIndexCount = 2;
          createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
          createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = present_mode;
        createInfo.clipped = VK_TRUE;
        //createInfo.imageUsage = ;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swap_chain) != VK_SUCCESS) {
          throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, nullptr);
        swap_chain_images.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, swap_chain_images.data());

        swap_chain_image_format = surface_format.format;
        swap_chain_size = fan::vec2(extent.width, extent.height);
      }
    #endif

      VkImageView create_image_view(VkImage image, VkFormat format, VkImageAspectFlags aspect_flags) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspect_flags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;


        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
          throw std::runtime_error("failed to create texture image view!");
        }

        return imageView;
      }

      void create_image_views() {
        swap_chain_image_views.resize(swap_chain_images.size());

        fan::vulkan::vai_t::properties_t vp;
        vp.format = swap_chain_image_format;
        vp.swap_chain_size = swap_chain_size;
        vp.usage_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        vp.aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT;

        // Resize vectors to hold image views for each swap chain image
        mainColorImageViews.resize(swap_chain_image_views.size());
        postProcessedColorImageViews.resize(swap_chain_image_views.size());
        depthImageViews.resize(swap_chain_image_views.size());
        downscaleImageViews1.resize(swap_chain_image_views.size());
        upscaleImageViews1.resize(swap_chain_image_views.size());

        for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
          mainColorImageViews[i].open(*this, vp);
          mainColorImageViews[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

          postProcessedColorImageViews[i].open(*this, vp);
          postProcessedColorImageViews[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

          fan::vulkan::vai_t::properties_t depth_vp = vp;
          depth_vp.aspect_flags = VK_IMAGE_ASPECT_DEPTH_BIT;
          depth_vp.format = find_depth_format();
          depth_vp.usage_flags = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
          depthImageViews[i].open(*this, depth_vp);
          depthImageViews[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, depth_vp.aspect_flags);

          downscaleImageViews1[i].open(*this, vp);
          downscaleImageViews1[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

          upscaleImageViews1[i].open(*this, vp);
          upscaleImageViews1[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        }

        for (uint32_t i = 0; i < swap_chain_images.size(); i++) {
          swap_chain_image_views[i] = create_image_view(swap_chain_images[i], swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT);
        }
      }


      void create_render_pass() {
        //--------------attachment description--------------

        VkAttachmentDescription mainColorAttachment{};
        mainColorAttachment.format = swap_chain_image_format;
        mainColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        mainColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        mainColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        mainColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        mainColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        mainColorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        mainColorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // For post-process input

        VkAttachmentDescription postProcessedColorAttachment{};
        postProcessedColorAttachment.format = swap_chain_image_format;
        postProcessedColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        postProcessedColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR /*shapes_top ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE*/;
        postProcessedColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        postProcessedColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        postProcessedColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        postProcessedColorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        postProcessedColorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = find_depth_format();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        //--------------attachment description--------------

        VkAttachmentReference mainSceneColorRef{};
        mainSceneColorRef.attachment = 0;
        mainSceneColorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference postProcessInputRef{};
        postProcessInputRef.attachment = 0;
        postProcessInputRef.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference postProcessOutputRef{};
        postProcessOutputRef.attachment = 1;
        postProcessOutputRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthRef{};
        depthRef.attachment = 2;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription mainSceneSubpass{};
        mainSceneSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        mainSceneSubpass.colorAttachmentCount = 1;
        mainSceneSubpass.pColorAttachments = &mainSceneColorRef;
        mainSceneSubpass.pDepthStencilAttachment = &depthRef;

        VkSubpassDescription postProcessSubpass{};
        postProcessSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        postProcessSubpass.inputAttachmentCount = 1;
        postProcessSubpass.pInputAttachments = &postProcessInputRef;
        postProcessSubpass.colorAttachmentCount = 1;
        postProcessSubpass.pColorAttachments = &postProcessOutputRef;

        VkSubpassDependency extToMainDep{};
        extToMainDep.srcSubpass = VK_SUBPASS_EXTERNAL;
        extToMainDep.dstSubpass = 0;
        extToMainDep.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        extToMainDep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        extToMainDep.srcAccessMask = 0;
        extToMainDep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkSubpassDependency mainToPostDep{};
        mainToPostDep.srcSubpass = 0;
        mainToPostDep.dstSubpass = 1;
        mainToPostDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        mainToPostDep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        mainToPostDep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        mainToPostDep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkSubpassDependency postToExtDep{};
        postToExtDep.srcSubpass = 1;
        postToExtDep.dstSubpass = VK_SUBPASS_EXTERNAL;
        postToExtDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        postToExtDep.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        postToExtDep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        postToExtDep.dstAccessMask = 0;

        VkAttachmentDescription attachments[] = {
          mainColorAttachment,
          postProcessedColorAttachment,
          depthAttachment
        };

        VkSubpassDescription subpasses[] = {
          mainSceneSubpass,
          postProcessSubpass
        };

        VkSubpassDependency dependencies[] = {
          extToMainDep,
          mainToPostDep,
          postToExtDep
        };

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = std::size(attachments);
        renderPassInfo.pAttachments = attachments;
        renderPassInfo.subpassCount = std::size(subpasses);
        renderPassInfo.pSubpasses = subpasses;
        renderPassInfo.dependencyCount = std::size(dependencies);
        renderPassInfo.pDependencies = dependencies;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &render_pass) != VK_SUCCESS) {
          throw std::runtime_error("failed to create render pass");
        }
      }

      void create_framebuffers() {
        swap_chain_framebuffers.resize(swap_chain_image_views.size());

        for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
          VkImageView attachments[] = {
            mainColorImageViews[i].image_view,
            swap_chain_image_views[i],
            depthImageViews[i].image_view,
          };

          VkFramebufferCreateInfo framebufferInfo{};
          framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
          framebufferInfo.renderPass = render_pass;
          framebufferInfo.attachmentCount = std::size(attachments);
          framebufferInfo.pAttachments = attachments;
          framebufferInfo.width = swap_chain_size.x;
          framebufferInfo.height = swap_chain_size.y;
          framebufferInfo.layers = 1;

          if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swap_chain_framebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
          }
        }
      }

      void create_command_pool() {
        queue_family_indices_t queueFamilyIndices = find_queue_families(physical_device);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphics_family.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &command_pool) != VK_SUCCESS) {
          throw std::runtime_error("failed to create graphics command pool!");
        }
}

      VkFormat find_supported_format(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
        for (VkFormat format : candidates) {
          VkFormatProperties props;
          vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

          if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
          }
          else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
          }
        }

        throw std::runtime_error("failed to find supported format!");
      }

      VkFormat find_depth_format() {
        return find_supported_format(
          { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
          VK_IMAGE_TILING_OPTIMAL,
          VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
      }

      bool has_stencil_component(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
      }

      void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        fan::vulkan::validate(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = find_memory_type(memRequirements.memoryTypeBits, properties);

        fan::vulkan::validate(vkAllocateMemory(device, &allocInfo, nullptr, &buffer_memory));
        fan::vulkan::validate(vkBindBufferMemory(device, buffer, buffer_memory, 0));
      }

      VkCommandBuffer begin_single_time_commands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = command_pool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
      }

      void end_single_time_commands(VkCommandBuffer command_buffer) {
        vkEndCommandBuffer(command_buffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &command_buffer;

        vkQueueSubmit(graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphics_queue);

        vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
      }

      void copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = begin_single_time_commands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, src_buffer, dst_buffer, 1, &copyRegion);

        end_single_time_commands(commandBuffer);
      }

      uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
          if ((type_filter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
          }
        }

        fan::throw_error("failed to find suitable memory type!");
        return {};
      }

      void create_command_buffers() {
        command_buffers.resize(max_frames_in_flight);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = command_pool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)command_buffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, command_buffers.data()) != VK_SUCCESS) {
          throw std::runtime_error("failed to allocate command buffers!");
        }
      }


      void bind_draw(
        const fan::vulkan::context_t::pipeline_t& pipeline,
        uint32_t descriptor_count,
        VkDescriptorSet* descriptor_sets) {
        vkCmdBindPipeline(command_buffers[current_frame], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.m_pipeline);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent.width = swap_chain_size.x;
        scissor.extent.height = swap_chain_size.y;
        vkCmdSetScissor(command_buffers[current_frame], 0, 1, &scissor);

        vkCmdBindDescriptorSets(
          command_buffers[current_frame],
          VK_PIPELINE_BIND_POINT_GRAPHICS,
          pipeline.m_layout,
          0,
          descriptor_count,
          descriptor_sets,
          0,
          nullptr
        );
      }

      // assumes things are already bound
      void bindless_draw(
        uint32_t vertex_count,
        uint32_t instance_count,
        uint32_t first_instance) {
        vkCmdDraw(command_buffers[current_frame], vertex_count, instance_count, 0, first_instance);
      }

      void draw(
        uint32_t vertex_count,
        uint32_t instance_count,
        uint32_t first_instance,
        const fan::vulkan::context_t::pipeline_t& pipeline,
        uint32_t descriptor_count,
        VkDescriptorSet* descriptor_sets
      ) {
        bind_draw(pipeline, descriptor_count, descriptor_sets);
        bindless_draw(vertex_count, instance_count, first_instance);
      }

      void create_sync_objects() {
        image_available_semaphores.resize(max_frames_in_flight);
        render_finished_semaphores.resize(max_frames_in_flight);
        in_flight_fences.resize(max_frames_in_flight);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < max_frames_in_flight; i++) {
          if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &image_available_semaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
          }
        }
      }

       //----------------------------------------------imgui stuff----------------------------------------------
      bool                     SwapChainRebuild = false;
      #if defined(fan_gui)
      ImGui_ImplVulkanH_Window MainWindowData;
      uint32_t                 MinImageCount = 2;

      void ImGuiSetupVulkanWindow() {
        MainWindowData.Surface = surface;
        MainWindowData.SurfaceFormat = surface_format;
        MainWindowData.Swapchain = swap_chain;
        MainWindowData.PresentMode = present_mode;
        MainWindowData.ClearEnable = shapes_top;

        IM_ASSERT(MinImageCount >= 2);
        ImGui_ImplVulkanH_CreateOrResizeWindow(instance, physical_device, device, &MainWindowData, queue_family, /*g_Allocator*/nullptr, swap_chain_size.x, swap_chain_size.y, MinImageCount);
        swap_chain = MainWindowData.Swapchain;
        update_swapchain_dependencies();
      }

      void ImGuiFrameRender(VkResult next_image_khr_err, fan::color clear_color) {
        ImGui_ImplVulkanH_Window* wd = &MainWindowData;
        VkResult err = next_image_khr_err;
        if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
          SwapChainRebuild = true;
        if (err == VK_ERROR_OUT_OF_DATE_KHR)
          return;
        if (err != VK_SUBOPTIMAL_KHR)
          fan::vulkan::validate(err);

        wd->FrameIndex = image_index;

        ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];

        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass = wd->RenderPass;
        info.framebuffer = fd->Framebuffer;
        info.renderArea.extent.width = wd->Width;
        info.renderArea.extent.height = wd->Height;

        vkCmdBeginRenderPass(command_buffers[current_frame], &info, VK_SUBPASS_CONTENTS_INLINE);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffers[current_frame]);

        vkCmdEndRenderPass(command_buffers[current_frame]);
      }
      //----------------------------------------------imgui stuff----------------------------------------------
#endif
      VkResult end_render()  {
        //// render_fullscreen_pl loco fbo?
        if (!command_buffer_in_use) {
          return VK_SUCCESS;
        }
        if (vkEndCommandBuffer(command_buffers[current_frame]) != VK_SUCCESS) {
          fan::throw_error("failed to record command buffer!");
        }

        command_buffer_in_use = false;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { image_available_semaphores[current_frame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &command_buffers[current_frame];

        VkSemaphore signalSemaphores[] = { render_finished_semaphores[current_frame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphics_queue, 1, &submitInfo, in_flight_fences[current_frame]) != VK_SUCCESS) {
          throw std::runtime_error("failed to submit draw command buffer!");
        }
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swap_chain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &image_index;
        auto result = vkQueuePresentKHR(present_queue, &presentInfo);

        current_frame = (current_frame + 1) % max_frames_in_flight;
        return result;
      }

      void begin_compute_shader() {
        //?
        //vkWaitForFences(device, 1, &inFlightFences[current_frame], VK_TRUE, UINT64_MAX);

        vkResetFences(device, 1, &in_flight_fences[current_frame]);

        vkResetCommandBuffer(command_buffers[current_frame], /*VkCommandBufferResetFlagBits*/ 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(command_buffers[current_frame], &beginInfo) != VK_SUCCESS) {
          fan::throw_error("failed to begin recording command buffer!");
        }

        command_buffer_in_use = true;
      }

      void end_compute_shader() {
        if (vkEndCommandBuffer(command_buffers[current_frame]) != VK_SUCCESS) {
          fan::throw_error("failed to record command buffer!");
        }

        command_buffer_in_use = false;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &command_buffers[current_frame];

        if (vkQueueSubmit(graphics_queue, 1, &submitInfo, in_flight_fences[current_frame]) != VK_SUCCESS) {
          throw std::runtime_error("failed to submit draw command buffer!");
        }
      }

      VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats) {
        for (const auto& availableFormat : available_formats) {
          // VK_FORMAT_B8G8R8A8_SRGB

          if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
          }
        }

        return available_formats[0];
      }

      VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes) {
        for (const auto& available_present_mode : available_present_modes) {
          if (available_present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR && !vsync) {
            return VK_PRESENT_MODE_IMMEDIATE_KHR;
          }
          else if (available_present_mode == VK_PRESENT_MODE_FIFO_KHR && vsync) {
            return VK_PRESENT_MODE_FIFO_KHR;
          }
        }

        return available_present_modes[0];
      }


      VkExtent2D choose_swap_extent(const fan::vec2ui& framebuffer_size, const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
          return capabilities.currentExtent;
        }
        else {
          VkExtent2D actualExtent = {
            framebuffer_size.x,
            framebuffer_size.y
          };

          actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
          actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

          return actualExtent;
        }
      }

      swap_chain_support_details_t query_swap_chain_support(VkPhysicalDevice device) {
        swap_chain_support_details_t details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
          details.formats.resize(formatCount);
          vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
          details.present_modes.resize(presentModeCount);
          vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.present_modes.data());
        }

        return details;
      }

      bool is_device_suitable(VkPhysicalDevice device) {
        queue_family_indices_t indices = find_queue_families(device);

        bool extensionsSupported = check_device_extension_support(device);

        bool swapChainAdequate
        #if defined(loco_window)
          = false;
        if (extensionsSupported) {
          swap_chain_support_details_t swapChainSupport = query_swap_chain_support(device);
          swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.present_modes.empty();
        }
      #else
          = true;
      #endif

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        return indices.is_complete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
      }

      bool check_device_extension_support(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
          requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
      }

      queue_family_indices_t find_queue_families(VkPhysicalDevice device) {
        queue_family_indices_t indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
          if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphics_family = i;
          }

          VkBool32 presentSupport = false;

        #if defined(loco_window)
          vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

          if (presentSupport) {
            indices.present_family = i;
          }
        #endif

          if (indices.is_complete()) {
            break;
          }

          i++;
        }

        return indices;
      }

      std::vector<std::string> get_required_extensions() {

        uint32_t extensions_count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensions_count, nullptr);
        if (extensions_count == 0) {
          throw std::runtime_error("Could not get the number of Instance extensions.");
        }

        std::vector<VkExtensionProperties> available_extensions;

        available_extensions.resize(extensions_count);

        vkEnumerateInstanceExtensionProperties(nullptr, &extensions_count, &available_extensions[0]);

        if (extensions_count == 0) {
          throw std::runtime_error("Could not enumerate Instance extensions.");
        }

        std::vector<std::string> extension_str(available_extensions.size());

        for (int i = 0; i < available_extensions.size(); i++) {
          extension_str[i] = available_extensions[i].extensionName;
        }

      #if fan_debug >= fan_debug_high
        if (supports_validation_layers) {
          extension_str.push_back((char*)VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
      #endif

        return extension_str;
      }


      bool check_validation_layer_support() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
          bool layerFound = false;

          for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
              layerFound = true;
              break;
            }
          }

          if (!layerFound) {
            return false;
          }
        }

        return true;
      }

      static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData
      ) {
        if (pCallbackData->pMessageIdName && std::string(pCallbackData->pMessageIdName) == "Loader Message") {
          return VK_FALSE;
        }
        fan::print("validation layer:", pCallbackData->pMessage);
        // system("pause");
      //  exit(0);

        return VK_FALSE;
      }

#if defined(loco_window)
void set_vsync(fan::window_t* window, bool flag) {
  vsync = flag;
  recreate_swap_chain(window->get_size());
}
#endif

      VkInstance instance;
      VkDebugUtilsMessengerEXT debug_messenger;
      VkSurfaceKHR surface;

      VkPhysicalDevice physical_device = VK_NULL_HANDLE;
      VkDevice device;

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
      uint32_t queue_family = -1;
      uint32_t min_image_count = 0;
      uint32_t image_count = 0;
      
      std::vector<VkCommandBuffer> command_buffers;

      std::vector<VkSemaphore> image_available_semaphores;
      std::vector<VkSemaphore> render_finished_semaphores;
      std::vector<VkFence> in_flight_fences;
      uint32_t current_frame = 0;

      bool enable_clear = true;
      bool shapes_top = false;

      bool vsync = true;
      uint32_t image_index;

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
      uint32_t get_draw_mode(uint8_t draw_mode) {
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

void fan::vulkan::image_create(const fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = image_size.x;
  imageInfo.extent.height = image_size.y;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateImage(context.device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
    throw std::runtime_error("failed to create image!");
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(context.device, image, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = context.find_memory_type(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(context.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate image memory!");
  }

  vkBindImageMemory(context.device, image, imageMemory, 0);
}

auto fan::graphics::format_converter::image_global_to_vulkan(const fan::graphics::image_load_properties_t& p) {
  return fan::vulkan::context_t::image_load_properties_t{
    .visual_output = global_to_vulkan_address_mode(p.visual_output),
    .format = global_to_vulkan_format(p.format),
    .min_filter = global_to_vulkan_filter(p.min_filter),
    .mag_filter = global_to_vulkan_filter(p.mag_filter),
  };
}

void fan::vulkan::vai_t::open(fan::vulkan::context_t& context, const properties_t& p) {
  fan::vulkan::image_create(
    context,
    p.swap_chain_size,
    p.format,
    VK_IMAGE_TILING_OPTIMAL,
    p.usage_flags,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    image,
    memory
  );
  image_view = context.create_image_view(image, p.format, p.aspect_flags);
  format = p.format;
}
void fan::vulkan::vai_t::close(fan::vulkan::context_t& context) {
  if (image_view != 0) {
    vkDestroyImageView(context.device, image_view, nullptr);
    image_view = 0;
  }
  if (image != 0) {
    vkDestroyImage(context.device, image, nullptr);
    image = 0;
  }
  if (memory != 0) {
    vkFreeMemory(context.device, memory, nullptr);
    memory = 0;
  }
}

fan::graphics::context_functions_t fan::graphics::get_vk_context_functions() {
	fan::graphics::context_functions_t cf;
  cf.shader_create = [](void* context) { 
    return ((fan::vulkan::context_t*)context)->shader_create();
  }; 
  cf.shader_get = [](void* context, fan::graphics::shader_nr_t nr) { 
    return (void*)&((fan::vulkan::context_t*)context)->shader_get(nr);
  }; 
  cf.shader_erase = [](void* context, fan::graphics::shader_nr_t nr) { 
    ((fan::vulkan::context_t*)context)->shader_erase(nr); 
  }; 
  cf.shader_use = [](void* context, fan::graphics::shader_nr_t nr) { 
    ((fan::vulkan::context_t*)context)->shader_use(nr); 
  }; 
  cf.shader_set_vertex = [](void* context, fan::graphics::shader_nr_t nr, const std::string& vertex_code) { 
    ((fan::vulkan::context_t*)context)->shader_set_vertex(nr, vertex_code); 
  }; 
  cf.shader_set_fragment = [](void* context, fan::graphics::shader_nr_t nr, const std::string& fragment_code) { 
    ((fan::vulkan::context_t*)context)->shader_set_fragment(nr, fragment_code); 
  }; 
  cf.shader_compile = [](void* context, fan::graphics::shader_nr_t nr) { 
    return ((fan::vulkan::context_t*)context)->shader_compile(nr); 
  }; 
    /*image*/
  cf.image_create = [](void* context) {
    return ((fan::vulkan::context_t*)context)->image_create();
  }; 
  cf.image_get_handle = [](void* context, fan::graphics::image_nr_t nr) { 
    return (uint64_t)((fan::vulkan::context_t*)context)->image_get_handle(nr); 
  }; 
  cf.image_get = [](void* context, fan::graphics::image_nr_t nr) {
    return (void*)&((fan::vulkan::context_t*)context)->image_get(nr);
  }; 
  cf.image_erase = [](void* context, fan::graphics::image_nr_t nr) { 
    ((fan::vulkan::context_t*)context)->image_erase(nr); 
  }; 
  cf.image_bind = [](void* context, fan::graphics::image_nr_t nr) { 
    ((fan::vulkan::context_t*)context)->image_bind(nr); 
  }; 
  cf.image_unbind = [](void* context, fan::graphics::image_nr_t nr) { 
    ((fan::vulkan::context_t*)context)->image_unbind(nr); 
  }; 
  cf.image_get_settings = [](void* context, fan::graphics::image_nr_t nr) -> fan::graphics::image_load_properties_t& {
    return ((fan::vulkan::context_t*)context)->image_get_settings(nr);
  };
  cf.image_set_settings = [](void* context, fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings) { 
    ((fan::vulkan::context_t*)context)->image_bind(nr);
    ((fan::vulkan::context_t*)context)->image_set_settings(fan::graphics::format_converter::image_global_to_vulkan(settings));
  }; 
  cf.image_load_info = [](void* context, const fan::image::info_t& image_info) { 
    return ((fan::vulkan::context_t*)context)->image_load(image_info);
  }; 
  cf.image_load_info_props = [](void* context, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) { 
    return ((fan::vulkan::context_t*)context)->image_load(image_info, fan::graphics::format_converter::image_global_to_vulkan(p));
  }; 
  cf.image_load_path = [](void* context, const std::string& path, const std::source_location& callers_path) { 
    return ((fan::vulkan::context_t*)context)->image_load(path, callers_path);
  }; 
  cf.image_load_path_props = [](void* context, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path) { 
    return ((fan::vulkan::context_t*)context)->image_load(path, fan::graphics::format_converter::image_global_to_vulkan(p));
  }; 
  cf.image_load_colors = [](void* context, fan::color* colors, const fan::vec2ui& size_) { 
    return ((fan::vulkan::context_t*)context)->image_load(colors, size_);
  }; 
  cf.image_load_colors_props = [](void* context, fan::color* colors, const fan::vec2ui& size_, const fan::graphics::image_load_properties_t& p) { 
    return ((fan::vulkan::context_t*)context)->image_load(colors, size_, fan::graphics::format_converter::image_global_to_vulkan(p));
  }; 
  cf.image_unload = [](void* context, fan::graphics::image_nr_t nr) { 
    ((fan::vulkan::context_t*)context)->image_unload(nr); 
  }; 
  cf.create_missing_texture = [](void* context) { 
    return ((fan::vulkan::context_t*)context)->create_missing_texture();
  }; 
  cf.create_transparent_texture = [](void* context) { 
    return ((fan::vulkan::context_t*)context)->create_transparent_texture();
  }; 
  cf.image_reload_image_info = [](void* context, fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) { 
    return ((fan::vulkan::context_t*)context)->image_reload(nr, image_info); 
  }; 
  cf.image_reload_image_info_props = [](void* context, fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) { 
    return ((fan::vulkan::context_t*)context)->image_reload(nr, image_info, fan::graphics::format_converter::image_global_to_vulkan(p)); 
  }; 
  cf.image_reload_path = [](void* context, fan::graphics::image_nr_t nr, const std::string& path, const std::source_location& callers_path) { 
    return ((fan::vulkan::context_t*)context)->image_reload(nr, path, callers_path); 
  }; 
  cf.image_reload_path_props = [](void* context, fan::graphics::image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path) { 
    return ((fan::vulkan::context_t*)context)->image_reload(nr, path, fan::graphics::format_converter::image_global_to_vulkan(p), callers_path); 
  };
  cf.image_create_color = [](void* context, const fan::color& color) { 
    return ((fan::vulkan::context_t*)context)->image_create(color);
  }; 
  cf.image_create_color_props = [](void* context, const fan::color& color, const fan::graphics::image_load_properties_t& p) { 
    return ((fan::vulkan::context_t*)context)->image_create(color, fan::graphics::format_converter::image_global_to_vulkan(p));
  };
  /*camera*/
  cf.camera_create = [](void* context) {
    return ((fan::vulkan::context_t*)context)->camera_create();
  };
  cf.camera_get = [](void* context, fan::graphics::camera_nr_t nr) -> decltype(auto) {
    return ((fan::vulkan::context_t*)context)->camera_get(nr);
  };
  cf.camera_erase = [](void* context, camera_nr_t nr) { 
    ((fan::vulkan::context_t*)context)->camera_erase(nr); 
  };
  cf.camera_create_params = [](void* context, const fan::vec2& x, const fan::vec2& y) {
    return ((fan::vulkan::context_t*)context)->camera_create(x, y);
  };
  cf.camera_get_position = [](void* context, camera_nr_t nr) { 
    return ((fan::vulkan::context_t*)context)->camera_get_position(nr); 
  };
  cf.camera_set_position = [](void* context, camera_nr_t nr, const fan::vec3& cp) { 
    ((fan::vulkan::context_t*)context)->camera_set_position(nr, cp); 
  };
  cf.camera_get_size = [](void* context, camera_nr_t nr) { 
    return ((fan::vulkan::context_t*)context)->camera_get_size(nr); 
  };
  cf.camera_set_ortho = [](void* context, camera_nr_t nr, fan::vec2 x, fan::vec2 y) { 
    ((fan::vulkan::context_t*)context)->camera_set_ortho(nr, x, y); 
  };
  cf.camera_set_perspective = [](void* context, camera_nr_t nr, f32_t fov, const fan::vec2& window_size) { 
    ((fan::vulkan::context_t*)context)->camera_set_perspective(nr, fov, window_size); 
  };
  cf.camera_rotate = [](void* context, camera_nr_t nr, const fan::vec2& offset) { 
    ((fan::vulkan::context_t*)context)->camera_rotate(nr, offset); 
  };
  /*viewport*/
  cf.viewport_create = [](void* context) {
    return ((fan::vulkan::context_t*)context)->viewport_create();
  };
  cf.viewport_get = [](void* context, viewport_nr_t nr) -> fan::graphics::context_viewport_t&{ 
    return ((fan::vulkan::context_t*)context)->viewport_get(nr);
  };
  cf.viewport_erase = [](void* context, viewport_nr_t nr) { 
    ((fan::vulkan::context_t*)context)->viewport_erase(nr); 
  };
  cf.viewport_get_position = [](void* context, viewport_nr_t nr) { 
    return ((fan::vulkan::context_t*)context)->viewport_get_position(nr); 
  };
  cf.viewport_get_size = [](void* context, viewport_nr_t nr) { 
    return ((fan::vulkan::context_t*)context)->viewport_get_size(nr); 
  };
  cf.viewport_set = [](void* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) { 
    ((fan::vulkan::context_t*)context)->viewport_set(viewport_position_, viewport_size_, window_size); 
  };
  cf.viewport_set_nr = [](void* context, viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) { 
    ((fan::vulkan::context_t*)context)->viewport_set(nr, viewport_position_, viewport_size_, window_size); 
  };
  cf.viewport_zero = [](void* context, viewport_nr_t nr) { 
    ((fan::vulkan::context_t*)context)->viewport_zero(nr); 
  };
  cf.viewport_inside = [](void* context, viewport_nr_t nr, const fan::vec2& position) { 
    return ((fan::vulkan::context_t*)context)->viewport_inside(nr, position); 
  };
  cf.viewport_inside_wir = [](void* context, viewport_nr_t nr, const fan::vec2& position) { 
    return ((fan::vulkan::context_t*)context)->viewport_inside_wir(nr, position); 
  };
  return cf;
}
#endif