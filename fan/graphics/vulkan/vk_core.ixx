module;

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
#include <vk_mem_alloc.h>
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

#include <fan/utility.h>

export module fan.graphics.vulkan.core;
import std;


export import :types;
export import :vai;

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

inline constexpr auto validationLayers = std::to_array<const char*>({
  "VK_LAYER_KHRONOS_validation"
});

inline constexpr auto deviceExtensions = std::to_array<const char*>({
  VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
#if defined(ENABLE_RAYTRACING_DEPENDENCIES)
  VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
  VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
  VK_KHR_SPIRV_1_4_EXTENSION_NAME,
  VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
  VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
  VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
#endif
});


VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);

export namespace fan {
  namespace vulkan {
    struct context_t;
    void image_create(const fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VmaAllocation& allocation);
  }
}

namespace fan::graphics::format_converter {
  VkFormat global_to_vulkan_format(std::uintptr_t format);
  VkSamplerAddressMode global_to_vulkan_address_mode(std::uintptr_t mode);
  VkFilter global_to_vulkan_filter(std::uintptr_t filter);

  std::uint32_t vulkan_to_global_format(VkFormat format);
  std::uint32_t vulkan_to_global_address_mode(VkSamplerAddressMode mode);
  std::uint32_t vulkan_to_global_filter(VkFilter filter);

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

    struct context_t {

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
        descriptor_t(descriptor_t&& other) noexcept { *this = std::move(other); }
        descriptor_t& operator=(descriptor_t&& other) noexcept {
          m_properties = std::move(other.m_properties);
          m_layout = other.m_layout;
          std::memcpy(m_descriptor_set, other.m_descriptor_set, sizeof(m_descriptor_set));
          other.m_layout = VK_NULL_HANDLE;
          std::memset(other.m_descriptor_set, 0, sizeof(other.m_descriptor_set));
          return *this;
        }

        using properties_t = std::vector<fan::vulkan::write_descriptor_set_t>;
        void open(fan::vulkan::context_t& context, const properties_t& properties) {
          m_properties = properties;
          std::vector<VkDescriptorSetLayoutBinding> uboLayoutBinding(properties.size());
          for (std::uint16_t i = 0; i < properties.size(); ++i) {
            uboLayoutBinding[i].binding = properties[i].binding;
            uboLayoutBinding[i].descriptorCount = m_properties[i].descriptor_count;
            if (uboLayoutBinding[i].descriptorCount == 0) {
              uboLayoutBinding[i].descriptorCount = m_properties[i].use_image ?
                max_textures : 1;
            }
            uboLayoutBinding[i].descriptorType = properties[i].type;
            uboLayoutBinding[i].stageFlags = properties[i].flags;
          }

          std::vector<VkDescriptorBindingFlags> binding_flags(uboLayoutBinding.size());
          bool has_update_after_bind = false;
          for (std::uint32_t i = 0; i < (std::uint32_t)uboLayoutBinding.size(); ++i) {
            bool is_bindless =
              uboLayoutBinding[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
              uboLayoutBinding[i].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
              uboLayoutBinding[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
              uboLayoutBinding[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
              uboLayoutBinding[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

            binding_flags[i] = is_bindless ?
              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
              VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
              VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0;
            has_update_after_bind |= is_bindless;
          }

          VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{};
          binding_flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
          binding_flags_info.bindingCount = (std::uint32_t)binding_flags.size();
          binding_flags_info.pBindingFlags = binding_flags.data();

          VkDescriptorSetLayoutCreateInfo layoutInfo{};
          layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
          layoutInfo.pNext = has_update_after_bind ? &binding_flags_info : nullptr;
          layoutInfo.flags = has_update_after_bind ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0;
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
          if (m_layout == VK_NULL_HANDLE) { return; }
          vkDestroyDescriptorSetLayout(context.device, m_layout, 0);
          m_layout = VK_NULL_HANDLE;
        }

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
        VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
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
          pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
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

      struct shader_t {
        int projection_view[2]{ -1, -1 };
        fan::vulkan::context_t::uniform_block_t<fan::vulkan::view_projection_t, fan::vulkan::max_camera>* projection_view_block;
        VkPipelineShaderStageCreateInfo shader_stages[3]{};
        std::uint32_t compile_generation = 0;
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


      bool shader_compile(fan::graphics::shader_nr_t nr);

      struct image_format {
        static constexpr auto b8g8r8a8_unorm = VK_FORMAT_B8G8R8A8_UNORM;
        static constexpr auto r8b8g8a8_unorm = VK_FORMAT_R8G8B8A8_UNORM;
        static constexpr auto r8_unorm = VK_FORMAT_R8_UNORM;
        static constexpr auto r8_uint = VK_FORMAT_R8_UINT;
        static constexpr auto r8g8_unorm = VK_FORMAT_R8G8_UNORM;
        static constexpr auto r8g8b8_unorm = VK_FORMAT_R8G8B8_UNORM;
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
        static constexpr VkFormat format = image_format::r8b8g8a8_unorm;
        static constexpr VkFilter min_filter = image_filter::nearest;
        static constexpr VkFilter mag_filter = image_filter::nearest;
      };
      struct image_load_properties_t {
        VkSamplerAddressMode visual_output = image_load_properties_defaults::visual_output;
        std::uint8_t internal_format = 0;
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
      struct buffer_t {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        void* mapped = nullptr;

        operator VkBuffer() const { return buffer; }
        explicit operator bool() const { return buffer != VK_NULL_HANDLE; }
      };

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
      struct buffer_barrier_t {
        buffer_t* buffer = nullptr;
        VkAccessFlags src_access = 0;
        VkAccessFlags dst_access = 0;
        VkDeviceSize offset = 0;
        VkDeviceSize size = VK_WHOLE_SIZE;
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
            return 1;
          }
          case fan::vulkan::context_t::image_format::r8g8_unorm:
          {
            return 2;
          }
          case fan::vulkan::context_t::image_format::r8g8b8_unorm:
          {
            return 3;
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
          {
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
          VkRenderPass render_pass = VK_NULL_HANDLE;
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
          pipelineInfo.renderPass = p.render_pass == VK_NULL_HANDLE ? context.render_pass : p.render_pass;
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

      void destroy_shape_resources();

      void cleanup_swap_chain_dependencies();

      void cleanup_swap_chain();

      void recreate_swap_chain_dependencies();

      void update_swapchain_dependencies();
      void recreate_swap_chain(fan::window_t* window, VkResult err);


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
      void create_allocator();
      void destroy_allocator();
      void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VmaAllocation& allocation, VmaAllocationInfo* allocation_info = nullptr);
      void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, buffer_t& buffer, VmaAllocationInfo* allocation_info = nullptr);
      void destroy_buffer(VkBuffer& buffer, VmaAllocation& allocation);
      void destroy_buffer(buffer_t& buffer);
      VkResult map_buffer(buffer_t& buffer, void** data);
      void unmap_buffer(buffer_t& buffer);
      void invalidate_buffer(buffer_t& buffer, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

      VkCommandBuffer begin_single_time_commands();

      void end_single_time_commands(VkCommandBuffer command_buffer);
      void copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size);
      void copy_buffer_cmd(VkCommandBuffer cmd, VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize src_offset, VkDeviceSize dst_offset, VkDeviceSize size);
      void fill_buffer_cmd(VkCommandBuffer cmd, buffer_t& buffer, VkDeviceSize offset, VkDeviceSize size, std::uint32_t data);
      void buffer_barrier_cmd(VkCommandBuffer cmd, buffer_t& buffer, VkAccessFlags src_access, VkAccessFlags dst_access, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
      void buffer_barriers_cmd(VkCommandBuffer cmd, const std::vector<buffer_barrier_t>& barriers, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage);

      template <typename T>
      void upload_buffer(const std::vector<T>& data, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation) {
        if (data.empty()) return;
        VkDeviceSize size = sizeof(T) * data.size();
        buffer_t staging;
        create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
        void* mapped = nullptr;
        fan::vulkan::validate(map_buffer(staging, &mapped));
        std::memcpy(mapped, data.data(), size);
        unmap_buffer(staging);
        create_buffer(size, usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, allocation);
        copy_buffer(staging, buffer, size);
        destroy_buffer(staging);
      }
      template <typename T>
      void upload_buffer(const std::vector<T>& data, VkBufferUsageFlags usage, buffer_t& buffer) {
        if (data.empty()) return;
        upload_buffer(data, usage, buffer.buffer, buffer.allocation);
        buffer.size = sizeof(T) * data.size();
      }

      void insert_image_barrier(
        VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout,
        VkAccessFlags src_access, VkAccessFlags dst_access,
        VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage
      ) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask = src_access;
        barrier.dstAccessMask = dst_access;
        vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
      }


      void create_command_buffers();
      void bind_draw(
        const fan::vulkan::context_t::pipeline_t& pipeline,
        std::uint32_t descriptor_count,
        VkDescriptorSet* descriptor_sets);
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

      bool                     SwapChainRebuild = false;
      #if defined(FAN_GUI)
      ImGui_ImplVulkanH_Window MainWindowData;
      std::uint32_t                 MinImageCount = 2;
      void ImGuiSetupVulkanWindow();

      static void ImGuiFrameRender(void* ctx, VkResult next_image_khr_err, fan::color clear_color);
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
      VmaAllocator allocator = VK_NULL_HANDLE;

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
      VkFormat main_color_format = VK_FORMAT_R8G8B8A8_UNORM;
      fan::vec2 swap_chain_size;
      std::vector<VkFramebuffer> swap_chain_framebuffers;
      VkPresentModeKHR present_mode;
      VkSurfaceFormatKHR surface_format;

      VkRenderPass render_pass;

      VkCommandPool command_pool;
      VkCommandBuffer single_time_cmd = VK_NULL_HANDLE;
      VkFence single_time_fence = VK_NULL_HANDLE;
      std::uint32_t queue_family = -1;
      std::uint32_t min_image_count = 0;
      std::uint32_t image_count = 0;
      std::vector<VkCommandBuffer> command_buffers;

      std::vector<std::function<void()>> pre_begin_cmd_cb;
      std::vector<std::function<void(VkCommandBuffer)>> begin_cmd_cb;

      std::vector<VkSemaphore> image_available_semaphores;
      std::uint32_t acquire_semaphore_index = 0;
      VkSemaphore current_acquire_semaphore = VK_NULL_HANDLE;
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

void fan::vulkan::image_create(const fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VmaAllocation& allocation);
namespace fan::graphics::format_converter {
  fan::vulkan::context_t::image_load_properties_t image_global_to_vulkan(const fan::graphics::image_load_properties_t& p);
}

export namespace fan::graphics {
  fan::graphics::context_functions_t get_vk_context_functions();
  fan::vulkan::context_t& get_vk_context();
}
