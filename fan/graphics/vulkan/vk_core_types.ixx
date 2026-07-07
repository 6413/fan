module;

#if defined(fan_platform_windows)
  #define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
  #define VK_USE_PLATFORM_XLIB_KHR
#endif
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <fan/utility.h>

export module fan.graphics.vulkan.core:types;
import std;

import fan.types.vector;
import fan.types.matrix;
import fan.types.fstring;
import fan.types.color;
import fan.types.compile_time_string;

import fan.window;

import fan.memory;

import fan.utility;
import fan.print;
import fan.print.error;
import fan.graphics.image_load;
import fan.graphics.common_context;

export struct queue_family_indices_t {
  std::optional<std::uint32_t> graphics_family;
  std::optional<std::uint32_t> present_family;
  bool is_complete() {
    return graphics_family.has_value()
      && present_family.has_value()
      ;
  }
};

export struct swap_chain_support_details_t {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

export struct uniform_buffer_object_t {
  alignas(16) fan::mat4 model;
  alignas(16) fan::mat4 view;
  alignas(16) fan::mat4 proj;
};

export namespace fan {
  namespace vulkan {
    struct context_t;

    struct view_projection_t {
      fan::mat4 projection;
      fan::mat4 view;
    };

    void validate(VkResult result);

    inline constexpr std::uint16_t max_camera = 16;
    inline constexpr std::uint16_t max_textures = 1024;
    struct write_descriptor_set_t {
      std::uint32_t binding;
      std::uint32_t dst_binding = 0;
      VkDescriptorType type;
      VkShaderStageFlags flags;
      VkBuffer buffer = nullptr;
      std::uint64_t range;
      bool use_image = false;
      std::uint32_t descriptor_count = 0;
      std::vector<VkDescriptorImageInfo> image_infos{max_textures};
    };

    inline constexpr std::uint32_t max_frames_in_flight = 2;

    class frame_deletion_queue_t {
    public:
      template<typename F>
      void push_function(F&& deletor) {
        deletors.push_back(std::forward<F>(deletor));
      }

      void push_buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation) {
        push_function([=]() {
          vmaDestroyBuffer(allocator, buffer, allocation);
        });
      }

      void push_image(VmaAllocator allocator, VkImage image, VmaAllocation allocation) {
        push_function([=]() {
          vmaDestroyImage(allocator, image, allocation);
        });
      }

      void push_image_view(VkDevice device, VkImageView view) {
        push_function([=]() {
          vkDestroyImageView(device, view, nullptr);
        });
      }

      void push_pipeline(VkDevice device, VkPipeline pipeline) {
        push_function([=]() {
          vkDestroyPipeline(device, pipeline, nullptr);
        });
      }

      void flush() {
        for (auto it = deletors.rbegin(); it != deletors.rend(); ++it) {
          (*it)();
        }
        deletors.clear();
      }

      void merge(frame_deletion_queue_t& other) {
        deletors.insert(deletors.end(), std::make_move_iterator(other.deletors.begin()), std::make_move_iterator(other.deletors.end()));
        other.deletors.clear();
      }

    private:
      std::vector<std::function<void()>> deletors;
    };

    class staging_ring_buffer_t {
    public:
      struct allocation_t {
        VkBuffer buffer;
        VkDeviceSize offset;
        void* mapped_ptr;
        bool is_spilled;
        VmaAllocation fallback_allocation;
      };

      void init(VkDevice device, VmaAllocator allocator, VkDeviceSize capacity = 32 * 1024 * 1024) {
        this->allocator = allocator;
        total_capacity = capacity;
        head = 0;
        tail = 0;

        VkBufferCreateInfo buffer_info{
          .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
          .size = total_capacity,
          .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        };

        VmaAllocationCreateInfo alloc_info{
          .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | 
                   VMA_ALLOCATION_CREATE_MAPPED_BIT,
          .usage = VMA_MEMORY_USAGE_AUTO
        };

        vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &buffer, &allocation, &alloc_info_out);
      }

      void destroy() {
        if (buffer) {
          vmaDestroyBuffer(allocator, buffer, allocation);
          buffer = VK_NULL_HANDLE;
        }
      }

      allocation_t allocate(VkDeviceSize size, VkDeviceSize alignment = 16) {
        VkDeviceSize aligned_size = (size + alignment - 1) & ~(alignment - 1);
        VkDeviceSize aligned_head = (head + alignment - 1) & ~(alignment - 1);

        if (aligned_size > total_capacity) {
          return allocate_fallback(size);
        }

        if (aligned_head + aligned_size <= total_capacity) {
          if (head < tail && (aligned_head + aligned_size) >= tail) {
            return allocate_fallback(size);
          }
          
          allocation_t result{
            .buffer = buffer,
            .offset = aligned_head,
            .mapped_ptr = static_cast<std::uint8_t*>(alloc_info_out.pMappedData) + aligned_head,
            .is_spilled = false,
            .fallback_allocation = VK_NULL_HANDLE
          };
          head = aligned_head + aligned_size;
          return result;
        }

        VkDeviceSize wrapped_head = 0;
        if (wrapped_head + aligned_size >= tail) {
          return allocate_fallback(size);
        }

        allocation_t result{
          .buffer = buffer,
          .offset = wrapped_head,
          .mapped_ptr = alloc_info_out.pMappedData,
          .is_spilled = false,
          .fallback_allocation = VK_NULL_HANDLE
        };
        head = wrapped_head + aligned_size;
        return result;
      }

      void advance_tail(VkDeviceSize completed_offset) {
        tail = std::max(tail, completed_offset);
      }

      VkDeviceSize get_head() const { return head; }

    private:
      allocation_t allocate_fallback(VkDeviceSize size) {
        VkBufferCreateInfo buffer_info{
          .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
          .size = size,
          .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        };

        VmaAllocationCreateInfo alloc_info{
          .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | 
                   VMA_ALLOCATION_CREATE_MAPPED_BIT,
          .usage = VMA_MEMORY_USAGE_AUTO
        };

        VkBuffer fallback_buf;
        VmaAllocation fallback_alloc;
        VmaAllocationInfo info;
        vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &fallback_buf, &fallback_alloc, &info);

        return allocation_t{
          .buffer = fallback_buf,
          .offset = 0,
          .mapped_ptr = info.pMappedData,
          .is_spilled = true,
          .fallback_allocation = fallback_alloc
        };
      }

      VmaAllocator allocator = VK_NULL_HANDLE;
      VkBuffer buffer = VK_NULL_HANDLE;
      VmaAllocation allocation = VK_NULL_HANDLE;
      VmaAllocationInfo alloc_info_out{};
      VkDeviceSize total_capacity = 0;
      VkDeviceSize head = 0;
      VkDeviceSize tail = 0;
    };
    inline std::uint32_t makeAccessMaskPipelineStageFlags(std::uint32_t accessMask) {
      static constexpr std::uint32_t accessPipes[] = {
        VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        VK_ACCESS_INDEX_READ_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        VK_ACCESS_UNIFORM_READ_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT
        | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT
        | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT
        | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
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

    inline VkPipelineColorBlendAttachmentState get_default_color_blend() {
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
