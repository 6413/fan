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

      void init(VkDevice device, VmaAllocator allocator, VkDeviceSize capacity = 32 * 1024 * 1024);

      void destroy() {
        if (buffer) {
          vmaDestroyBuffer(allocator, buffer, allocation);
          buffer = VK_NULL_HANDLE;
        }
      }

      allocation_t allocate(VkDeviceSize size, VkDeviceSize alignment = 16);

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
    std::uint32_t makeAccessMaskPipelineStageFlags(std::uint32_t accessMask);

    VkPipelineColorBlendAttachmentState get_default_color_blend();
  }
}
