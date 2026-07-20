module;

#if defined(FAN_2D)

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
import fan.window;

#if defined(fan_platform_windows)
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <Windows.h>
  #define GLFW_EXPOSE_NATIVE_WIN32
  #define GLFW_EXPOSE_NATIVE_WGL
  #define GLFW_NATIVE_INCLUDE_NONE
#endif
#ifndef GLFW_INCLUDE_NONE
  #define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <fan/utility.h>

#endif

export module fan.graphics.vulkan.core;

#if defined(FAN_2D)

import std;

export import :types;
export import :vai;
export import :image;
export import :compute;
export import :pipeline;
export import :camera_subsystem;
export import :shader_subsystem;

import fan.types;
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

#define ENABLE_RAYTRACING_DEPENDENCIES

export namespace fan::vulkan {
  void* VKAPI_PTR vk_allocation_cb(void* pUserData, size_t size, size_t alignment, VkSystemAllocationScope allocationScope) {
    return fan::memory_profile_malloc_cb(size);
  }
  void* VKAPI_PTR vk_reallocation_cb(void* pUserData, void* pOriginal, size_t size, size_t alignment, VkSystemAllocationScope allocationScope) {
    return fan::memory_profile_realloc_cb(pOriginal, size);
  }
  void VKAPI_PTR vk_free_cb(void* pUserData, void* pMemory) {
    fan::memory_profile_free_cb(pMemory);
  }

  VkAllocationCallbacks g_allocation_callbacks = {
    .pUserData = nullptr,
    .pfnAllocation = vk_allocation_cb,
    .pfnReallocation = vk_reallocation_cb,
    .pfnFree = vk_free_cb,
    .pfnInternalAllocation = nullptr,
    .pfnInternalFree = nullptr
  };
}

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
  VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
});

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);

export {
extern PFN_vkCreateShadersEXT fan_vkCreateShadersEXT;
extern PFN_vkDestroyShaderEXT fan_vkDestroyShaderEXT;
extern PFN_vkCmdBindShadersEXT fan_vkCmdBindShadersEXT;
extern PFN_vkCmdSetVertexInputEXT fan_vkCmdSetVertexInputEXT;
extern PFN_vkCmdSetColorBlendEnableEXT fan_vkCmdSetColorBlendEnableEXT;
extern PFN_vkCmdSetColorBlendEquationEXT fan_vkCmdSetColorBlendEquationEXT;
extern PFN_vkCmdSetColorWriteMaskEXT fan_vkCmdSetColorWriteMaskEXT;
extern PFN_vkCmdSetRasterizerDiscardEnable fan_vkCmdSetRasterizerDiscardEnable;
extern PFN_vkCmdSetPolygonModeEXT fan_vkCmdSetPolygonModeEXT;
extern PFN_vkCmdSetDepthTestEnable fan_vkCmdSetDepthTestEnable;
extern PFN_vkCmdSetDepthWriteEnable fan_vkCmdSetDepthWriteEnable;
extern PFN_vkCmdSetDepthCompareOp fan_vkCmdSetDepthCompareOp;
extern PFN_vkCmdSetDepthBoundsTestEnable fan_vkCmdSetDepthBoundsTestEnable;
extern PFN_vkCmdSetCullMode fan_vkCmdSetCullMode;
extern PFN_vkCmdSetFrontFace fan_vkCmdSetFrontFace;
extern PFN_vkCmdSetDepthBiasEnable fan_vkCmdSetDepthBiasEnable;
extern PFN_vkCmdSetStencilTestEnable fan_vkCmdSetStencilTestEnable;
extern PFN_vkCmdSetStencilOp fan_vkCmdSetStencilOp;
extern PFN_vkCmdSetPrimitiveTopology fan_vkCmdSetPrimitiveTopology;
extern PFN_vkCmdSetPrimitiveRestartEnable fan_vkCmdSetPrimitiveRestartEnable;
extern PFN_vkCmdSetAlphaToCoverageEnableEXT fan_vkCmdSetAlphaToCoverageEnableEXT;
extern PFN_vkCmdSetRasterizationSamplesEXT fan_vkCmdSetRasterizationSamplesEXT;
extern PFN_vkCmdSetSampleMaskEXT fan_vkCmdSetSampleMaskEXT;
}

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

}

export namespace fan {
  namespace vulkan {

    struct context_t {
      frame_deletion_queue_t frame_deletion_queues[max_frames_in_flight];
      frame_deletion_queue_t main_deletion_queue;
      frame_deletion_queue_t pending_deletion_queue;
      staging_ring_buffer_t staging_ring_buffer;

      frame_deletion_queue_t& get_current_deletion_queue() {
        return pending_deletion_queue;
      }
      frame_deletion_queue_t& get_current_deletion_queue(std::uint32_t frame) {
        return frame_deletion_queues[frame];
      }

      void flush_deletion_queues();
      void retire_frame_deletions(std::uint32_t frame) {
        get_current_deletion_queue(frame).flush();
        get_current_deletion_queue(frame).merge(pending_deletion_queue);
      }

      using push_constants_t = fan::vulkan::push_constants_t;
      using descriptor_t = fan::vulkan::descriptor_t;
      using descriptor_pool_t = fan::vulkan::descriptor_pool_t;
      descriptor_pool_t descriptor_pool;
      VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
      VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
      double timestamp_period = 1.0;
      std::uint64_t gpu_timestamps[2] = {0, 0};
      #include "memory.h"
      #include "ssbo.h"

      using image_format = fan::vulkan::image_format;
      using image_sampler_address_mode = fan::vulkan::image_sampler_address_mode;
      using image_filter = fan::vulkan::image_filter;
      using image_load_properties_defaults = fan::vulkan::image_load_properties_defaults;
      using image_load_properties_t = fan::vulkan::image_load_properties_t;
      using primitive_topology_t = fan::vulkan::primitive_topology_t;
      void transition_image_layout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
      void transition_image_layout_cmd(VkCommandBuffer cmd, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
      void copy_buffer_to_image(
        VkBuffer buffer,
        VkImage image,
        VkFormat format,
        const fan::vec2ui& size,
        VkDeviceSize buffer_offset = 0
      );
      void copy_buffer_to_image_cmd(
        VkCommandBuffer cmd,
        VkBuffer buffer,
        VkImage image,
        VkFormat format,
        const fan::vec2ui& size,
        VkDeviceSize buffer_offset = 0
      );
      void create_texture_sampler(VkSampler& sampler, const image_load_properties_t& lp);

      using image_t = fan::vulkan::image_t;
      using buffer_t = fan::vulkan::buffer_t;
      using compute_pipeline_t = fan::vulkan::compute_pipeline_t;
      using compute_slot_ring_t = fan::vulkan::compute_slot_ring_t;
      using buffer_barrier_t = fan::vulkan::buffer_barrier_t;

      VkFormat get_format_from_channels(int channels);

      struct pending_image_upload_t {
        fan::graphics::decoded_image_payload_t payload;
        std::function<void(const fan::graphics::decoded_image_payload_t&)> callback;
      };
      std::vector<pending_image_upload_t> pending_image_uploads;
      std::mutex async_image_mutex;

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
      fan::graphics::image_nr_t request_image_load_async(
        fan::str_view_t path,
        const fan::vulkan::context_t::image_load_properties_t& p,
        std::function<void(const fan::graphics::decoded_image_payload_t&)> on_gpu_uploaded
      );
      void process_async_image_uploads();

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

      using pipeline_t = fan::vulkan::pipeline_t;

      struct image_cache_entry_t {
        fan::graphics::image_nr_t nr;
        std::uint32_t ref_count;
      };
      std::unordered_map<std::string, image_cache_entry_t> image_cache;

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

      VkCommandBuffer begin_async_transfer_commands();
      void end_async_transfer_commands(VkCommandBuffer command_buffer);
      void copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size);
      void copy_buffer_cmd(VkCommandBuffer cmd, VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize src_offset, VkDeviceSize dst_offset, VkDeviceSize size);
      void fill_buffer_cmd(VkCommandBuffer cmd, buffer_t& buffer, VkDeviceSize offset, VkDeviceSize size, std::uint32_t data);
      void upload_to_buffer(VkBuffer dest_buffer, const void* data, std::size_t size, VkDeviceSize dst_offset = 0);
      void buffer_barrier_cmd(VkCommandBuffer cmd, buffer_t& buffer, VkAccessFlags src_access, VkAccessFlags dst_access, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
      void buffer_barriers_cmd(VkCommandBuffer cmd, const std::vector<buffer_barrier_t>& barriers, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage);

      void upload_buffer(const void* data, std::size_t size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation);
      void upload_buffer(const void* data, std::size_t size, VkBufferUsageFlags usage, buffer_t& buffer);

      void insert_image_barrier(
        VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout,
        VkAccessFlags src_access, VkAccessFlags dst_access,
        VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage
      );


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
      VkResult end_render(fan::window_t* window);
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
      VkFormat main_color_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
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
      std::uint32_t begin_count = 0;


      fan::window_t::resize_handle_t window_resize_handle;
      bool enable_clear = true;
      bool shapes_top = false;

      bool vsync = true;
      std::uint32_t image_index;

      fan::vulkan::context_t::pipeline_t render_fullscreen_pl;

      bool command_buffer_in_use = false;
#if FAN_DEBUG >= fan_debug_high
      bool supports_validation_layers = true;
#else
      bool supports_validation_layers = false;
#endif

      bool window_shown = false;
      fan::vulkan::context_t::memory_write_queue_t memory_queue;
      fan::vulkan::shader_subsystem_t shaders;
      fan::vulkan::camera_subsystem_t cameras;
    };
  }
}

export namespace fan {
  namespace vulkan {
    namespace core {
      std::uint32_t get_draw_mode(std::uint8_t draw_mode);
    }
  }
}

void fan::vulkan::image_create(const fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VmaAllocation& allocation);
namespace fan::graphics::format_converter {
  fan::vulkan::context_t::image_load_properties_t image_global_to_vulkan(const fan::graphics::image_load_properties_t& p);
  fan::graphics::image_load_properties_t image_vulkan_to_global(const fan::vulkan::context_t::image_load_properties_t& p);
}

export namespace fan::graphics {
  fan::graphics::context_functions_t get_vk_context_functions();
  fan::vulkan::context_t& get_vk_context();
}


#endif