#pragma once

#include <fan/graphics/common_context.h>

#if defined(fan_platform_windows)
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
#define VK_USE_PLATFORM_XLIB_KHR
#endif

#if defined(loco_imgui)
  #include <fan/imgui/imgui_impl_vulkan.h>
#endif

#define loco_window

//for window surface
#if defined(loco_window)
  #include <fan/window/window.h>
#endif

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>

#include <fan/types/matrix.h>

#include <optional>
#include <vector>
#include <set>

#include <fan/graphics/camera.h>
#include <fan/graphics/image_load.h>


const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);

struct queue_family_indices_t {
  std::optional<uint32_t> graphics_family;
#if defined(loco_window)
  std::optional<uint32_t> present_family;
#endif
  bool is_complete();
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

namespace fan {
  namespace vulkan {
    struct context_t;
    void image_create(fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
  }
}

namespace fan::graphics::format_converter {
  VkSamplerAddressMode global_to_vulkan_address_mode(uintptr_t mode);
  VkFilter global_to_vulkan_filter(uintptr_t filter);
}

namespace fan {
  namespace vulkan {

    void validate(VkResult result);

    static constexpr uint16_t max_camera = 16;
    static constexpr uint16_t max_textures = 0xffff;

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

    static constexpr uint32_t max_frames_in_flight = 1;

    static uint32_t makeAccessMaskPipelineStageFlags(uint32_t accessMask);

    // view and image
    struct vai_t {
      struct properties_t {
        fan::vec2 swap_chain_size;
        VkFormat format;
        VkImageUsageFlags usage_flags;
        VkImageAspectFlags aspect_flags;
      };
      void open(auto& context, const properties_t& p);
      void close(auto& context);

      void transition_image_layout(auto& context, VkImageLayout newLayout, VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT);

      VkFormat format;

      VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;

      VkImage image;
      VkImageView image_view;
      VkDeviceMemory memory;
    };

    VkPipelineColorBlendAttachmentState get_default_color_blend();
  }
}

namespace fan {
  namespace vulkan {

    struct context_t {

      struct push_constants_t {
        uint32_t texture_id;
        uint32_t camera_id;
      };

      struct descriptor_t {

        using properties_t = std::vector<fan::vulkan::write_descriptor_set_t>;

        void open(fan::vulkan::context_t& context, const properties_t& properties);
        void close(fan::vulkan::context_t& context);

        // for buffer update, need to manually call .m_properties.common
        void update(
          fan::vulkan::context_t& context,
          uint32_t n,
          uint32_t begin = 0,
          uint32_t texture_n = max_textures,
          uint32_t texture_begin = 0
        );

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
        void open(fan::vulkan::context_t& context);
        void close(fan::vulkan::context_t& context);

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
        fan::vulkan::context_t::uniform_block_t<fan::vulkan::context_t::view_projection_t, fan::vulkan::max_camera> projection_view_block;
        VkPipelineShaderStageCreateInfo shader_stages[2]{};
      };

      static std::vector<uint32_t> compile_file(const fan::string& source_name,
        shaderc_shader_kind kind,
        const fan::string& source);

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

      void transition_image_layout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
      void copy_buffer_to_image(VkBuffer buffer, VkImage image, VkFormat format, const fan::vec2ui& size, const fan::vec2ui& stride = 1);
      void create_texture_sampler(VkSampler& sampler, const image_load_properties_t& lp);

      struct image_t {
        #include <fan/graphics/image_common.h>
        VkImage image_index = 0;
        VkImageView image_view;
        VkDeviceMemory image_memory;
        VkSampler sampler;
        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;
        void* data;
      };

      std::vector<VkDescriptorImageInfo> image_pool; // for draw
      //-----------------------------image-----------------------------

      //-----------------------------viewport-----------------------------

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

        void open(fan::vulkan::context_t& context, const properties_t& p);
        void close(fan::vulkan::context_t& context);

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

      void destroy_vulkan_soft();

    public:
      void imgui_close();

      void close();

      void cleanup_swap_chain_dependencies();

      void cleanup_swap_chain();

      void recreate_swap_chain_dependencies();

      // if swapchain changes, reque
      void update_swapchain_dependencies();

      void recreate_swap_chain(fan::window_t* window, VkResult err);

      void recreate_swap_chain(const fan::vec2i& window_size);

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

      VkFormat find_supported_foramt(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

      VkFormat find_depth_format();

      bool has_stencil_component(VkFormat format);

      void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory);

      VkCommandBuffer begin_single_time_commands();

      void end_single_time_commands(VkCommandBuffer command_buffer);

      void copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size);

      uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);

      void create_command_buffers();

      void bind_draw(
        const fan::vulkan::context_t::pipeline_t& pipeline,
        uint32_t descriptor_count,
        VkDescriptorSet* descriptor_sets);

      // assumes things are already bound
      void bindless_draw(
        uint32_t vertex_count,
        uint32_t instance_count,
        uint32_t first_instance);

      void draw(
        uint32_t vertex_count,
        uint32_t instance_count,
        uint32_t first_instance,
        const fan::vulkan::context_t::pipeline_t& pipeline,
        uint32_t descriptor_count,
        VkDescriptorSet* descriptor_sets
      );

      void create_sync_objects();

       //----------------------------------------------imgui stuff----------------------------------------------
      bool                     SwapChainRebuild = false;
      #if defined(loco_imgui)
      ImGui_ImplVulkanH_Window MainWindowData;
      uint32_t                 MinImageCount = 2;
      
      void ImGuiSetupVulkanWindow();

      void ImGuiFrameRender(VkResult next_image_khr_err, fan::color clear_color);
      //----------------------------------------------imgui stuff----------------------------------------------
#endif
      VkResult end_render();

      void begin_compute_shader();

      void end_compute_shader();

      VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats);

      VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes);

      VkExtent2D choose_swap_extent(const fan::vec2ui& framebuffer_size, const VkSurfaceCapabilitiesKHR& capabilities);

      swap_chain_support_details_t query_swap_chain_support(VkPhysicalDevice device);

      bool is_device_suitable(VkPhysicalDevice device);

      bool check_device_extension_support(VkPhysicalDevice device);

      queue_family_indices_t find_queue_families(VkPhysicalDevice device);

      std::vector<fan::string> get_required_extensions();

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

namespace fan {
  namespace vulkan {

    inline void fan::vulkan::context_t::descriptor_t::open(fan::vulkan::context_t& context, const std::vector<fan::vulkan::write_descriptor_set_t>& properties) {
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

    inline void fan::vulkan::context_t::descriptor_t::close(fan::vulkan::context_t& context) {
      vkDestroyDescriptorSetLayout(context.device, m_layout, 0);
    }

    inline void fan::vulkan::context_t::descriptor_t::update(
      fan::vulkan::context_t& context,
      uint32_t n,
      uint32_t begin,
      uint32_t texture_n,
      uint32_t texture_begin
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

    namespace core {
      uint32_t get_draw_mode(uint8_t draw_mode);
    }
  }
}