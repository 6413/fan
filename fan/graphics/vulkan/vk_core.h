#pragma once

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

#include <fan/types/matrix.h>

#include <optional>
#include <vector>
#include <set>


const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
#if defined(loco_window)
  std::optional<uint32_t> presentFamily;
#endif

  bool is_complete();
};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

struct UniformBufferObject {
  alignas(16) fan::mat4 model;
  alignas(16) fan::mat4 view;
  alignas(16) fan::mat4 proj;
};

namespace fan {
  namespace vulkan {
    struct context_t;
    void create_image(fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
  }
}

namespace fan {
  namespace vulkan {

    void validate(VkResult result);

    static constexpr uint16_t max_camera = 16;
    static constexpr uint16_t max_textures = 16;

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
      VkDescriptorImageInfo image_infos[max_textures];
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

      void transition_image_layout(auto& context, VkImageLayout newLayout);

      VkFormat format;

      VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;

      VkImage image;
      VkImageView image_view;
      VkDeviceMemory memory;
    };
  }
}

namespace fan {
  namespace vulkan {

    namespace core {
      struct memory_write_queue_t;
    }

    struct context_t {
      struct descriptor_pool_t {

#define loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_image_sampler
        void open(fan::vulkan::context_t& context);
        void close(fan::vulkan::context_t& context);

        VkDescriptorPool m_descriptor_pool;
      }descriptor_pool;

      struct shader_t {
        int projection_view[2]{ -1, -1 };
        //fan::vulkan::core::uniform_block_t<fan::vulkan::context_t::viewprojection_t, fan::vulkan::max_camera> projection_view_block;
        VkPipelineShaderStageCreateInfo shader_stages[2];
        // can be risky without constructor copy
        std::string svertex, sfragment;

        std::unordered_map<std::string, std::string> uniform_type_table;
      };

      // NOTE opengl, should probably be combined into global since they are same
      #include <fan/graphics/opengl/shader_list_builder_settings.h>
      #include <BLL/BLL.h>
      shader_list_t shader_list;

      using shader_nr_t = shader_list_NodeReference_t;

      struct pipeline_t {

        struct properties_t {
          uint32_t descriptor_layout_count;
          VkDescriptorSetLayout* descriptor_layout;
          fan::vulkan::context_t::shader_nr_t shader;
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

        VkPipelineLayout m_layout;
        VkPipeline m_pipeline;
      };

      template <uint32_t count>
      struct descriptor_t {

        void open(fan::vulkan::context_t& context, std::array<fan::vulkan::write_descriptor_set_t, count> properties);
        void close(fan::vulkan::context_t& context);

        // for buffer update, need to manually call .m_properties.common
        void update(
          fan::vulkan::context_t& context,
          uint32_t n = count,
          uint32_t begin = 0,
          uint32_t texture_n = max_textures,
          uint32_t texture_begin = 0
        );

        std::array<fan::vulkan::write_descriptor_set_t, count> m_properties;
        VkDescriptorSetLayout m_layout;
        VkDescriptorSet m_descriptor_set[fan::vulkan::max_frames_in_flight];
      };

      shader_nr_t shader_create();
      shader_t& shader_get(shader_nr_t nr);
      void shader_erase(shader_nr_t nr);

      void shader_use(shader_nr_t nr);

      void shader_set_vertex(shader_nr_t nr, const fan::string& vertex_code);
      void shader_set_fragment(shader_nr_t nr, const fan::string& fragment_code);
      bool shader_compile(shader_nr_t nr);

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

      void create_loco_framebuffer();

      void create_wboit_views();
      void create_depth_resources();

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

      void begin_render(const fan::color& clear_color);

       //----------------------------------------------imgui stuff----------------------------------------------

      ImGui_ImplVulkanH_Window MainWindowData;
      uint32_t                 MinImageCount = 2;
      bool                     SwapChainRebuild = false;
      
      void ImGuiSetupVulkanWindow();

      void ImGuiFrameRender(VkResult next_image_khr_err, fan::color clear_color);
      //----------------------------------------------imgui stuff----------------------------------------------

      VkResult end_render();

      void begin_compute_shader();

      void end_compute_shader();

      VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats);

      VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes);

      VkExtent2D choose_swap_extent(const fan::vec2ui& framebuffer_size, const VkSurfaceCapabilitiesKHR& capabilities);

      SwapChainSupportDetails query_swap_chain_support(VkPhysicalDevice device);

      bool is_device_suitable(VkPhysicalDevice device);

      bool check_device_extension_support(VkPhysicalDevice device);

      QueueFamilyIndices find_queue_families(VkPhysicalDevice device);

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

      VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
      VkDevice device;

      VkQueue graphics_queue;
#if defined(loco_window)
      VkQueue present_queue;
#endif

      VkSwapchainKHR swap_chain;
      std::vector<VkImage> swap_chain_images;
      VkFormat swap_chain_image_format;
      fan::vec2 swap_chain_size;
      std::vector<VkImageView> swap_chain_image_views;
      std::vector<VkFramebuffer> swap_chain_framebuffers;
      VkPresentModeKHR present_mode;
      VkSurfaceFormatKHR surface_format;

      VkRenderPass render_pass;

      VkCommandPool command_pool;
      uint32_t queue_family = -1;
      uint32_t min_image_count = 0;
      uint32_t image_count = 0;

      vai_t vai_depth;
      vai_t vai_bitmap[2];

      std::vector<VkCommandBuffer> command_buffers;

      std::vector<VkSemaphore> image_available_semaphores;
      std::vector<VkSemaphore> render_finished_semaphores;
      std::vector<VkFence> in_flight_fences;
      uint32_t current_frame = 0;

      bool vsync = true;
      uint32_t image_index;

      fan::vulkan::context_t::pipeline_t render_fullscreen_pl;

      bool command_buffer_in_use = false;

      bool supports_validation_layers = true;
    };
  }
}

#include "memory.h"
#include "uniform_block.h"
//#include "ssbo.h"

namespace fan {
  namespace vulkan {

    template <uint32_t count>
    inline void fan::vulkan::context_t::descriptor_t<count>::open(fan::vulkan::context_t& context, std::array<fan::vulkan::write_descriptor_set_t, count> properties) {
      m_properties = properties;

      VkDescriptorSetLayoutBinding uboLayoutBinding[count]{};
      for (uint16_t i = 0; i < count; ++i) {
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
      layoutInfo.pBindings = uboLayoutBinding;

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

    template <uint32_t count>
    inline void fan::vulkan::context_t::descriptor_t<count>::close(fan::vulkan::context_t& context) {
      vkDestroyDescriptorSetLayout(context.device, m_layout, 0);
    }

    template <uint32_t count>
    inline void fan::vulkan::context_t::descriptor_t<count>::update(
      fan::vulkan::context_t& context,
      uint32_t n,
      uint32_t begin,
      uint32_t texture_n,
      uint32_t texture_begin
    ) {
      VkDescriptorBufferInfo bufferInfo[count * max_frames_in_flight]{};

      for (size_t frame = 0; frame < max_frames_in_flight; frame++) {

        std::array<VkWriteDescriptorSet, count> descriptorWrites{};

        for (uint32_t j = begin; j < begin + n; ++j) {

          if (m_properties[j].buffer) {
            bufferInfo[frame * count + j].buffer = m_properties[j].buffer;
            bufferInfo[frame * count + j].offset = 0;
            bufferInfo[frame * count + j].range = m_properties[j].range;
          }

          descriptorWrites[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          descriptorWrites[j].dstSet = m_descriptor_set[frame];
          descriptorWrites[j].descriptorType = m_properties[j].type;
          descriptorWrites[j].descriptorCount = 1;
          descriptorWrites[j].pBufferInfo = &bufferInfo[frame * count + j];
          descriptorWrites[j].dstBinding = m_properties[j].dst_binding;

          // FIX
          if (m_properties[j].use_image) {
            descriptorWrites[j].pImageInfo = &m_properties[j].image_infos[texture_begin];
            descriptorWrites[j].descriptorCount = texture_n;
          }
        }
        vkUpdateDescriptorSets(context.device, n, descriptorWrites.data() + begin, 0, nullptr);
      }
    }
  }
}