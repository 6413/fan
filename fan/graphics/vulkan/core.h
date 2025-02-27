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
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
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

    struct context_t {

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

      #include "memory.h"
      #include "uniform_block.h"
      #include "ssbo.h"

      struct descriptor_pool_t {

#define loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_image_sampler
        void open(fan::vulkan::context_t& context);
        void close(fan::vulkan::context_t& context);

        VkDescriptorPool m_descriptor_pool;
      }descriptor_pool;

      //-----------------------------camera-----------------------------

      struct camera_t : fan::camera {
        fan::mat4 m_projection = fan::mat4(1);
        fan::mat4 m_view = fan::mat4(1);
        f32_t zfar = 1000.f;
        f32_t znear = 0.1f;

        union {
          struct {
            f32_t left;
            f32_t right;
            f32_t up;
            f32_t down;
          };
          fan::vec4 v;
        }coordinates;
      };

    static constexpr f32_t znearfar = 0xffff;

    protected:

      #include <fan/graphics/opengl/camera_list_builder_settings.h>
      #include <BLL/BLL.h>
    public:
      using camera_nr_t = camera_list_NodeReference_t;

      camera_list_t camera_list;

      camera_nr_t camera_create();
      camera_t& camera_get(camera_nr_t nr);
      void camera_erase(camera_nr_t nr);

      camera_nr_t camera_open(const fan::vec2& x, const fan::vec2& y);

      fan::vec3 camera_get_position(camera_nr_t nr);
      void camera_set_position(camera_nr_t nr, const fan::vec3& cp);
      fan::vec2 camera_get_size(camera_nr_t nr);

      void camera_set_ortho(camera_nr_t nr, fan::vec2 x, fan::vec2 y);
      void camera_set_perspective(camera_nr_t nr, f32_t fov, const fan::vec2& window_size);

      void camera_rotate(camera_nr_t nr, const fan::vec2& offset);

      //-----------------------------camera-----------------------------

      //-----------------------------shader-----------------------------
      struct view_projection_t {
        fan::mat4 view;
        fan::mat4 projection;
      };

      struct shader_t {
        int projection_view[2]{ -1, -1 };
        fan::vulkan::context_t::uniform_block_t<fan::vulkan::context_t::view_projection_t, fan::vulkan::max_camera> projection_view_block;
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

      
      static std::vector<uint32_t> compile_file(const fan::string& source_name,
        shaderc_shader_kind kind,
        const fan::string& source);

      shader_nr_t shader_create();
      shader_t& shader_get(shader_nr_t nr);
      void shader_erase(shader_nr_t nr);

      void shader_use(shader_nr_t nr);

      void shader_set_vertex(shader_nr_t nr, const fan::string& vertex_code);
      void shader_set_fragment(shader_nr_t nr, const fan::string& fragment_code);
      bool shader_compile(shader_nr_t nr);

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
        static constexpr VkSamplerAddressMode visual_output = image_sampler_address_mode::clamp_to_border;
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
        //uintptr_t           internal_format = load_properties_defaults::internal_format;
        //uintptr_t           format = load_properties_defaults::format;
        //uintptr_t           type = load_properties_defaults::type;
        VkFormat format = image_load_properties_defaults::format;
        VkFilter           min_filter = image_load_properties_defaults::min_filter;
        VkFilter           mag_filter = image_load_properties_defaults::mag_filter;
        // unused opengl filler
        uint8_t internal_format = 0;
      };

      void transition_image_layout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
      void copy_buffer_to_image(VkBuffer buffer, VkImage image, VkFormat format, const fan::vec2ui& size, const fan::vec2ui& stride = 1);
      void create_texture_sampler(VkSampler& sampler, const image_load_properties_t& lp);

            // TODO
      struct image_list_texture_index_t {
        uint8_t sprite = -1;
        uint8_t letter = -1;
        uint8_t yuv420p = -1;
      };

      struct image_t {
        // --common--
        fan::vec2 size;
        // --common--
        image_list_texture_index_t texture_index;
        VkImage image_index;
        VkImageView image_view;
        VkDeviceMemory image_memory;
        VkSampler sampler;
        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;
        void* data;
      };

      #include <fan/graphics/opengl/image_list_builder_settings.h>
      #include <BLL/BLL.h>
      image_list_t image_list;

      using image_nr_t = image_list_NodeReference_t;
      
      image_nr_t image_create();
      uint64_t image_get_handle(image_nr_t nr);
      image_t& image_get(image_nr_t nr);

      void image_erase(image_nr_t nr);

      void image_bind(image_nr_t nr);
      void image_unbind(image_nr_t nr);

      void image_set_settings(const image_load_properties_t& p);

      image_nr_t image_load(const fan::image::image_info_t& image_info);
      image_nr_t image_load(const fan::image::image_info_t& image_info, const image_load_properties_t& p);
      image_nr_t image_load(const fan::string& path);
      image_nr_t image_load(const fan::string& path, const image_load_properties_t& p);
      image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_);
      image_nr_t image_load(fan::color* colors, const fan::vec2ui& size_, const image_load_properties_t& p);

      void image_unload(image_nr_t nr);

      image_nr_t create_missing_texture();
      image_nr_t create_transparent_texture();

      void image_reload_pixels(image_nr_t nr, const fan::image::image_info_t& image_info);
      void image_reload_pixels(image_nr_t nr, const fan::image::image_info_t& image_info, const image_load_properties_t& p);

      image_nr_t image_create(const fan::color& color);
      image_nr_t image_create(const fan::color& color, const fan::vulkan::context_t::image_load_properties_t& p);

      //-----------------------------image-----------------------------

      //-----------------------------viewport-----------------------------

      struct viewport_t {
        fan::vec2 viewport_position;
        fan::vec2 viewport_size;
      };

    protected:
      #include <fan/graphics/opengl/viewport_list_builder_settings.h>
      #include <BLL/BLL.h>
    public:

      using viewport_nr_t = viewport_list_NodeReference_t;

      viewport_list_t viewport_list;

      viewport_nr_t viewport_create();
      viewport_t& viewport_get(viewport_nr_t nr);
      void viewport_erase(viewport_nr_t nr);

      fan::vec2 viewport_get_position(viewport_nr_t nr);
      fan::vec2 viewport_get_size(viewport_nr_t nr);

      void viewport_set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
      void viewport_set(viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
      void viewport_zero(viewport_nr_t nr);

      bool viewport_inside(viewport_nr_t nr, const fan::vec2& position);
      bool viewport_inside_wir(viewport_nr_t nr, const fan::vec2& position);

      //-----------------------------viewport-----------------------------

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

      fan::vulkan::context_t::memory_write_queue_t memory_queue;
    };
  }
}
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