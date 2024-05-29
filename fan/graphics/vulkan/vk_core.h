#pragma once

#include <optional>

#if defined(fan_platform_windows)
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
#define VK_USE_PLATFORM_XLIB_KHR
#endif

#include <vulkan/vulkan.h>
#pragma comment(lib, "lib/vulkan/vulkan-1.lib")

#include <vector>
#include <set>

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  }
  else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
#if defined(loco_window)
  std::optional<uint32_t> presentFamily;
#endif

  bool isComplete() {
    return graphicsFamily.has_value()
#if defined(loco_window)
      && presentFamily.has_value()
#endif
      ;
  }
};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

struct UniformBufferObject {
  alignas(16) fan::mat4 model;
  alignas(16) fan::mat4 view;
  alignas(16) fan::mat4 proj;
};

namespace fan {
  namespace vulkan {

    struct context_t;

    static void createImage(fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
  }
}

namespace fan {
  namespace vulkan {

    static void validate(VkResult result) {
      if (result != VK_SUCCESS) {
        fan::throw_error("function failed");
      }
    }

    namespace core {
      struct memory_t;
    }

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

    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 1;

    struct context_t;
    struct viewport_t;
    struct shader_t;

    struct pipeline_t {

      struct properties_t {
        uint32_t descriptor_layout_count;
        VkDescriptorSetLayout* descriptor_layout;
        void* shader;
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

      void open(fan::vulkan::context_t& context, VkDescriptorPool descriptor_pool, std::array<fan::vulkan::write_descriptor_set_t, count> properties);
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
      VkDescriptorSet m_descriptor_set[fan::vulkan::MAX_FRAMES_IN_FLIGHT];
    };

    static uint32_t makeAccessMaskPipelineStageFlags(uint32_t accessMask) {
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
      void open(auto& context, const properties_t& p) {
        fan::vulkan::createImage(
          context,
          p.swap_chain_size,
          p.format,
          VK_IMAGE_TILING_OPTIMAL,
          p.usage_flags,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          image,
          memory
        );
        image_view = context.createImageView(image, p.format, p.aspect_flags);
        format = p.format;
      }
      void close(auto& context) {
        vkDestroyImageView(context.device, image_view, nullptr);
        vkDestroyImage(context.device, image, nullptr);
        vkFreeMemory(context.device, memory, nullptr);
      }

      void transition_image_layout(auto& context, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = context.beginSingleTimeCommands(&context);

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = old_layout;
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

        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;

        vkCmdPipelineBarrier(
          commandBuffer,
          sourceStage, destinationStage,
          0,
          0, nullptr,
          0, nullptr,
          1, &barrier
        );

        context.endSingleTimeCommands(&context, commandBuffer);

        old_layout = newLayout;
      }

      VkFormat format;

      VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;

      VkImage image;
      VkImageView image_view;
      VkDeviceMemory memory;
    };
  }
}

#if defined(loco_window)

#include "themes_list_builder_settings.h"
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#include <BLL/BLL.h>

#include "themes_list_builder_settings.h"
#define BLL_set_declare_NodeReference 0
#define BLL_set_declare_rest 1
#include <BLL/BLL.h>

#include "viewport_list_builder_settings.h"
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#include <BLL/BLL.h>

namespace fan {

  namespace vulkan {

    struct viewport_t {
      viewport_t() {
        viewport_reference.sic();
      }

      void open();
      void close();

      fan::vec2 get_position() const
      {
        return viewport_position;
      }

      fan::vec2 get_size() const
      {
        return viewport_size;
      }

      void set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);
      static void set_viewport(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);

      bool inside(const fan::vec2& position) const {
        return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport_position - viewport_size / 2, viewport_size * 2);
      }

      fan::vec2 viewport_position;
      fan::vec2 viewport_size;

      fan::vulkan::viewport_list_NodeReference_t viewport_reference;
    };

  }
}

#include "viewport_list_builder_settings.h"
#define BLL_set_declare_NodeReference 0
#define BLL_set_declare_rest 1
#include <BLL/BLL.h>

namespace fan_2d {
  namespace graphics {
    namespace gui {
      struct theme_t;
    }
  }
}

#endif

namespace fan {
  namespace vulkan {

    struct context_t {


      static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
      static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

      context_t() {
        createInstance();
        setupDebugMessenger();
        createInstance();
        setupDebugMessenger();
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
      }
#if defined(loco_window)
      context_t(fan::window_t* window) {
        window->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
          recreateSwapChain(d.size);
          });

        createInstance();
        setupDebugMessenger();
        createSurface(window->get_handle());
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain(window->get_size());
        createImageViews();
        createRenderPass();
        createCommandPool();
#if defined(loco_wboit)
        create_wboit_views();
#endif
        create_loco_framebuffer();
        createDepthResources();
        createFramebuffers();
        createCommandBuffers();
        createSyncObjects();
      }
#endif

      ~context_t() {
        cleanupSwapChain();

        vkDestroyRenderPass(device, renderPass, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
          vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
          vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
          vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

#if fan_debug >= fan_debug_high
        if (supports_validation_layers) {
          DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
#endif

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
      }

      void cleanupSwapChain() {
        vai_depth.close(*this);

        for (auto framebuffer : swapChainFramebuffers) {
          vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto imageView : swapChainImageViews) {
          vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
      }

#if defined(loco_window)
      void recreateSwapChain(const fan::vec2i& window_size) {

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain(window_size);
        createImageViews();
        createDepthResources();
        createFramebuffers();
      }
#endif

      void createInstance() {
#if fan_debug >= fan_debug_high
        if (!checkValidationLayerSupport()) {
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

        auto extensions = getRequiredExtensions();
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

          populateDebugMessengerCreateInfo(debugCreateInfo);
          createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
#endif

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
          throw std::runtime_error("failed to create instance!");
        }
      }



      void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
      }

      void setupDebugMessenger() {
#if fan_debug < fan_debug_high
        return;
#endif

        if (!supports_validation_layers) {
          return;
        }

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
          throw std::runtime_error("failed to set up debug messenger!");
        }
      }

#if defined(loco_window)
      void createSurface(auto handle) {
#ifdef fan_platform_windows

        VkWin32SurfaceCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        create_info.hwnd = handle;

        create_info.hinstance = GetModuleHandle(nullptr);

        vkCreateWin32SurfaceKHR(instance, &create_info, nullptr, &surface);

#elif defined(fan_platform_unix)

        VkXlibSurfaceCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        create_info.window = handle;
        create_info.dpy = fan::sys::get_display();

        validate(vkCreateXlibSurfaceKHR(instance, &create_info, nullptr, &surface));

#endif
      }
#endif

      void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
          throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
          if (isDeviceSuitable(device)) {
            physicalDevice = device;
            break;
          }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
          throw std::runtime_error("failed to find a suitable GPU!");
        }
      }

      void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
          indices.graphicsFamily.value(),
        #if defined(loco_window)
          indices.presentFamily.value()
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

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;

        //deviceFeatures.vertexPipelineStoresAndAtomics = VK_TRUE;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = queueCreateInfos.size();
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = deviceExtensions.size();
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

#if fan_debug >= fan_debug_high
        if (supports_validation_layers) {
          createInfo.enabledLayerCount = validationLayers.size();
          createInfo.ppEnabledLayerNames = validationLayers.data();
        }
#endif

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
          throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
#if defined(loco_window)
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
#endif
      }

#if defined(loco_window)
      void createSwapChain(const fan::vec2ui& framebuffer_size) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(framebuffer_size, swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
          imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
          createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
          createInfo.queueFamilyIndexCount = 2;
          createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
          createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        //createInfo.imageUsage = ;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
          throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swap_chain_size = fan::vec2(extent.width, extent.height);
      }
#endif

      VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
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

      void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (uint32_t i = 0; i < swapChainImages.size(); i++) {
          swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        }
      }

      void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef[2]{};
        colorAttachmentRef[0].attachment = 0;
        colorAttachmentRef[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachmentRef[1].attachment = 1;
        colorAttachmentRef[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 4;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription fbo_color_attachment[2]{};

        fbo_color_attachment[0].format = swapChainImageFormat;
        fbo_color_attachment[0].samples = VK_SAMPLE_COUNT_1_BIT;
        fbo_color_attachment[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        fbo_color_attachment[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        fbo_color_attachment[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        fbo_color_attachment[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        fbo_color_attachment[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        fbo_color_attachment[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        fbo_color_attachment[1].format = swapChainImageFormat;
        fbo_color_attachment[1].samples = VK_SAMPLE_COUNT_1_BIT;
        fbo_color_attachment[1].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        fbo_color_attachment[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        fbo_color_attachment[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        fbo_color_attachment[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        fbo_color_attachment[1].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        fbo_color_attachment[1].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference inputAttachmentRef[] = {
          {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
          },
          //{
          //  .attachment = 1,
          //  .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
          //}
        };

        VkAttachmentReference subpasscolor_locofbo_attachments[]{
          {
            .attachment = 2,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          },
        };

        //subpasscolor_locofbo_attachments[1].attachment = 3;
        //subpasscolor_locofbo_attachments[1].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass[]{
          {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = std::size(colorAttachmentRef),
            .pColorAttachments = colorAttachmentRef,
            .pDepthStencilAttachment = &depthAttachmentRef
          },
          {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = std::size(inputAttachmentRef),
            .pInputAttachments = inputAttachmentRef,
            .colorAttachmentCount = std::size(subpasscolor_locofbo_attachments),
            .pColorAttachments = subpasscolor_locofbo_attachments,
            .pDepthStencilAttachment = &depthAttachmentRef,
          }
        };

        VkSubpassDependency subpassDependencies[4]{};
        subpassDependencies[0].srcSubpass = 0;
        subpassDependencies[0].dstSubpass = VK_SUBPASS_EXTERNAL;
        subpassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpassDependencies[0].srcAccessMask = 0;
        subpassDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        subpassDependencies[1].srcSubpass = 0;
        subpassDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        subpassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpassDependencies[1].srcAccessMask = 0;
        subpassDependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        subpassDependencies[2].srcSubpass = 0;
        subpassDependencies[2].dstSubpass = 1;
        subpassDependencies[2].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        subpassDependencies[2].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpassDependencies[2].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        subpassDependencies[2].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        subpassDependencies[3].srcSubpass = 0;
        subpassDependencies[3].dstSubpass = 1;
        subpassDependencies[3].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        subpassDependencies[3].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpassDependencies[3].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        subpassDependencies[3].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkAttachmentDescription attachments[] = {
          fbo_color_attachment[0],
          fbo_color_attachment[0],
          colorAttachment,
          fbo_color_attachment[1],
          depthAttachment,
        };
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = std::size(attachments);
        renderPassInfo.pAttachments = attachments;
        renderPassInfo.subpassCount = std::size(subpass);
        renderPassInfo.pSubpasses = subpass;
        renderPassInfo.dependencyCount = std::size(subpassDependencies);
        renderPassInfo.pDependencies = subpassDependencies;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
          throw std::runtime_error("failed to create renderpass");
        }
      }

      void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
          VkImageView attachments[] = {
            #if defined(loco_wboit)
              vai_wboit_color.image_view,
              vai_wboit_reveal.image_view,
            #endif

            vai_bitmap[0].image_view,
            vai_bitmap[1].image_view,
            swapChainImageViews[i],
            vai_bitmap[1].image_view,
            vai_depth.image_view,
          };

          VkFramebufferCreateInfo framebufferInfo{};
          framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
          framebufferInfo.renderPass = renderPass;
          framebufferInfo.attachmentCount = std::size(attachments);
          framebufferInfo.pAttachments = attachments;
          framebufferInfo.width = swap_chain_size.x;
          framebufferInfo.height = swap_chain_size.y;
          framebufferInfo.layers = 1;

          if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
          }
        }
      }

      void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
          throw std::runtime_error("failed to create graphics command pool!");
        }
      }

      void create_loco_framebuffer();

      void create_wboit_views();
      void createDepthResources();

      VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
        for (VkFormat format : candidates) {
          VkFormatProperties props;
          vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

          if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
          }
          else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
          }
        }

        throw std::runtime_error("failed to find supported format!");
      }

      VkFormat findDepthFormat() {
        return findSupportedFormat(
          { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
          VK_IMAGE_TILING_OPTIMAL,
          VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
      }

      bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
      }

      void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
          throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
          throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
      }

      static VkCommandBuffer beginSingleTimeCommands(fan::vulkan::context_t* context) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = context->commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(context->device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
      }

      static void endSingleTimeCommands(fan::vulkan::context_t* context, VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(context->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(context->graphicsQueue);

        vkFreeCommandBuffers(context->device, context->commandPool, 1, &commandBuffer);
      }

      void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands(this);

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(this, commandBuffer);
      }

      uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
          if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
          }
        }

        throw std::runtime_error("failed to find suitable memory type!");
      }

      void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
          throw std::runtime_error("failed to allocate command buffers!");
        }
      }

      void bind_draw(
        const fan::vulkan::pipeline_t& pipeline,
        uint32_t descriptor_count,
        VkDescriptorSet* descriptor_sets) {
        vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.m_pipeline);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent.width = swap_chain_size.x;
        scissor.extent.height = swap_chain_size.y;
        vkCmdSetScissor(commandBuffers[currentFrame], 0, 1, &scissor);

        vkCmdBindDescriptorSets(
          commandBuffers[currentFrame],
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
        vkCmdDraw(commandBuffers[currentFrame], vertex_count, instance_count, 0, first_instance);
      }

      void draw(
        uint32_t vertex_count,
        uint32_t instance_count,
        uint32_t first_instance,
        const fan::vulkan::pipeline_t& pipeline,
        uint32_t descriptor_count,
        VkDescriptorSet* descriptor_sets
      ) {
        bind_draw(pipeline, descriptor_count, descriptor_sets);
        bindless_draw(vertex_count, instance_count, first_instance);
      }

      void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
          if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
          }
        }
      }

      //void updateUniformBuffer(uint32_t currentImage) {
      //  static auto startTime = std::chrono::high_resolution_clock::now();

      //  auto currentTime = std::chrono::high_resolution_clock::now();
      //  float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

      //  fan::camera camera;
      //  //camera.get
      //  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
      //  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);
      //  UniformBufferObject ubo{};
      //  ubo.model = fan::mat4(1);
      //  ubo.view = fan::mat4(1);
      //  ubo.proj = fan::math::ortho<fan::mat4>(
      //    ortho_x.x,
      //    ortho_x.y,
      //    ortho_y.y,
      //    ortho_y.x,
      //    -1,
      //    0x10000
      //  );
      //  //ubo.proj[1][1] *= -1;

      //  void* data;
      //  vkMapMemory(device, uniform_block.common.memory[currentImage].device_memory, 0, sizeof(ubo), 0, &data);
      //  memcpy(data, &ubo, sizeof(ubo));
      //  vkUnmapMemory(device, uniform_block.common.memory[currentImage].device_memory);
      //}
#if defined(loco_window)
      void begin_render(fan::window_t* window);

      void end_render(fan::window_t* window) {
#if defined(loco_wboit)
        vkCmdNextSubpass(commandBuffers[currentFrame], VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, render_fullscreen_pl.m_pipeline);
        vkCmdDraw(commandBuffers[currentFrame], 6, 1, 0, 0);
#endif

        fan::vulkan::viewport_t::set_viewport(0, swap_chain_size, swap_chain_size);

        // render_fullscreen_pl loco fbo?
        vkCmdNextSubpass(commandBuffers[currentFrame], VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, render_fullscreen_pl.m_pipeline);

        vkCmdDraw(commandBuffers[currentFrame], 6, 1, 0, 0);


        vkCmdEndRenderPass(commandBuffers[currentFrame]);

        if (vkEndCommandBuffer(commandBuffers[currentFrame]) != VK_SUCCESS) {
          fan::throw_error("failed to record command buffer!");
        }

        command_buffer_in_use = false;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
          throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;

        presentInfo.pImageIndices = &image_index;
        auto result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
          recreateSwapChain(window->get_size());
        }
        else if (result != VK_SUCCESS) {
          throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
      }
#endif

      void begin_compute_shader() {
        //?
        //vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffers[currentFrame], &beginInfo) != VK_SUCCESS) {
          fan::throw_error("failed to begin recording command buffer!");
        }

        command_buffer_in_use = true;
      }

      void end_compute_shader() {
        if (vkEndCommandBuffer(commandBuffers[currentFrame]) != VK_SUCCESS) {
          fan::throw_error("failed to record command buffer!");
        }

        command_buffer_in_use = false;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
          throw std::runtime_error("failed to submit draw command buffer!");
        }
      }

      VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
          throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
      }

      VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
          // VK_FORMAT_B8G8R8A8_SRGB

          if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
          }
        }

        return availableFormats[0];
      }

      VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& available_present_mode : availablePresentModes) {
          if (available_present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR && !vsync) {
            return VK_PRESENT_MODE_IMMEDIATE_KHR;
          }
          else if (available_present_mode == VK_PRESENT_MODE_FIFO_KHR && vsync) {
            return VK_PRESENT_MODE_FIFO_KHR;
          }
        }

        return availablePresentModes[0];
      }

      VkExtent2D chooseSwapExtent(const fan::vec2ui& framebuffer_size, const VkSurfaceCapabilitiesKHR& capabilities) {
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

      SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

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
          details.presentModes.resize(presentModeCount);
          vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
      }

      bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate
#if defined(loco_window)
          = false;
        if (extensionsSupported) {
          SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
          swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }
#else
          = true;
#endif

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
      }

      bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
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

      QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
          if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
          }

          VkBool32 presentSupport = false;

#if defined(loco_window)
          vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

          if (presentSupport) {
            indices.presentFamily = i;
          }
#endif

          if (indices.isComplete()) {
            break;
          }

          i++;
        }

        return indices;
      }

      std::vector<fan::string> getRequiredExtensions() {

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

        std::vector<fan::string> extension_str(available_extensions.size());

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

      bool checkValidationLayerSupport() {
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

      static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
          throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
      }

      static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData
      ) {
        if (pCallbackData->pMessageIdName && fan::string(pCallbackData->pMessageIdName) == "Loader Message") {
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
        recreateSwapChain(window->get_size());
      }
#endif

      VkInstance instance;
      VkDebugUtilsMessengerEXT debugMessenger;
      VkSurfaceKHR surface;

      VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
      VkDevice device;

      VkQueue graphicsQueue;
#if defined(loco_window)
      VkQueue presentQueue;
#endif

      VkSwapchainKHR swapChain;
      std::vector<VkImage> swapChainImages;
      VkFormat swapChainImageFormat;
      fan::vec2 swap_chain_size;
      std::vector<VkImageView> swapChainImageViews;
      std::vector<VkFramebuffer> swapChainFramebuffers;

      VkRenderPass renderPass;

      VkCommandPool commandPool;

      vai_t vai_depth;
      vai_t vai_bitmap[2];
#if defined(loco_wboit)
      vai_t vai_wboit_color;
      vai_t vai_wboit_reveal;
#endif

      std::vector<VkCommandBuffer> commandBuffers;

      std::vector<VkSemaphore> imageAvailableSemaphores;
      std::vector<VkSemaphore> renderFinishedSemaphores;
      std::vector<VkFence> inFlightFences;
      uint32_t currentFrame = 0;

#if defined(loco_window)
      fan::vulkan::viewport_list_t viewport_list;
      fan::vulkan::theme_list_t theme_list;
#endif

      bool vsync = true;
      uint32_t image_index;

      fan::vulkan::pipeline_t render_fullscreen_pl;

      bool command_buffer_in_use = false;

      bool supports_validation_layers = true;
    };
  }
}

#include "memory.h"
#include "uniform_block.h"
#include "ssbo.h"

namespace fan {
  namespace vulkan {

    static void createImage(fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
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
      allocInfo.memoryTypeIndex = fan::vulkan::core::findMemoryType(context, memRequirements.memoryTypeBits, properties);

      if (vkAllocateMemory(context.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
      }

      vkBindImageMemory(context.device, image, imageMemory, 0);
    }

    inline void context_t::create_loco_framebuffer() {
      vai_t::properties_t p;
      p.format = swapChainImageFormat;
      p.swap_chain_size = swap_chain_size;
      p.usage_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      p.aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT;
      vai_bitmap[0].open(*this, p);
      vai_bitmap[0].transition_image_layout(*this, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      p.format = VK_FORMAT_B8G8R8A8_UNORM; // TODO should it be VK_FORMAT_R8_UINT?
      vai_bitmap[1].open(*this, p);
      vai_bitmap[1].transition_image_layout(*this, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

#if defined(loco_wboit)
    void context_t::create_wboit_views() {
      vai_t::properties_t p;
      p.format = VK_FORMAT_R16G16B16A16_SFLOAT;
      p.swap_chain_size = swap_chain_size;
      p.usage_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      p.aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT;
      vai_wboit_color.open(this, p);
      vai_wboit_color.transition_image_layout(this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

      p.format = VK_FORMAT_R16_SFLOAT;
      vai_wboit_reveal.open(this, p);
      vai_wboit_reveal.transition_image_layout(this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
#endif

    template <uint32_t count>
    inline void descriptor_t<count>::open(fan::vulkan::context_t& context, VkDescriptorPool descriptor_pool, std::array<fan::vulkan::write_descriptor_set_t, count> properties) {
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

      std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts;
      for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        layouts[i] = m_layout;
      }
      VkDescriptorSetAllocateInfo allocInfo{};
      allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocInfo.descriptorPool = descriptor_pool;
      allocInfo.descriptorSetCount = layouts.size();
      allocInfo.pSetLayouts = layouts.data();

      validate(vkAllocateDescriptorSets(context.device, &allocInfo, m_descriptor_set));
    }

    template <uint32_t count>
    inline void descriptor_t<count>::close(fan::vulkan::context_t& context) {
      vkDestroyDescriptorSetLayout(context.device, m_layout, 0);
    }

    template <uint32_t count>
    inline void descriptor_t<count>::update(
      fan::vulkan::context_t& context,
      uint32_t n,
      uint32_t begin,
      uint32_t texture_n,
      uint32_t texture_begin
    ) {
      VkDescriptorBufferInfo bufferInfo[count * MAX_FRAMES_IN_FLIGHT]{};

      for (size_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; frame++) {

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

inline void fan::vulkan::context_t::createDepthResources() {
  vai_t::properties_t p;
  p.swap_chain_size = swap_chain_size;
  p.format = findDepthFormat();
  p.usage_flags = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  p.aspect_flags = VK_IMAGE_ASPECT_DEPTH_BIT;
  vai_depth.open(*this, p);
}


inline void fan::vulkan::pipeline_t::close(fan::vulkan::context_t& context) {
  vkDestroyPipeline(context.device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(context.device, m_layout, nullptr);
}