#pragma once

#if defined(fan_platform_windows)
  #define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
  #define VK_USE_PLATFORM_XLIB_KHR
#endif

#include <vulkan/vulkan.h>
#pragma comment(lib, "lib/vulkan/vulkan-1.lib")

#include <vector>
#include <set>

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
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

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  bool isComplete() {
    return graphicsFamily.has_value() && presentFamily.has_value();
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

    void validate(VkResult result) {
      if (result != VK_SUCCESS) {
        fan::throw_error("function failed");
      }
    }

    namespace core {
      struct uniform_block_common_t;
      struct memory_t;
    }

    template <uint16_t count>
    struct descriptor_set_layout_t {
      struct write_descriptor_set_t {
        // glsl layout binding
        uint32_t binding;

        // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        // VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
        VkDescriptorType type;

        // VK_SHADER_STAGE_VERTEX_BIT
        // VK_SHADER_STAGE_FRAGMENT_BIT
        VkShaderStageFlags flags;

        fan::vulkan::core::uniform_block_common_t* block_common;

        // hardcode descriptor count to 1 in both layoutset and descriptor
        // init dstarrayelement with 0
        // init layoutbinding to 0

      }write_descriptor_sets[count];
    };

    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    struct context_t;
    struct viewport_t;
    struct matrices_t;
    struct image_t;

    struct cid_t {
      uint16_t bm_id;
      uint16_t block_id;
      uint8_t instance_id;
    };

  }
}

#include "vk_descriptor.h"
#include "vk_pipeline.h"

#include "viewport_list_builder_settings.h"
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#include _FAN_PATH(BLL/BLL.h)

namespace fan {

  namespace vulkan {

    struct viewport_t {

      void open(fan::vulkan::context_t* context);
      void close(fan::vulkan::context_t* context);

      fan::vec2 get_position() const
      {
        return viewport_position;
      }

      fan::vec2 get_size() const
      {
        return viewport_size;
      }

      void set(fan::vulkan::context_t* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size);

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
#include _FAN_PATH(BLL/BLL.h)

fan::vulkan::viewport_list_NodeReference_t::viewport_list_NodeReference_t(fan::vulkan::viewport_t* viewport) {
  NRI = viewport->viewport_reference.NRI;
}

#include "matrices_list_builder_settings.h"
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#include _FAN_PATH(BLL/BLL.h)

namespace fan {
  namespace vulkan {
    struct matrices_t {

      void open(fan::vulkan::context_t* context);
      void close(fan::vulkan::context_t* context);

      fan::vec3 get_camera_position() const {
        return camera_position;
      }
      void set_camera_position(const fan::vec3& cp) {
        camera_position = cp;

        m_view[3][0] = 0;
        m_view[3][1] = 0;
        m_view[3][2] = 0;
        m_view = m_view.translate(camera_position);
        fan::vec3 position = m_view.get_translation();
        constexpr fan::vec3 front(0, 0, 1);

        m_view = fan::math::look_at_left<fan::mat4>(position, position + front, fan::camera::world_up);
      }

      void set_ortho(const fan::vec2& x, const fan::vec2& y) {
        m_projection = fan::math::ortho<fan::mat4>(
          x.x,
          x.y,
          y.y,
          y.x,
          -1,
          0x10000
          );
        coordinates.left = x.x;
        coordinates.right = x.y;
        coordinates.bottom = y.y;
        coordinates.top = y.x;

        m_view[3][0] = 0;
        m_view[3][1] = 0;
        m_view[3][2] = 0;
        m_view = m_view.translate(camera_position);
        fan::vec3 position = m_view.get_translation();
        constexpr fan::vec3 front(0, 0, 1);

        m_view = fan::math::look_at_left<fan::mat4>(position, position + front, fan::camera::world_up);
      }

      fan::mat4 m_projection;
      // temporary
      fan::mat4 m_view;

      fan::vec3 camera_position;

      union {
        struct {
          f32_t left;
          f32_t right;
          f32_t top;
          f32_t bottom;
        };
        fan::vec4 v;
      }coordinates;

      matrices_list_NodeReference_t matrices_reference;
    };

    static void open_matrices(fan::vulkan::context_t* context, matrices_t* matrices, const fan::vec2& x, const fan::vec2& y);
  }
}

#include "matrices_list_builder_settings.h"
#define BLL_set_declare_NodeReference 0
#define BLL_set_declare_rest 1
#include _FAN_PATH(BLL/BLL.h)

fan::vulkan::matrices_list_NodeReference_t::matrices_list_NodeReference_t(fan::vulkan::matrices_t* matrices) {
  NRI = matrices->matrices_reference.NRI;
}

namespace fan {
	namespace vulkan {

		struct context_t {


      static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
      static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

      void open() {

        createInstance();
        setupDebugMessenger();
      }

      void bind_to_window(fan::window_t* window) {
        createSurface(window->get_handle());
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain(window->get_size());
        createImageViews();
        createRenderPass();
        
        createCommandPool();
        createDepthResources();
        createFramebuffers();
        //createTextureImage();
        //createTextureImageView();
        //createTextureSampler();
        createCommandBuffers();
        createSyncObjects();

        pipelines.open();
        descriptor_sets.open(this);
      }

      void cleanupSwapChain() {
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        for (auto framebuffer : swapChainFramebuffers) {
          vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto imageView : swapChainImageViews) {
          vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
      }

      void recreateSwapChain(const fan::vec2i& window_size) {

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain(window_size);
        createImageViews();
        createDepthResources();
        createFramebuffers();
      }

      void createInstance() {
        #if fan_debug >= fan_debug_high
        if (!checkValidationLayerSupport()) {
          fan::print_warning("validation layers not supported");
          supports_validation_layers = false;
        }
        #endif

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

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
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
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
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

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

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        #if fan_debug >= fan_debug_high
          if (supports_validation_layers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
          }
        #endif

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
          throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
      }

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
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

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

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
          throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
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

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
          throw std::runtime_error("failed to create render pass!");
        }
      }

      void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
          std::array<VkImageView, 2> attachments = {
              swapChainImageViews[i],
              depthImageView
          };

          VkFramebufferCreateInfo framebufferInfo{};
          framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
          framebufferInfo.renderPass = renderPass;
          framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
          framebufferInfo.pAttachments = attachments.data();
          framebufferInfo.width = swapChainExtent.width;
          framebufferInfo.height = swapChainExtent.height;
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

      void createDepthResources() {
        VkFormat depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
      }

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

      void createTextureImage() {
        VkDeviceSize imageSize = 4 * 4 * 4;

        uint8_t pixels[]{
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255,
          255, 0, 255, 255
        };

        if (!pixels) {
          throw std::runtime_error("failed to load texture image!");
        }

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);

        createImage(4, 4, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(4), static_cast<uint32_t>(4));
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
      }

      void createTextureImageView() {
        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
      }

      void createTextureSampler() {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
          throw std::runtime_error("failed to create texture sampler!");
        }
      }

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

      void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
          throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
          throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
      }

      void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

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
          commandBuffer,
          sourceStage, destinationStage,
          0,
          0, nullptr,
          0, nullptr,
          1, &barrier
        );

        endSingleTimeCommands(commandBuffer);
      }

      void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

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
            width,
            height,
            1
        };

        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endSingleTimeCommands(commandBuffer);
      }

      void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * std::numeric_limits<uint16_t>::max();

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptor_sets.descriptorPool) != VK_SUCCESS) {
          throw std::runtime_error("failed to create descriptor pool!");
        }
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

      VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
      }

      void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
      }

      void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
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

      void draw(
        uint32_t vertex_count, 
        uint32_t instance_count, 
        uint32_t first_instance, 
        pipelines_t::nr_t pipeline_nr,
        VkDescriptorSet* descriptor_set
      ) {
        vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.pipeline_list[pipeline_nr].pipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffers[currentFrame], 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffers[currentFrame], 0, 1, &scissor);

        vkCmdBindDescriptorSets(
          commandBuffers[currentFrame],
          VK_PIPELINE_BIND_POINT_GRAPHICS,
          pipelines.pipeline_list[pipeline_nr].pipeline_layout, 
          0, 
          1,
          descriptor_set,
          0,
          nullptr
        );

        vkCmdDraw(commandBuffers[currentFrame], vertex_count, instance_count, 0, first_instance);
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

      void render(fan::window_t* window, fan::function_t<void()> draw_lambda) {

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);

        draw_lambda();

        //draw(6, 1, 0, (pipelines_t::nr_t)2, &descriptor_sets.descriptor_list[(descriptor_sets_t::nr_t)2].descriptor_set[currentFrame]);

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
          if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
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
              static_cast<uint32_t>(framebuffer_size.x),
              static_cast<uint32_t>(framebuffer_size.y)
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

        bool swapChainAdequate = false;
        if (extensionsSupported) {
          SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
          swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

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
          vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

          if (presentSupport) {
            indices.presentFamily = i;
          }

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

      static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
      }

      void set_vsync(fan::window_t* window, bool flag) {
        vsync = flag;
        recreateSwapChain(window->get_size());
      }

      VkInstance instance;
      VkDebugUtilsMessengerEXT debugMessenger;
      VkSurfaceKHR surface;

      VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
      VkDevice device;

      VkQueue graphicsQueue;
      VkQueue presentQueue;

      VkSwapchainKHR swapChain;
      std::vector<VkImage> swapChainImages;
      VkFormat swapChainImageFormat;
      VkExtent2D swapChainExtent;
      std::vector<VkImageView> swapChainImageViews;
      std::vector<VkFramebuffer> swapChainFramebuffers;

      VkRenderPass renderPass;

      VkCommandPool commandPool;

      VkImage depthImage;
      VkDeviceMemory depthImageMemory;
      VkImageView depthImageView;

      VkImage textureImage;
      VkDeviceMemory textureImageMemory;
      VkImageView textureImageView;
      VkSampler textureSampler;

      std::vector<VkCommandBuffer> commandBuffers;

      std::vector<VkSemaphore> imageAvailableSemaphores;
      std::vector<VkSemaphore> renderFinishedSemaphores;
      std::vector<VkFence> inFlightFences;
      uint32_t currentFrame = 0;

      fan::vulkan::viewport_list_t viewport_list;
      fan::vulkan::matrices_list_t matrices_list;

      pipelines_t pipelines;

      descriptor_sets_t descriptor_sets;

      bool vsync = true;
      uint32_t image_index;

      bool command_buffer_in_use = false;

      bool supports_validation_layers = true;
		};
	}
}

#include "uniform_block.h"
#include "vk_shader.h"

void fan::vulkan::pipelines_t::close(fan::vulkan::context_t* context) {
  auto it = pipeline_list.GetNodeFirst();
  while (it != pipeline_list.dst) {
    auto data = pipeline_list.GetNodeByReference(it)->data;
    vkDestroyPipeline(context->device, data.pipeline, nullptr);
    it = it.Next(&pipeline_list);
  }
}

fan::vulkan::pipelines_t::nr_t fan::vulkan::pipelines_t::push(fan::vulkan::context_t* context, const fan::vulkan::pipelines_t::properties_t& p) {
	auto nr = pipeline_list.NewNode();
	auto& node = pipeline_list[nr];

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
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
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  std::vector<VkDynamicState> dynamicStates = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR
  };
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = p.descriptor_set_layout;

  if (vkCreatePipelineLayout(context->device, &pipelineLayoutInfo, nullptr, &node.pipeline_layout) != VK_SUCCESS) {
    fan::throw_error("failed to create pipeline layout!");
  }

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = p.shader->shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = node.pipeline_layout;
  pipelineInfo.renderPass = context->renderPass;
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

  if (vkCreateGraphicsPipelines(context->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &node.pipeline) != VK_SUCCESS) {
    fan::throw_error("failed to create graphics pipeline!");
  }

	return nr;
}

inline void fan::vulkan::viewport_t::open(fan::vulkan::context_t* context) {
  viewport_reference = context->viewport_list.NewNode();
  context->viewport_list[viewport_reference].viewport_id = this;
}

inline void fan::vulkan::viewport_t::close(fan::vulkan::context_t* context) {
  context->viewport_list.Recycle(viewport_reference);
}

void fan::vulkan::viewport_t::set(fan::vulkan::context_t* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  viewport_position = viewport_position_;
  viewport_size = viewport_size_;

  VkViewport viewport{};
  viewport.x = viewport_position.x;
  viewport.y = window_size.y - viewport_size_.y - viewport_position.y;
  viewport.width = viewport_size.x;
  viewport.height = viewport_size.y;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  if (!context->command_buffer_in_use) {
    if (vkBeginCommandBuffer(context->commandBuffers[context->currentFrame], &beginInfo) != VK_SUCCESS) {
      fan::throw_error("failed to begin recording command buffer!");
    }
    context->command_buffer_in_use = true;
  }

  vkCmdSetViewport(context->commandBuffers[context->currentFrame], 0, 1, &viewport);

  if (!context->command_buffer_in_use) {
    if (vkEndCommandBuffer(context->commandBuffers[context->currentFrame]) != VK_SUCCESS) {
      fan::throw_error("failed to record command buffer!");
    }
    context->command_buffer_in_use = false;
  }
}

void fan::vulkan::matrices_t::open(fan::vulkan::context_t* context) {
  m_view = fan::mat4(1);
  camera_position = 0;
  matrices_reference = context->matrices_list.NewNode();
  context->matrices_list[matrices_reference].matrices_id = this;
}
void fan::vulkan::matrices_t::close(fan::vulkan::context_t* context) {
  context->matrices_list.Recycle(matrices_reference);
}

void fan::vulkan::open_matrices(fan::vulkan::context_t* context, fan::vulkan::matrices_t* matrices, const fan::vec2& x, const fan::vec2& y) {
  matrices->open(context);
  matrices->set_ortho(fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
}

void fan::vulkan::descriptor_sets_t::open(fan::vulkan::context_t* context) {
  std::array<VkDescriptorPoolSize, 2> poolSizes{};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

  //poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  //poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * std::numeric_limits<uint16_t>::max();

  if (vkCreateDescriptorPool(context->device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void fan::vulkan::descriptor_sets_t::close(fan::vulkan::context_t* context) {
  vkDestroyDescriptorPool(context->device, descriptorPool, nullptr);
}

template <uint16_t count>
fan::vulkan::descriptor_sets_t::layout_nr_t 
fan::vulkan::descriptor_sets_t::push_layout(
  fan::vulkan::context_t* context,
  fan::vulkan::descriptor_set_layout_t<count> properties
) {
  auto nr = descriptor_layout_list.NewNode();
  auto& node = descriptor_layout_list[nr];

  VkDescriptorSetLayoutBinding uboLayoutBinding[count]{};
  for (uint16_t i = 0; i < count; ++i) {
    uboLayoutBinding[i].binding = properties.write_descriptor_sets[i].binding;
    uboLayoutBinding[i].descriptorCount = 1;
    uboLayoutBinding[i].descriptorType = properties.write_descriptor_sets[i].type;
    uboLayoutBinding[i].stageFlags = properties.write_descriptor_sets[i].flags;
  }

  //VkDescriptorSetLayoutBinding samplerLayoutBinding{};
  //samplerLayoutBinding.binding = 2;
  //samplerLayoutBinding.descriptorCount = 1;
  //samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  //samplerLayoutBinding.pImmutableSamplers = nullptr;
  //samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  //std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = std::size(uboLayoutBinding);
  layoutInfo.pBindings = uboLayoutBinding;

  if (vkCreateDescriptorSetLayout(context->device, &layoutInfo, nullptr, &node.layout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }

  return nr;
}

template<uint16_t count>
inline fan::vulkan::descriptor_sets_t::nr_t
fan::vulkan::descriptor_sets_t::push(
  fan::vulkan::context_t* context,
  fan::vulkan::descriptor_sets_t::layout_nr_t layout_nr,
  fan::vulkan::descriptor_set_layout_t<count> properties
) {
  auto nr = descriptor_list.NewNode();
  auto& node = descriptor_list[nr];

  std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptor_layout_list[layout_nr].layout);
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
  allocInfo.pSetLayouts = layouts.data();

  if (vkAllocateDescriptorSets(context->device, &allocInfo, node.descriptor_set) != VK_SUCCESS) {
    fan::throw_error("failed to allocate descriptor sets!");
  }

  VkDescriptorBufferInfo bufferInfo[count * MAX_FRAMES_IN_FLIGHT]{};

  for (size_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; frame++) {
    for (uint32_t j = 0; j < count; ++j) {
      bufferInfo[frame * count + j].buffer = properties.write_descriptor_sets[j].block_common->memory[frame].uniform_buffer;
      bufferInfo[frame * count + j].offset = 0;
      bufferInfo[frame * count + j].range = properties.write_descriptor_sets[j].block_common->m_buffer_size;
    }
  }

  for (size_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; frame++) {

    std::array<VkWriteDescriptorSet, count> descriptorWrites{};

    for (uint32_t j = 0; j < count; ++j) {

      descriptorWrites[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[j].dstSet = node.descriptor_set[frame];
      descriptorWrites[j].descriptorType = properties.write_descriptor_sets[j].type;
      descriptorWrites[j].descriptorCount = count;
      descriptorWrites[j].pBufferInfo = bufferInfo;

      // ADD IMAGEVIEW AND SAMPLER TO PER BLOCK

      /*VkDescriptorImageInfo imageInfo{};
      *
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo.imageView = textureImageView;
      imageInfo.sampler = vksampler;*/

      //descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      //descriptorWrites[2].dstSet = node.descriptor_set[i];
      //descriptorWrites[2].dstBinding = 2;
      //descriptorWrites[2].dstArrayElement = 0;
      //descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      //descriptorWrites[2].descriptorCount = 1;
      //descriptorWrites[2].pImageInfo = &imageInfo;

    }
    vkUpdateDescriptorSets(context->device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
  }
  return nr;
}

void fan::vulkan::core::uniform_write_queue_t::process(fan::vulkan::context_t* context) {
  uint32_t it = write_queue.begin();

  while (it != write_queue.end()) {

    uint64_t src = write_queue[it]->m_min_edit;
    uint64_t dst = write_queue[it]->m_max_edit;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
      auto device_memory = write_queue[it]->memory[i].device_memory;
      uint8_t* buffer = (uint8_t*)&write_queue[it][1];

      //buffer += src;

      // UPDATE BUFFER HERE

      void* data;
      validate(vkMapMemory(context->device, device_memory, 0, dst - src, 0, &data));
      //data += src; ??
      memcpy(data, buffer, dst - src);
      // unnecessary?
      vkUnmapMemory(context->device, device_memory);
      //fan::vulkan::core::edit_glbuffer(context, write_queue[it]->m_vbo, buffer, src, dst - src, fan::opengl::GL_UNIFORM_BUFFER);
    }

    write_queue[it]->on_edit(context);

    it = write_queue.next(it);
  }

  write_queue.clear();
}

void fan::vulkan::shader_t::open(fan::vulkan::context_t* context) {
  projection_view_block.open(context);
  projection_view_block.push_ram_instance(context, {});
}

void fan::vulkan::shader_t::close(fan::vulkan::context_t* context, fan::vulkan::core::uniform_write_queue_t* write_queue) {
  vkDestroyShaderModule(context->device, shaderStages[0].module, nullptr);
  vkDestroyShaderModule(context->device, shaderStages[1].module, nullptr);
  projection_view_block.close(context, write_queue);
}

VkShaderModule fan::vulkan::shader_t::createShaderModule(fan::vulkan::context_t* context, const fan::string& code) {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(context->device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }

  return shaderModule;
}