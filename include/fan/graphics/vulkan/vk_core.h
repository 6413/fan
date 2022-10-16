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

#include "vk_pipeline.h"

namespace fan {
  namespace vulkan {

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

#include "viewport_list_builder_settings.h"
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#include _FAN_PATH(BLL/BLL.h)

namespace fan {

  namespace vulkan {

    namespace core {
      struct uniform_block_common_t;
    }

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

      static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

      static constexpr const char* validationLayers[] = {
        "VK_LAYER_KHRONOS_validation"
      };

      static constexpr const char* deviceExtensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
      };

      void open() {
        createInstance();
        setupDebugMessenger();
      }
      void close() {
        cleanupSwapChain();

        // erase pipeline here
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
          vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
          vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
          vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        #if fan_debug >= fan_debug_high
          DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        #endif

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
      }

      void bind_to_window(fan::window_t* window) {
        createSurface(window->get_handle());
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain(window->get_size());
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();

        createFramebuffers();
        createCommandPool();
        createCommandBuffer();
        createSyncObjects();

        window->add_resize_callback([this, window](const fan::window_t::resize_cb_data_t& d) {
          recreateSwapChain(window->get_size());
        });
      }

      struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
      };

      std::vector<fan::string> getRequiredExtensions() {

        uint32_t extensions_count = 0;
        VkResult result = VK_SUCCESS;

        result = vkEnumerateInstanceExtensionProperties(nullptr, &extensions_count, nullptr);
        if ((result != VK_SUCCESS) ||
          (extensions_count == 0)) {
          throw std::runtime_error("Could not get the number of Instance extensions.");
        }

        std::vector<VkExtensionProperties> available_extensions;

        available_extensions.resize(extensions_count);

        result = vkEnumerateInstanceExtensionProperties(nullptr, &extensions_count, &available_extensions[0]);

        if ((result != VK_SUCCESS) ||
          (extensions_count == 0)) {
          throw std::runtime_error("Could not enumerate Instance extensions.");
        }

        std::vector<fan::string> extension_str(available_extensions.size());

        for (int i = 0; i < available_extensions.size(); i++) {
          extension_str[i] = available_extensions[i].extensionName;
        }

        #if fan_debug >= fan_debug_high
          extension_str.push_back((char*)VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
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

      static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
      }

      void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
      }

      void createInstance() {
        #if fan_debug >= fan_debug_high
          if (!checkValidationLayerSupport()) {
            fan::throw_error("validation layers requested, but not available!");
          }
        #endif
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "test";
        appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0); // VK_MAKE_VERSION
        appInfo.pEngineName = "fan";
        appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        std::vector<char*> extension_names(extensions.size() + 1);
        for (uint32_t i = 0; i < extensions.size(); ++i) {
          extension_names[i] = new char[extensions[i].size() + 1];
          memcpy(extension_names[i], extensions[i].data(), extensions[i].size() + 1);
        }
        createInfo.ppEnabledExtensionNames = extension_names.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        #if fan_debug >= fan_debug_high
          createInfo.enabledLayerCount = std::size(validationLayers);
          createInfo.ppEnabledLayerNames = validationLayers;

          populateDebugMessengerCreateInfo(debugCreateInfo);
          createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        #else
          createInfo.enabledLayerCount = 0;

          createInfo.pNext = nullptr;
        #endif

        auto ret = vkCreateInstance(&createInfo, nullptr, &instance);

        if (ret != VK_SUCCESS) {
          throw std::runtime_error("failed to create instance!");
        }
      }

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

      void setupDebugMessenger() {
        #if fan_debug >= fan_debug_high 
          VkDebugUtilsMessengerCreateInfoEXT createInfo;
          populateDebugMessengerCreateInfo(createInfo);

          if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
          }
        #endif
      }

      void createSurface(void* handle) {

        #ifdef fan_platform_windows

        VkWin32SurfaceCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        create_info.hwnd = (HWND)handle;

        create_info.hinstance = GetModuleHandle(nullptr);

        if (vkCreateWin32SurfaceKHR(instance, &create_info, nullptr, &surface) != VK_SUCCESS) {
          throw std::runtime_error("failed to create window surface");
        }

        #elif defined(fan_platform_unix)

        VkXlibSurfaceCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        create_info.window = (Window)handle;
        create_info.dpy = fan::sys::get_display();

        int x = 0;

        if ((x = vkCreateXlibSurfaceKHR(instance, &create_info, nullptr, &surface)) != VK_SUCCESS) {
          fan::print("error:", x);
          throw std::runtime_error("failed to create window surface");
        }

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

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = std::size(deviceExtensions);
        createInfo.ppEnabledExtensionNames = deviceExtensions;

        #if fan_debug >= fan_debug_high
          createInfo.enabledLayerCount = std::size(validationLayers);
          createInfo.ppEnabledLayerNames = validationLayers;
        #else
          createInfo.enabledLayerCount = 0;
        #endif

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
          throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
      }

      struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() {
          return graphicsFamily.has_value() && presentFamily.has_value();
        }
      };

      bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        return indices.isComplete();
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

      void createSwapChain(const fan::vec2i& framebuffer_size) {
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

        createInfo.oldSwapchain = VK_NULL_HANDLE;

        auto ret = vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain);

        if (ret != VK_SUCCESS) {
          throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
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

      void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
          VkImageViewCreateInfo createInfo{};
          createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
          createInfo.image = swapChainImages[i];
          createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
          createInfo.format = swapChainImageFormat;
          createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
          createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
          createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
          createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
          createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
          createInfo.subresourceRange.baseMipLevel = 0;
          createInfo.subresourceRange.levelCount = 1;
          createInfo.subresourceRange.baseArrayLayer = 0;
          createInfo.subresourceRange.layerCount = 1;

          if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
          }
        }
      }

      void createGraphicsPipeline() {

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

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
          throw std::runtime_error("failed to create render pass!");
        }
      }

      void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
          VkImageView attachments[] = {
            swapChainImageViews[i]
          };

          VkFramebufferCreateInfo framebufferInfo{};
          framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
          framebufferInfo.renderPass = renderPass;
          framebufferInfo.attachmentCount = 1;
          framebufferInfo.pAttachments = attachments;
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
          throw std::runtime_error("failed to create command pool!");
        }
      }

      void createCommandBuffer() {
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

      void recordCommandBuffer(const VkPipeline& pipeline, VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
          throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
          throw std::runtime_error("failed to record command buffer!");
        }
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

      void render(fan::window_t* window, const fan::function_t<void()>& l) {
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
          recreateSwapChain(window->get_size());
          return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
          throw std::runtime_error("failed to acquire swap chain image!");
        }

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);

        l();
        // DRAW SHAPE HERE WITH CORRECT PIPELINE

        //recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

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

        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
          recreateSwapChain(window->get_size());
        }
        else if (result != VK_SUCCESS) {
          throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
      }

      void recreateSwapChain(const fan::vec2i& framebuffer_size) {
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain(framebuffer_size);
        createImageViews();
        createFramebuffers();
      }

      void cleanupSwapChain() {
        for (auto framebuffer : swapChainFramebuffers) {
          vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto imageView : swapChainImageViews) {
          vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
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

      VkRenderPass renderPass;
      fan::vulkan::pipelines_t pipelines;
      VkPipelineLayout pipelineLayout;

      std::vector<VkFramebuffer> swapChainFramebuffers;

      VkCommandPool commandPool;
      std::vector<VkCommandBuffer> commandBuffers;

      std::vector<VkSemaphore> imageAvailableSemaphores;
      std::vector<VkSemaphore> renderFinishedSemaphores;
      std::vector<VkFence> inFlightFences;
      uint32_t currentFrame = 0;

      bool vsync = true;

      fan::vulkan::viewport_list_t viewport_list;
      fan::vulkan::matrices_list_t matrices_list;
		};
	}
}

#include "vk_shader.h"
#include "uniform_block.h"

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
	auto node = pipeline_list[nr];

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 0;
	vertexInputInfo.vertexAttributeDescriptionCount = 0;

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
	rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

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
	pipelineLayoutInfo.setLayoutCount = 0;
	pipelineLayoutInfo.pushConstantRangeCount = 0;

	if (vkCreatePipelineLayout(context->device, &pipelineLayoutInfo, nullptr, &context->pipelineLayout) != VK_SUCCESS) {
		fan::throw_error("failed to create pipeline layout!");
	}

	VkGraphicsPipelineCreateInfo pipeline_infos;

	pipeline_infos.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeline_infos.stageCount = 2;
	pipeline_infos.pStages = p.shader->shaderStages;
	pipeline_infos.pVertexInputState = &vertexInputInfo;
	pipeline_infos.pInputAssemblyState = &inputAssembly;
	pipeline_infos.pViewportState = &viewportState;
	pipeline_infos.pRasterizationState = &rasterizer;
	pipeline_infos.pMultisampleState = &multisampling;
	pipeline_infos.pColorBlendState = &colorBlending;
	pipeline_infos.pDynamicState = &dynamicState;
	pipeline_infos.layout = context->pipelineLayout;
	pipeline_infos.renderPass = context->renderPass;
	pipeline_infos.subpass = 0;
	pipeline_infos.basePipelineHandle = VK_NULL_HANDLE;

	if (vkCreateGraphicsPipelines(
		context->device,
		VK_NULL_HANDLE,
		1,
		&pipeline_infos,
		nullptr,
		&node.pipeline
	) != VK_SUCCESS) {
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
  vkCmdSetViewport(context->commandBuffers[context->currentFrame], 0, 1, &viewport);
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