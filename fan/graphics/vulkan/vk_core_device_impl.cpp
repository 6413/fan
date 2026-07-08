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
#ifndef GLFW_INCLUDE_NONE
  #define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

module fan.graphics.vulkan.core;

import std;

import fan.time;

import fan.types.fstring;
import fan.types.color;

#if defined(loco_window)
import fan.window;
#endif

import fan.utility;
import fan.print;
import fan.print.error;
import fan.graphics.image_load;
import fan.graphics.common_context;

import fan.math;
import fan.math.intersection;

#define __fan_internal_camera_list (*fan::graphics::ctx().camera_list)
#define __fan_internal_shader_list (*fan::graphics::ctx().shader_list)
#define __fan_internal_image_list (*fan::graphics::ctx().image_list)
#define __fan_internal_viewport_list (*fan::graphics::ctx().viewport_list)

#define ENABLE_RAYTRACING_DEPENDENCIES

#define VK_CTX ((fan::vulkan::context_t*)context)

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
void fan::vulkan::context_t::open_no_window() {
  fan::time::timer t;
  if (fan::time::is_measuring()) {
    fan::time::measure(t, "open_no_window start");
  }
  fan::time::measure(t, "create_instance");
  setup_debug_messenger();
  fan::time::measure(t, "setup_debug_messenger");
  pick_physical_device();
  fan::time::measure(t, "pick_physical_device");
  create_logical_device();
  fan::time::measure(t, "create_logical_device");
  create_allocator();
  fan::time::measure(t, "create_allocator");
  create_command_pool();
  fan::time::measure(t, "create_command_pool");
  create_command_buffers();
  fan::time::measure(t, "create_command_buffers");
  create_sync_objects();
  if (fan::time::is_measuring()) {
    fan::time::print_measure("create_sync_objects took:", t.millis(), "ms");
  }
}
#if defined(loco_window)

void fan::vulkan::context_t::open(fan::window_t& window) {
  if (fan::time::is_measuring()) {
    fan::time::print_measure("open start");
  }
  fan::time::timer t_total;
  fan::time::timer t;
  window_resize_handle = window.add_resize_callback([&](const fan::window_t::resize_data_t& d) {
    SwapChainRebuild = true;
  });

  fan::time::measure(t, "window callback");
  
  create_instance();
  fan::time::measure(t, "create_instance");

  setup_debug_messenger();
  fan::time::measure(t, "setup_debug_messenger");

  create_surface(window);
  fan::time::measure(t, "create_surface");

  pick_physical_device();
  fan::time::measure(t, "pick_physical_device");

  create_logical_device();
  fan::time::measure(t, "create_logical_device");

  create_allocator();
  fan::time::measure(t, "create_allocator");

  create_swap_chain(window.get_size());
  fan::time::measure(t, "create_swap_chain");

  create_command_pool();
  fan::time::measure(t, "create_command_pool");

#if !defined(FAN_GUI)
  create_image_views();
  fan::time::measure(t, "create_image_views");
#endif

  create_render_pass();
  fan::time::measure(t, "create_render_pass");

#if !defined(FAN_GUI)
  create_framebuffers();
  fan::time::measure(t, "create_framebuffers");
#endif

  create_command_buffers();
  fan::time::measure(t, "create_command_buffers");

  create_sync_objects();
  fan::time::measure(t, "create_sync_objects");

  descriptor_pool.open(*this);
  fan::time::measure(t, "descriptor_pool.open");

#if defined(FAN_GUI)
  ImGuiSetupVulkanWindow();
  fan::time::measure(t, "ImGuiSetupVulkanWindow");
#endif

  if (fan::time::is_measuring()) {
    fan::time::print_measure("open total took:", t_total.millis(), "ms");
  }
}

#endif
void fan::vulkan::context_t::close_vais(std::vector<fan::vulkan::vai_t>& v) {
  for (auto& e : v) {
    e.close(*this);
  }
}
void fan::vulkan::context_t::destroy_vulkan_soft() {
  vkDeviceWaitIdle(device);

  if (single_time_fence != VK_NULL_HANDLE) {
    vkWaitForFences(device, 1, &single_time_fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(device, single_time_fence, nullptr);
    single_time_fence = VK_NULL_HANDLE;
  }
  if (single_time_cmd != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(device, command_pool, 1, &single_time_cmd);
    single_time_cmd = VK_NULL_HANDLE;
  }

  close_vais(mainColorImageViews);
  close_vais(postProcessedColorImageViews);
  close_vais(depthImageViews);
  close_vais(downscaleImageViews1);
  close_vais(upscaleImageViews1);
  close_vais(vai_depth);

  for (std::size_t i = 0; i < image_available_semaphores.size(); i++) {
    vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
    vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
  }

  for (std::size_t i = 0; i < in_flight_fences.size(); i++) {
    vkDestroyFence(device, in_flight_fences[i], nullptr);
  }

  flush_deletion_queues();

  vkDestroyRenderPass(device, render_pass, nullptr);
  vkDestroyCommandPool(device, command_pool, nullptr);

#if FAN_DEBUG >= fan_debug_high
  if (supports_validation_layers) {
    DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
  }
#endif
}
void fan::vulkan::context_t::gui_close() {
  vkDeviceWaitIdle(device);

  vkFreeCommandBuffers(device, command_pool, command_buffers.size(), command_buffers.data());

  cleanup_swap_chain_dependencies();
  descriptor_pool.close(*this);

#if defined(FAN_GUI)
  ImGui_ImplVulkanH_DestroyWindow(instance, device, &MainWindowData, nullptr);
#endif

  destroy_shape_resources();
  destroy_vulkan_soft();
  destroy_allocator();
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}

void fan::vulkan::context_t::close() {
  vkDeviceWaitIdle(device);

  cleanup_swap_chain();
  vkDestroySurfaceKHR(instance, surface, nullptr);

  destroy_shape_resources();
  destroy_vulkan_soft();
  destroy_allocator();

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}
void fan::vulkan::context_t::destroy_shape_resources() {
  {
    fan::graphics::camera_list_t::nrtra_t nrtra;
    fan::graphics::camera_list_t::nr_t nr;
    nrtra.Open(&__fan_internal_camera_list, &nr);
    while (nrtra.Loop(&__fan_internal_camera_list, &nr)) {
      camera_erase(nr);
    }
    nrtra.Close(&__fan_internal_camera_list);
  }
  {
    fan::graphics::shader_list_t::nrtra_t nrtra;
    fan::graphics::shader_list_t::nr_t nr;
    nrtra.Open(&__fan_internal_shader_list, &nr);
    while (nrtra.Loop(&__fan_internal_shader_list, &nr)) {
      shader_erase(nr);
    }
    nrtra.Close(&__fan_internal_shader_list);
  }
  {
    fan::graphics::image_list_t::nrtra_t nrtra;
    fan::graphics::image_list_t::nr_t nr;
    nrtra.Open(&__fan_internal_image_list, &nr);
    while (nrtra.Loop(&__fan_internal_image_list, &nr)) {
      image_erase(nr);
    }
    nrtra.Close(&__fan_internal_image_list);
  }
  {
    fan::graphics::viewport_list_t::nrtra_t nrtra;
    fan::graphics::viewport_list_t::nr_t nr;
    nrtra.Open(&__fan_internal_viewport_list, &nr);
    while (nrtra.Loop(&__fan_internal_viewport_list, &nr)) {
      viewport_erase(nr);
    }
    nrtra.Close(&__fan_internal_viewport_list);
  }
}
void fan::vulkan::context_t::cleanup_swap_chain_dependencies() {
  vkDeviceWaitIdle(device);
  close_vais(mainColorImageViews);
  close_vais(postProcessedColorImageViews);
  close_vais(depthImageViews);
  close_vais(downscaleImageViews1);
  close_vais(upscaleImageViews1);
  close_vais(vai_depth);

  for (auto framebuffer : swap_chain_framebuffers) {
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  }
  for (auto& i : swap_chain_image_views) {
    vkDestroyImageView(device, i, nullptr);
  }
}
void fan::vulkan::context_t::cleanup_swap_chain() {
  cleanup_swap_chain_dependencies();
  if (swap_chain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device, swap_chain, nullptr);
    swap_chain = VK_NULL_HANDLE;
  }
}
void fan::vulkan::context_t::recreate_swap_chain_dependencies() {
  create_image_views();
  create_framebuffers();
}
// if swapchain changes, reque
void fan::vulkan::context_t::update_swapchain_dependencies() {
  std::uint32_t imageCount =
  #if defined(FAN_GUI)
    MinImageCount + 1
  #else 
    min_image_count + 1
  #endif
  ;
  vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, nullptr);
  swap_chain_images.resize(imageCount);

  for (auto* view : {&mainColorImageViews, &postProcessedColorImageViews, &depthImageViews, &downscaleImageViews1, &upscaleImageViews1, &vai_depth}) {
    if (imageCount < view->size()) {
      for (std::size_t i = imageCount; i < view->size(); ++i) {
        (*view)[i].close(*this);
      }
    }
    view->resize(imageCount);
  }

  vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, swap_chain_images.data());
  recreate_swap_chain_dependencies();
}
void fan::vulkan::context_t::recreate_swap_chain(fan::window_t* window, VkResult err) {
  if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR || SwapChainRebuild) {
    int fb_width, fb_height;
    glfwGetFramebufferSize(*window, &fb_width, &fb_height);
    if (fb_width > 0 && fb_height > 0 &&
    #if defined(FAN_GUI)
      (
      #endif
        SwapChainRebuild
      #if defined(FAN_GUI)
        || MainWindowData.Width != fb_width ||
        MainWindowData.Height != fb_height)
    #endif
      ) {

      vkDeviceWaitIdle(device);
      swap_chain_support_details_t swapChainSupport = query_swap_chain_support(physical_device);
      present_mode = choose_swap_present_mode(swapChainSupport.present_modes);

    #if defined(FAN_GUI)
      MainWindowData.PresentMode = present_mode;
      MinImageCount = std::max<std::uint32_t>(
        2,
        (std::uint32_t)ImGui_ImplVulkanH_GetMinImageCountFromPresentMode(present_mode)
      );
      ImGui_ImplVulkan_SetMinImageCount(MinImageCount);
      ImGui_ImplVulkanH_CreateOrResizeWindow(instance, physical_device, device, &MainWindowData, queue_family, /*g_Allocator*/nullptr, fb_width, fb_height, MinImageCount);
      current_frame = MainWindowData.FrameIndex = 0;
    #else
      cleanup_swap_chain();
      create_swap_chain(fan::vec2ui((std::uint32_t)fb_width, (std::uint32_t)fb_height));
    #endif
      SwapChainRebuild = false;
    #if defined(FAN_GUI)
      swap_chain = MainWindowData.Swapchain;
    #endif
      swap_chain_size = fan::vec2(fb_width, fb_height);
      update_swapchain_dependencies();
    }
  }
  else if (err != VK_SUCCESS) {
    fan::throw_error("failed to present swap chain image");
  }
}
//void fan::vulkan::context_t::recreate_swap_chain(const fan::vec2i& window_size) {
      //  vkDeviceWaitIdle(device);
      //  cleanup_swap_chain();
      //  create_swap_chain(window_size);
      //  recreate_swap_chain_dependencies();
      //  // need to recreate some imgui's swapchain dependencies
      //#if defined(FAN_GUI)
      //  MainWindowData.Swapchain = swap_chain;
      //#endif
      //}
void fan::vulkan::context_t::create_instance() {
#if defined(fan_platform_windows)
  SetEnvironmentVariableA("DISABLE_VULKAN_OBS_CAPTURE", "1");
  SetEnvironmentVariableA("DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1", "1");
#endif


#if FAN_DEBUG >= fan_debug_high
  if (!check_validation_layer_support()) {
    fan::print_log(fan::log_level_e::warning, "VULKAN - validation layer:", "validation layers not supported");
    supports_validation_layers = false;
  }
#endif

  VkApplicationInfo appInfo {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "application";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 2, 0);
  appInfo.pEngineName = "fan";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 2, 0);
  appInfo.apiVersion = VK_API_VERSION_1_2;


  VkInstanceCreateInfo createInfo {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  auto extensions = get_required_extensions();
  createInfo.enabledExtensionCount = extensions.size();
  std::vector<const char*> extension_names(extensions.size());

  for (std::uint32_t i = 0; i < extensions.size(); ++i) {
    extension_names[i] = extensions[i].c_str();
  }
  createInfo.ppEnabledExtensionNames = extension_names.data();

  VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo {};
#if FAN_DEBUG >= fan_debug_high
  if (supports_validation_layers) {
    createInfo.enabledLayerCount = validationLayers.size();
    createInfo.ppEnabledLayerNames = validationLayers.data();

    populate_debug_messenger_create_info(debugCreateInfo);
    createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
  }

#endif

  if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
    fan::throw_error("failed to create instance!");
  }
}
void fan::vulkan::context_t::populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& create_info) {
  create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  create_info.pfnUserCallback = debug_callback;
}
void fan::vulkan::context_t::setup_debug_messenger() {
#if FAN_DEBUG < fan_debug_high
  return;
#endif

  if (!supports_validation_layers) {
    return;
  }

  VkDebugUtilsMessengerCreateInfoEXT createInfo;
  populate_debug_messenger_create_info(createInfo);

  if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debug_messenger) != VK_SUCCESS) {
    fan::throw_error("failed to set up debug messenger!");
  }
}
#if defined(loco_window)
void fan::vulkan::context_t::create_surface(GLFWwindow* window) {
  if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
    fan::throw_error("failed to create window surface!");
  }
}

#endif
void fan::vulkan::context_t::pick_physical_device() {
  std::uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

  if (deviceCount == 0) {
    fan::throw_error("failed to find GPUs with Vulkan support!");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  for (const auto& device : devices) {
    if (is_device_suitable(device)) {
      physical_device = device;
      break;
    }
  }

  if (physical_device == VK_NULL_HANDLE) {
    fan::throw_error("failed to find a suitable GPU!");
  }
}
void fan::vulkan::context_t::create_logical_device() {
  queue_family_indices_t indices = find_queue_families(physical_device);

  // -----------------------------
  // Queue creation
  // -----------------------------
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<std::uint32_t> uniqueQueueFamilies = {
    indices.graphics_family.value(),
  #if defined(loco_window)
    indices.present_family.value()
  #endif
  };

  f32_t queuePriority = 1.0f;
  for (std::uint32_t queueFamily : uniqueQueueFamilies) {
    VkDeviceQueueCreateInfo queueCreateInfo {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }

  // -----------------------------
  // Base features
  // -----------------------------
  VkPhysicalDeviceFeatures deviceFeatures {};
  deviceFeatures.samplerAnisotropy = VK_TRUE;

  // -----------------------------
  // Query device properties
  // -----------------------------
  VkPhysicalDeviceProperties2 deviceProperties {};
  deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  vkGetPhysicalDeviceProperties2(physical_device, &deviceProperties);

  // -----------------------------
  // Check extension support
  // -----------------------------
  if (!check_device_extension_support(physical_device)) {
    fan::throw_error("Required Vulkan device extensions missing.");
  }

  // Explicit RT extension check
  bool rt_ok = true;
  {
    std::uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extCount, exts.data());

    auto hasExt = [&](const char* name) {
      for (auto& e : exts) {
        if (strcmp(e.extensionName, name) == 0) return true;
      }
      return false;
    };

  #if defined(ENABLE_RAYTRACING_DEPENDENCIES)
    if (!hasExt(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) ||
      !hasExt(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) ||
      !hasExt(VK_KHR_SPIRV_1_4_EXTENSION_NAME) ||
      !hasExt(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) ||
      !hasExt(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) ||
      !hasExt(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME)) {
      rt_ok = false;
    }
  #endif
  }

#if defined(ENABLE_RAYTRACING_DEPENDENCIES)
  if (!rt_ok) {
    fan::throw_error("Ray tracing not supported on this GPU.");
  }
#endif

  // -----------------------------
  // Build feature chain
  // -----------------------------
  VkPhysicalDeviceFeatures2 features2 {};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.features = deviceFeatures;

  VkPhysicalDeviceVulkan12Features vulkan12 {};
  vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  vulkan12.runtimeDescriptorArray = VK_TRUE;
  vulkan12.descriptorIndexing = VK_TRUE;
  vulkan12.descriptorBindingVariableDescriptorCount = VK_TRUE;
  vulkan12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
  vulkan12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
  vulkan12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
  vulkan12.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
  vulkan12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
  vulkan12.descriptorBindingPartiallyBound = VK_TRUE;

#if defined(ENABLE_RAYTRACING_DEPENDENCIES)
  vulkan12.bufferDeviceAddress = VK_TRUE;

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel {};
  accel.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
  accel.accelerationStructure = VK_TRUE;
  accel.descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE;

  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt {};
  rt.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
  rt.rayTracingPipeline = VK_TRUE;

  // Chain: features2 → vulkan12 → accel → rt
  features2.pNext = &vulkan12;
  vulkan12.pNext = &accel;
  accel.pNext = &rt;
#else
  features2.pNext = &vulkan12;
  vulkan12.pNext = nullptr;
#endif

  // -----------------------------
  // Device create info
  // -----------------------------
  VkDeviceCreateInfo createInfo {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.queueCreateInfoCount = (std::uint32_t)queueCreateInfos.size();
  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  createInfo.pNext = &features2;
  createInfo.pEnabledFeatures = nullptr;

  createInfo.enabledExtensionCount = (std::uint32_t)deviceExtensions.size();
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();

  // -----------------------------
  // Create device
  // -----------------------------
  VkResult r = vkCreateDevice(physical_device, &createInfo, nullptr, &device);
  if (r != VK_SUCCESS) {
    fan::print_error("vkCreateDevice failed with code:", (int)r);
    fan::throw_error("failed to create logical device");
  }

  // -----------------------------
  // Get queues
  // -----------------------------
  vkGetDeviceQueue(device, indices.graphics_family.value(), 0, &graphics_queue);
#if defined(loco_window)
  vkGetDeviceQueue(device, indices.present_family.value(), 0, &present_queue);
#endif
}
#if defined(loco_window)
void fan::vulkan::context_t::create_swap_chain(const fan::vec2ui& framebuffer_size) {
  swap_chain_support_details_t swapChainSupport = query_swap_chain_support(physical_device);

  surface_format = choose_swap_surface_format(swapChainSupport.formats);
  present_mode = choose_swap_present_mode(swapChainSupport.present_modes);
  VkExtent2D extent = choose_swap_extent(framebuffer_size, swapChainSupport.capabilities);

  std::uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
  min_image_count = swapChainSupport.capabilities.minImageCount;
  image_count = imageCount;
  if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR createInfo {};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = surface;

  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surface_format.format;
  createInfo.imageColorSpace = surface_format.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

  queue_family_indices_t indices = find_queue_families(physical_device);
  queue_family = indices.graphics_family.value();
  std::uint32_t queueFamilyIndices[] = {indices.graphics_family.value(), indices.present_family.value()};

  if (indices.graphics_family != indices.present_family) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  }
  else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = present_mode;
  createInfo.clipped = VK_TRUE;
  //createInfo.imageUsage = ;

  if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swap_chain) != VK_SUCCESS) {
    fan::throw_error("failed to create swap chain!");
  }

  vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, nullptr);
  swap_chain_images.resize(imageCount);
  vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, swap_chain_images.data());

  swap_chain_image_format = surface_format.format;
  swap_chain_size = fan::vec2(extent.width, extent.height);
}

#endif
VkImageView fan::vulkan::context_t::create_image_view(VkImage image, VkFormat format, VkImageAspectFlags aspect_flags) {
  VkImageViewCreateInfo viewInfo {};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = aspect_flags;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;


  VkImageView imageView;
  if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
    fan::throw_error("failed to create texture image view!");
  }

  return imageView;
}
void fan::vulkan::context_t::create_image_views() {
  for (auto& view : swap_chain_image_views) {
    if (view != VK_NULL_HANDLE) {
      vkDestroyImageView(device, view, nullptr);
      view = VK_NULL_HANDLE;
    }
  }
  swap_chain_image_views.resize(swap_chain_images.size());

  fan::vulkan::vai_t::properties_t vp{
    .swap_chain_size = swap_chain_size,
    .format = main_color_format,
    .usage_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    .aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT
  };
  fan::vulkan::vai_t::properties_t depth_vp = vp;
  depth_vp.aspect_flags = VK_IMAGE_ASPECT_DEPTH_BIT;
  depth_vp.format = find_depth_format();
  depth_vp.usage_flags = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

  for (auto* view : {&mainColorImageViews, &postProcessedColorImageViews, &depthImageViews, &downscaleImageViews1, &upscaleImageViews1}) {
    if (swap_chain_images.size() < view->size()) {
      for (std::size_t i = swap_chain_images.size(); i < view->size(); ++i) {
        (*view)[i].close(*this);
      }
    }
    view->resize(swap_chain_images.size());
  }

  for (std::size_t i = 0; i < swap_chain_images.size(); ++i) {
    mainColorImageViews[i].close(*this);
    mainColorImageViews[i].open(*this, vp);
    mainColorImageViews[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

    postProcessedColorImageViews[i].close(*this);
    postProcessedColorImageViews[i].open(*this, vp);
    postProcessedColorImageViews[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

    depthImageViews[i].close(*this);
    depthImageViews[i].open(*this, depth_vp);
    depthImageViews[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, depth_vp.aspect_flags);

    downscaleImageViews1[i].close(*this);
    downscaleImageViews1[i].open(*this, vp);
    downscaleImageViews1[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

    upscaleImageViews1[i].close(*this);
    upscaleImageViews1[i].open(*this, vp);
    upscaleImageViews1[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

    swap_chain_image_views[i] = create_image_view(swap_chain_images[i], swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT);
  }
}
void fan::vulkan::context_t::create_render_pass() {
  //--------------attachment description--------------

  VkAttachmentDescription mainColorAttachment {};
  mainColorAttachment.format = main_color_format;
  mainColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  mainColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  mainColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  mainColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  mainColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  mainColorAttachment.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  mainColorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkAttachmentDescription depthAttachment {};
  depthAttachment.format = find_depth_format();
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  //--------------attachment description--------------

  VkAttachmentReference mainSceneColorRef {};
  mainSceneColorRef.attachment = 0;
  mainSceneColorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthRef {};
  depthRef.attachment = 1;
  depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription mainSceneSubpass {};
  mainSceneSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  mainSceneSubpass.colorAttachmentCount = 1;
  mainSceneSubpass.pColorAttachments = &mainSceneColorRef;
  mainSceneSubpass.pDepthStencilAttachment = &depthRef;

  VkSubpassDependency extToMainDep {};
  extToMainDep.srcSubpass = VK_SUBPASS_EXTERNAL;
  extToMainDep.dstSubpass = 0;
  extToMainDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  extToMainDep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  extToMainDep.srcAccessMask = 0;
  extToMainDep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  extToMainDep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  VkSubpassDependency mainToExtDep {};
  mainToExtDep.srcSubpass = 0;
  mainToExtDep.dstSubpass = VK_SUBPASS_EXTERNAL;
  mainToExtDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  mainToExtDep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  mainToExtDep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  mainToExtDep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  mainToExtDep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  VkAttachmentDescription attachments[] = {
    mainColorAttachment,
    depthAttachment
  };

  VkSubpassDescription subpasses[] = {
    mainSceneSubpass
  };

  VkSubpassDependency dependencies[] = {
    extToMainDep,
    mainToExtDep
  };

  VkRenderPassCreateInfo renderPassInfo {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = std::size(attachments);
  renderPassInfo.pAttachments = attachments;
  renderPassInfo.subpassCount = std::size(subpasses);
  renderPassInfo.pSubpasses = subpasses;
  renderPassInfo.dependencyCount = std::size(dependencies);
  renderPassInfo.pDependencies = dependencies;

  if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &render_pass) != VK_SUCCESS) {
    fan::throw_error("failed to create render pass");
  }
}
void fan::vulkan::context_t::create_framebuffers() {
  for (auto& fb : swap_chain_framebuffers) {
    if (fb != VK_NULL_HANDLE) {
      vkDestroyFramebuffer(device, fb, nullptr);
      fb = VK_NULL_HANDLE;
    }
  }
  swap_chain_framebuffers.resize(swap_chain_image_views.size());

  for (std::size_t i = 0; i < swap_chain_image_views.size(); i++) {
    VkImageView attachments[] = {
      mainColorImageViews[i].image_view,
      depthImageViews[i].image_view,
    };

    VkFramebufferCreateInfo framebufferInfo {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = render_pass;
    framebufferInfo.attachmentCount = std::size(attachments);
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = swap_chain_size.x;
    framebufferInfo.height = swap_chain_size.y;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swap_chain_framebuffers[i]) != VK_SUCCESS) {
      fan::throw_error("failed to create framebuffer!");
    }
  }
}
void fan::vulkan::context_t::create_command_pool() {
  queue_family_indices_t queueFamilyIndices = find_queue_families(physical_device);

  VkCommandPoolCreateInfo poolInfo {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = queueFamilyIndices.graphics_family.value();

  if (vkCreateCommandPool(device, &poolInfo, nullptr, &command_pool) != VK_SUCCESS) {
    fan::throw_error("failed to create graphics command pool!");
  }
}
VkFormat fan::vulkan::context_t::find_supported_format(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

    if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
      return format;
    }
    else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }

  fan::throw_error("failed to find supported format!");
  return {};
}
VkFormat fan::vulkan::context_t::find_depth_format() {
  return find_supported_format(
    {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
    VK_IMAGE_TILING_OPTIMAL,
    VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
  );
}
bool fan::vulkan::context_t::has_stencil_component(VkFormat format) {
  return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}
void fan::vulkan::context_t::create_allocator() {
  VmaAllocatorCreateInfo allocator_info{};
  allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  allocator_info.physicalDevice = physical_device;
  allocator_info.device = device;
  allocator_info.instance = instance;
  allocator_info.vulkanApiVersion = VK_API_VERSION_1_2;
  fan::vulkan::validate(vmaCreateAllocator(&allocator_info, &allocator));

  staging_ring_buffer.init(device, allocator, 32 * 1024 * 1024);
}
void fan::vulkan::context_t::destroy_allocator() {
  if (allocator != VK_NULL_HANDLE) {
    //char* stats_string = nullptr;
    //vmaBuildStatsString(allocator, &stats_string, VK_TRUE);
    //fan::print(stats_string);
    //vmaFreeStatsString(allocator, stats_string);
    staging_ring_buffer.destroy();
    vmaDestroyAllocator(allocator);
    allocator = VK_NULL_HANDLE;
  }
}

void fan::vulkan::context_t::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VmaAllocation& allocation, VmaAllocationInfo* allocation_info) {
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocation_create_info{};
  allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
  allocation_create_info.requiredFlags = properties;
  if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    allocation_create_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
      ((properties & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ?
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT :
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
  }

  fan::vulkan::validate(vmaCreateBuffer(allocator, &buffer_info, &allocation_create_info, &buffer, &allocation, allocation_info));
}

void fan::vulkan::context_t::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, buffer_t& buffer, VmaAllocationInfo* allocation_info) {
  create_buffer(size, usage, properties, buffer.buffer, buffer.allocation, allocation_info);
  buffer.size = size;
}
void fan::vulkan::context_t::destroy_buffer(VkBuffer& buffer, VmaAllocation& allocation) {
  if (buffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(allocator, buffer, allocation);
    buffer = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
  }
}
void fan::vulkan::context_t::destroy_buffer(buffer_t& buffer) {
  unmap_buffer(buffer);
  destroy_buffer(buffer.buffer, buffer.allocation);
  buffer.size = 0;
}
VkResult fan::vulkan::context_t::map_buffer(buffer_t& buffer, void** data) {
  if (buffer.mapped == nullptr) {
    VkResult result = vmaMapMemory(allocator, buffer.allocation, &buffer.mapped);
    if (result != VK_SUCCESS) {
      return result;
    }
  }
  *data = buffer.mapped;
  return VK_SUCCESS;
}
void fan::vulkan::context_t::unmap_buffer(buffer_t& buffer) {
  if (buffer.mapped != nullptr) {
    vmaUnmapMemory(allocator, buffer.allocation);
    buffer.mapped = nullptr;
  }
}
void fan::vulkan::context_t::invalidate_buffer(buffer_t& buffer, VkDeviceSize offset, VkDeviceSize size) {
  fan::vulkan::validate(vmaInvalidateAllocation(allocator, buffer.allocation, offset, size));
}
VkCommandBuffer fan::vulkan::context_t::begin_single_time_commands() {
  if (single_time_cmd == VK_NULL_HANDLE) {
    VkCommandBufferAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = command_pool;
    allocInfo.commandBufferCount = 1;
    fan::vulkan::validate(vkAllocateCommandBuffers(device, &allocInfo, &single_time_cmd));

    VkFenceCreateInfo fence_info {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fan::vulkan::validate(vkCreateFence(device, &fence_info, nullptr, &single_time_fence));
  }
  else {
    fan::vulkan::validate(vkWaitForFences(device, 1, &single_time_fence, VK_TRUE, UINT64_MAX));
    fan::vulkan::validate(vkResetFences(device, 1, &single_time_fence));
    fan::vulkan::validate(vkResetCommandBuffer(single_time_cmd, 0));
  }

  VkCommandBufferBeginInfo beginInfo {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  fan::vulkan::validate(vkBeginCommandBuffer(single_time_cmd, &beginInfo));

  return single_time_cmd;
}
void fan::vulkan::context_t::end_single_time_commands(VkCommandBuffer command_buffer) {
  fan::vulkan::validate(vkEndCommandBuffer(command_buffer));

  VkSubmitInfo submitInfo {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffer;

  fan::vulkan::validate(vkQueueSubmit(graphics_queue, 1, &submitInfo, single_time_fence));
  fan::vulkan::validate(vkWaitForFences(device, 1, &single_time_fence, VK_TRUE, UINT64_MAX));
}

VkCommandBuffer fan::vulkan::context_t::begin_async_transfer_commands() {
  VkCommandBufferAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = command_pool;
  allocInfo.commandBufferCount = 1;
  
  VkCommandBuffer cmd;
  fan::vulkan::validate(vkAllocateCommandBuffers(device, &allocInfo, &cmd));

  VkCommandBufferBeginInfo beginInfo {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  fan::vulkan::validate(vkBeginCommandBuffer(cmd, &beginInfo));

  return cmd;
}

void fan::vulkan::context_t::end_async_transfer_commands(VkCommandBuffer command_buffer) {
  fan::vulkan::validate(vkEndCommandBuffer(command_buffer));

  VkSubmitInfo submitInfo {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffer;

  fan::vulkan::validate(vkQueueSubmit(graphics_queue, 1, &submitInfo, VK_NULL_HANDLE));

  VkDevice device_handle = device;
  VkCommandPool pool_handle = command_pool;

  get_current_deletion_queue().push_function([=]() {
    vkFreeCommandBuffers(device_handle, pool_handle, 1, &command_buffer);
  });
}

void fan::vulkan::context_t::copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size) {
  VkCommandBuffer cmd = begin_single_time_commands();
  VkBufferCopy copyRegion {};
  copyRegion.size = size;
  vkCmdCopyBuffer(cmd, src_buffer, dst_buffer, 1, &copyRegion);
  end_single_time_commands(cmd);
}
void fan::vulkan::context_t::copy_buffer_cmd(VkCommandBuffer cmd, VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize src_offset, VkDeviceSize dst_offset, VkDeviceSize size) {
  VkBufferCopy r { .srcOffset = src_offset, .dstOffset = dst_offset, .size = size };
  vkCmdCopyBuffer(cmd, src_buffer, dst_buffer, 1, &r);
}

void fan::vulkan::context_t::upload_to_buffer(VkBuffer dest_buffer, std::span<const std::byte> data, VkDeviceSize dst_offset) {
  if (data.empty()) return;

  auto alloc = staging_ring_buffer.allocate(data.size());
  
  std::memcpy(alloc.mapped_ptr, data.data(), data.size());

  VkCommandBuffer cmd = begin_async_transfer_commands();
  
  VkBufferCopy copy_region{
      .srcOffset = alloc.offset,
      .dstOffset = dst_offset,
      .size = data.size()
  };
  
  vkCmdCopyBuffer(cmd, alloc.buffer, dest_buffer, 1, &copy_region);

  end_async_transfer_commands(cmd);

  VkDeviceSize captured_head = staging_ring_buffer.get_head();
  auto* ring_ptr = &staging_ring_buffer;
  auto allocator_handle = allocator;

  get_current_deletion_queue().push_function([=]() {
      if (alloc.is_spilled) {
          vmaDestroyBuffer(allocator_handle, alloc.buffer, alloc.fallback_allocation);
      } else {
          ring_ptr->advance_tail(captured_head);
      }
  });
}

void fan::vulkan::context_t::fill_buffer_cmd(VkCommandBuffer cmd, buffer_t& buffer, VkDeviceSize offset, VkDeviceSize size, std::uint32_t data) {
  vkCmdFillBuffer(cmd, buffer.buffer, offset, size, data);
}
void fan::vulkan::context_t::buffer_barrier_cmd(VkCommandBuffer cmd, buffer_t& buffer, VkAccessFlags src_access, VkAccessFlags dst_access, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage, VkDeviceSize offset, VkDeviceSize size) {
  buffer_barriers_cmd(cmd, {{&buffer, src_access, dst_access, offset, size}}, src_stage, dst_stage);
}
void fan::vulkan::context_t::buffer_barriers_cmd(VkCommandBuffer cmd, const std::vector<buffer_barrier_t>& barriers, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
  std::vector<VkBufferMemoryBarrier> vk_barriers;
  vk_barriers.reserve(barriers.size());
  for (auto& src : barriers) {
    if (src.buffer == nullptr || src.buffer->buffer == VK_NULL_HANDLE) { continue; }
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = src.src_access;
    barrier.dstAccessMask = src.dst_access;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = src.buffer->buffer;
    barrier.offset = src.offset;
    barrier.size = src.size;
    vk_barriers.push_back(barrier);
  }
  if (!vk_barriers.empty()) {
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, (std::uint32_t)vk_barriers.size(), vk_barriers.data(), 0, nullptr);
  }
}
void fan::vulkan::context_t::compute_pipeline_t::open(fan::vulkan::context_t& context, const std::string& path, VkDeviceSize push_size_, const std::vector<binding_t>& bindings) {
  push_size = push_size_;
  std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
  layout_bindings.reserve(bindings.size());
  for (auto& binding : bindings) {
    layout_bindings.push_back({binding.binding, binding.type, binding.descriptor_count, binding.stage_flags, nullptr});
  }
  std::vector<VkDescriptorBindingFlags> binding_flags(layout_bindings.size());
  for (auto& flags : binding_flags) {
    flags =
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
      VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
  }

  VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{};
  binding_flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
  binding_flags_info.bindingCount = (std::uint32_t)binding_flags.size();
  binding_flags_info.pBindingFlags = binding_flags.data();

  VkDescriptorSetLayoutCreateInfo descriptor_info{};
  descriptor_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptor_info.pNext = &binding_flags_info;
  descriptor_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
  descriptor_info.bindingCount = (std::uint32_t)layout_bindings.size();
  descriptor_info.pBindings = layout_bindings.data();
  fan::vulkan::validate(vkCreateDescriptorSetLayout(context.device, &descriptor_info, nullptr, &descriptor_layout));

  auto shader_code = fan::graphics::read_shader(path);
  auto spirv = fan::vulkan::context_t::compile_file(path, shaderc_compute_shader, shader_code);
  VkShaderModule shader = context.create_shader_module(spirv);

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = (std::uint32_t)push_size;

  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &descriptor_layout;
  layout_info.pushConstantRangeCount = push_size == 0 ? 0 : 1;
  layout_info.pPushConstantRanges = push_size == 0 ? nullptr : &push_range;
  fan::vulkan::validate(vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout));

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_info.stage.module = shader;
  pipeline_info.stage.pName = "main";
  pipeline_info.layout = pipeline_layout;
  fan::vulkan::validate(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline));
  vkDestroyShaderModule(context.device, shader, nullptr);
}
void fan::vulkan::context_t::compute_pipeline_t::close(fan::vulkan::context_t& context) {
  if (pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(context.device, pipeline, nullptr); }
  if (pipeline_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr); }
  if (descriptor_layout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(context.device, descriptor_layout, nullptr); }
  pipeline = VK_NULL_HANDLE;
  pipeline_layout = VK_NULL_HANDLE;
  descriptor_layout = VK_NULL_HANDLE;
  push_size = 0;
}
void fan::vulkan::context_t::compute_pipeline_t::dispatch(fan::vulkan::context_t& context, VkCommandBuffer cmd, VkDescriptorSet descriptor_set, const void* push, std::uint32_t x, std::uint32_t y, std::uint32_t z) const {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
  if (push_size != 0 && push != nullptr) {
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (std::uint32_t)push_size, push);
  }
  vkCmdDispatch(cmd, x, y, z);
}
void fan::vulkan::context_t::compute_slot_ring_t::open(fan::vulkan::context_t& context, std::uint32_t slot_count, VkDescriptorSetLayout descriptor_layout, const std::vector<buffer_properties_t>& buffer_properties_) {
  buffer_properties = buffer_properties_;
  slots.resize(slot_count);

  std::vector<VkDescriptorPoolSize> pool_sizes;
  for (auto& bp : buffer_properties) {
    auto it = std::find_if(pool_sizes.begin(), pool_sizes.end(), [&](const auto& p) { return p.type == bp.descriptor_type; });
    if (it == pool_sizes.end()) { pool_sizes.push_back({bp.descriptor_type, slot_count}); }
    else { it->descriptorCount += slot_count; }
  }
  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
  pool_info.maxSets = slot_count;
  pool_info.poolSizeCount = (std::uint32_t)pool_sizes.size();
  pool_info.pPoolSizes = pool_sizes.data();
  fan::vulkan::validate(vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool));

  std::vector<VkDescriptorSetLayout> layouts(slot_count, descriptor_layout);
  std::vector<VkDescriptorSet> sets(slot_count);
  VkDescriptorSetAllocateInfo descriptor_alloc{};
  descriptor_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptor_alloc.descriptorPool = descriptor_pool;
  descriptor_alloc.descriptorSetCount = slot_count;
  descriptor_alloc.pSetLayouts = layouts.data();
  fan::vulkan::validate(vkAllocateDescriptorSets(context.device, &descriptor_alloc, sets.data()));

  VkCommandBufferAllocateInfo cmd_alloc{};
  cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc.commandPool = context.command_pool;
  cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc.commandBufferCount = slot_count;
  std::vector<VkCommandBuffer> commands(slot_count);
  fan::vulkan::validate(vkAllocateCommandBuffers(context.device, &cmd_alloc, commands.data()));

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  for (std::uint32_t i = 0; i < slot_count; ++i) {
    auto& slot = slots[i];
    slot.command_buffer = commands[i];
    slot.descriptor_set = sets[i];
    fan::vulkan::validate(vkCreateFence(context.device, &fence_info, nullptr, &slot.fence));
    slot.buffers.resize(buffer_properties.size());
    for (std::uint32_t j = 0; j < buffer_properties.size(); ++j) {
      auto& bp = buffer_properties[j];
      context.create_buffer(bp.size, bp.usage, bp.memory, slot.buffers[j]);
      if (bp.map) { fan::vulkan::validate(context.map_buffer(slot.buffers[j], &slot.buffers[j].mapped)); }
    }
    std::vector<VkDescriptorBufferInfo> infos(buffer_properties.size());
    std::vector<VkWriteDescriptorSet> writes(buffer_properties.size());
    for (std::uint32_t j = 0; j < buffer_properties.size(); ++j) {
      infos[j] = {slot.buffers[j].buffer, 0, VK_WHOLE_SIZE};
      writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[j].dstSet = slot.descriptor_set;
      writes[j].dstBinding = j;
      writes[j].descriptorCount = 1;
      writes[j].descriptorType = buffer_properties[j].descriptor_type;
      writes[j].pBufferInfo = &infos[j];
    }
    vkUpdateDescriptorSets(context.device, (std::uint32_t)writes.size(), writes.data(), 0, nullptr);
  }
}
void fan::vulkan::context_t::compute_slot_ring_t::close(fan::vulkan::context_t& context) {
  std::vector<VkCommandBuffer> commands;
  for (auto& slot : slots) {
    if (slot.in_flight) { fan::vulkan::validate(vkWaitForFences(context.device, 1, &slot.fence, VK_TRUE, UINT64_MAX)); }
    for (auto& buffer : slot.buffers) { context.destroy_buffer(buffer); }
    if (slot.fence != VK_NULL_HANDLE) { vkDestroyFence(context.device, slot.fence, nullptr); }
    if (slot.command_buffer != VK_NULL_HANDLE) { commands.push_back(slot.command_buffer); }
  }
  if (!commands.empty()) { vkFreeCommandBuffers(context.device, context.command_pool, (std::uint32_t)commands.size(), commands.data()); }
  if (descriptor_pool != VK_NULL_HANDLE) { vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr); }
  slots.clear();
  buffer_properties.clear();
  descriptor_pool = VK_NULL_HANDLE;
  submit_slot = 0;
}
std::uint32_t fan::vulkan::context_t::compute_slot_ring_t::acquire() const {
  if (slots.empty()) { return invalid_slot; }
  for (std::uint32_t i = 0; i < slots.size(); ++i) {
    std::uint32_t index = (submit_slot + i) % (std::uint32_t)slots.size();
    if (!slots[index].in_flight) { return index; }
  }
  return invalid_slot;
}
VkCommandBuffer fan::vulkan::context_t::compute_slot_ring_t::begin(fan::vulkan::context_t& context, std::uint32_t slot_index) {
  auto& slot = slots[slot_index];
  fan::vulkan::validate(vkResetFences(context.device, 1, &slot.fence));
  fan::vulkan::validate(vkResetCommandBuffer(slot.command_buffer, 0));
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  fan::vulkan::validate(vkBeginCommandBuffer(slot.command_buffer, &begin_info));
  return slot.command_buffer;
}
void fan::vulkan::context_t::compute_slot_ring_t::submit(fan::vulkan::context_t& context, std::uint32_t slot_index) {
  auto& slot = slots[slot_index];
  fan::vulkan::validate(vkEndCommandBuffer(slot.command_buffer));
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &slot.command_buffer;
  fan::vulkan::validate(vkQueueSubmit(context.graphics_queue, 1, &submit_info, slot.fence));
  slot.in_flight = true;
  submit_slot = (slot_index + 1) % (std::uint32_t)slots.size();
}
bool fan::vulkan::context_t::compute_slot_ring_t::done(fan::vulkan::context_t& context, std::uint32_t slot_index) const {
  auto& slot = slots[slot_index];
  if (!slot.in_flight) { return false; }
  VkResult r = vkGetFenceStatus(context.device, slot.fence);
  if (r == VK_NOT_READY) { return false; }
  fan::vulkan::validate(r);
  return true;
}
void fan::vulkan::context_t::compute_slot_ring_t::set_idle(std::uint32_t slot_index) {
  slots[slot_index].in_flight = false;
}
std::uint32_t fan::vulkan::context_t::compute_slot_ring_t::free_slot_count() const {
  std::uint32_t count = 0;
  for (auto& slot : slots) { count += !slot.in_flight; }
  return count;
}
fan::vulkan::context_t::compute_slot_ring_t::slot_t& fan::vulkan::context_t::compute_slot_ring_t::get(std::uint32_t slot_index) {
  return slots[slot_index];
}
const fan::vulkan::context_t::compute_slot_ring_t::slot_t& fan::vulkan::context_t::compute_slot_ring_t::get(std::uint32_t slot_index) const {
  return slots[slot_index];
}
void fan::vulkan::context_t::create_command_buffers() {
  command_buffers.resize(max_frames_in_flight);

  VkCommandBufferAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = command_pool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = (std::uint32_t)command_buffers.size();

  if (vkAllocateCommandBuffers(device, &allocInfo, command_buffers.data()) != VK_SUCCESS) {
    fan::throw_error("failed to allocate command buffers!");
  }
}
void fan::vulkan::context_t::bind_draw(
  const fan::vulkan::context_t::pipeline_t& pipeline,
  std::uint32_t descriptor_count,
  VkDescriptorSet* descriptor_sets) {
  vkCmdBindPipeline(command_buffers[current_frame], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.m_pipeline);

  VkRect2D scissor {};
  scissor.offset = {0, 0};
  scissor.extent.width = swap_chain_size.x;
  scissor.extent.height = swap_chain_size.y;
  vkCmdSetScissor(command_buffers[current_frame], 0, 1, &scissor);

  vkCmdBindDescriptorSets(
    command_buffers[current_frame],
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
void fan::vulkan::context_t::bindless_draw(
  std::uint32_t vertex_count,
  std::uint32_t instance_count,
  std::uint32_t first_instance) {
  vkCmdDraw(command_buffers[current_frame], vertex_count, instance_count, 0, first_instance);
}
void fan::vulkan::context_t::draw(
  std::uint32_t vertex_count,
  std::uint32_t instance_count,
  std::uint32_t first_instance,
  const fan::vulkan::context_t::pipeline_t& pipeline,
  std::uint32_t descriptor_count,
  VkDescriptorSet* descriptor_sets
) {
  bind_draw(pipeline, descriptor_count, descriptor_sets);
  bindless_draw(vertex_count, instance_count, first_instance);
}
void fan::vulkan::context_t::create_sync_objects() {
  std::uint32_t acquire_count = std::max((std::uint32_t)swap_chain_images.size(), (std::uint32_t)3);
  image_available_semaphores.resize(acquire_count);
  render_finished_semaphores.resize(acquire_count);
  in_flight_fences.resize(max_frames_in_flight);

  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (std::size_t i = 0; i < acquire_count; i++) {
    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &image_available_semaphores[i]) != VK_SUCCESS ||
      vkCreateSemaphore(device, &semaphoreInfo, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS) {
      fan::throw_error("failed to create semaphore!");
    }
  }
  for (std::size_t i = 0; i < max_frames_in_flight; i++) {
    if (vkCreateFence(device, &fenceInfo, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
      fan::throw_error("failed to create fence!");
    }
  }
}
#if defined(FAN_GUI)
void fan::vulkan::context_t::ImGuiSetupVulkanWindow() {
  MainWindowData.Surface = surface;
  MainWindowData.SurfaceFormat = surface_format;
  MainWindowData.Swapchain = swap_chain;
  MainWindowData.PresentMode = present_mode;
  MainWindowData.ClearEnable = shapes_top;

  IM_ASSERT(MinImageCount >= 2);
  ImGui_ImplVulkanH_CreateOrResizeWindow(instance, physical_device, device, &MainWindowData, queue_family, /*g_Allocator*/nullptr, swap_chain_size.x, swap_chain_size.y, MinImageCount);
  swap_chain = MainWindowData.Swapchain;
  update_swapchain_dependencies();
}

#endif
#if defined(FAN_GUI)
void fan::vulkan::context_t::ImGuiFrameRender(void* ctx, VkResult next_image_khr_err, fan::color clear_color) {
  fan::vulkan::context_t& context = *(fan::vulkan::context_t*)ctx;
  ImGui_ImplVulkanH_Window* wd = &context.MainWindowData;
  VkResult err = next_image_khr_err;
  if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
    context.SwapChainRebuild = true;
  if (err == VK_ERROR_OUT_OF_DATE_KHR)
    return;
  if (err != VK_SUBOPTIMAL_KHR)
    fan::vulkan::validate(err);

  wd->FrameIndex = context.image_index;

  ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];

  VkRenderPassBeginInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  info.renderPass = wd->RenderPass;
  info.framebuffer = fd->Framebuffer;
  info.renderArea.extent.width = wd->Width;
  info.renderArea.extent.height = wd->Height;

  vkCmdBeginRenderPass(context.command_buffers[context.current_frame], &info, VK_SUBPASS_CONTENTS_INLINE);
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), context.command_buffers[context.current_frame]);

  vkCmdEndRenderPass(context.command_buffers[context.current_frame]);
}

#endif
VkResult fan::vulkan::context_t::end_render(fan::window_t* window) {
  //// render_fullscreen_pl loco fbo?
  if (!command_buffer_in_use) {
    return VK_SUCCESS;
  }

  if (vkEndCommandBuffer(command_buffers[current_frame]) != VK_SUCCESS) {
    fan::throw_error("failed to record command buffer!");
  }

  command_buffer_in_use = false;

  VkSubmitInfo submitInfo {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore waitSemaphores[] = {current_acquire_semaphore};
  VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffers[current_frame];

  VkSemaphore signalSemaphores[] = {render_finished_semaphores[image_index]};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  VkResult submit_result = vkQueueSubmit(graphics_queue, 1, &submitInfo, in_flight_fences[current_frame]);
  if (submit_result != VK_SUCCESS) {
    fan::print_error("vkQueueSubmit error:", (int)submit_result);
    fan::throw_error("failed to submit draw command buffer!");
  }

  VkPresentInfoKHR presentInfo {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = signalSemaphores;

  VkSwapchainKHR swapChains[] = {swap_chain};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;
  presentInfo.pImageIndices = &image_index;
  auto result = vkQueuePresentKHR(present_queue, &presentInfo);

  if (!window_shown) {
    window_shown = true;
    vkQueueWaitIdle(present_queue);
    window->show();
  }

  current_frame = (current_frame + 1) % max_frames_in_flight;
  return result;
}
VkSurfaceFormatKHR fan::vulkan::context_t::choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats) {
  for (const auto& availableFormat : available_formats) {
    // VK_FORMAT_B8G8R8A8_SRGB

    if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return availableFormat;
    }
  }

  return available_formats[0];
}
VkPresentModeKHR fan::vulkan::context_t::choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes) {
  if (vsync) {
    for (const auto& available_present_mode : available_present_modes) {
      if (available_present_mode == VK_PRESENT_MODE_FIFO_KHR) {
        return VK_PRESENT_MODE_FIFO_KHR;
      }
    }
  }
  else {
    for (const auto& preferred_mode : {
      VK_PRESENT_MODE_IMMEDIATE_KHR,
      VK_PRESENT_MODE_MAILBOX_KHR,
      VK_PRESENT_MODE_FIFO_RELAXED_KHR
    }) {
      for (const auto& available_present_mode : available_present_modes) {
        if (available_present_mode == preferred_mode) {
          return preferred_mode;
        }
      }
    }
  }

  return VK_PRESENT_MODE_FIFO_KHR;
}
VkExtent2D fan::vulkan::context_t::choose_swap_extent(const fan::vec2ui& framebuffer_size, const VkSurfaceCapabilitiesKHR& capabilities) {
  if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
    return capabilities.currentExtent;
  }
  else {
    VkExtent2D actualExtent = {
      framebuffer_size.x,
      framebuffer_size.y
    };

    actualExtent.width = fan::math::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = fan::math::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return actualExtent;
  }
}
swap_chain_support_details_t fan::vulkan::context_t::query_swap_chain_support(VkPhysicalDevice device) {
  swap_chain_support_details_t details;

  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

  std::uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

  if (formatCount != 0) {
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
  }

  std::uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

  if (presentModeCount != 0) {
    details.present_modes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.present_modes.data());
  }

  return details;
}
bool fan::vulkan::context_t::is_device_suitable(VkPhysicalDevice device) {
  queue_family_indices_t indices = find_queue_families(device);

  bool extensionsSupported = check_device_extension_support(device);

  bool swapChainAdequate
  #if defined(loco_window)
    = false;
  if (extensionsSupported) {
    swap_chain_support_details_t swapChainSupport = query_swap_chain_support(device);
    swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.present_modes.empty();
  }
#else
    = true;
#endif

  VkPhysicalDeviceFeatures supportedFeatures;
  vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

  return indices.is_complete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}
bool fan::vulkan::context_t::check_device_extension_support(VkPhysicalDevice device) {
  std::uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

  std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

  for (const auto& extension : availableExtensions) {
    requiredExtensions.erase(extension.extensionName);
  }

  return requiredExtensions.empty();
}
queue_family_indices_t fan::vulkan::context_t::find_queue_families(VkPhysicalDevice device) {
  queue_family_indices_t indices;

  std::uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

  int i = 0;
  for (const auto& queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphics_family = i;
    }

    VkBool32 presentSupport = false;

  #if defined(loco_window)
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

    if (presentSupport) {
      indices.present_family = i;
    }
  #endif

    if (indices.is_complete()) {
      break;
    }

    i++;
  }

  return indices;
}
std::vector<std::string> fan::vulkan::context_t::get_required_extensions() {
  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  std::vector<std::string> extension_str;
  if (glfwExtensions) {
    for (uint32_t i = 0; i < glfwExtensionCount; i++) {
      extension_str.push_back(glfwExtensions[i]);
    }
  }

#if FAN_DEBUG >= fan_debug_high
  if (supports_validation_layers) {
    extension_str.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
#endif

#if 0
  fan::print_dbg_tag("VULKAN", "Requested Vulkan Instance Extensions:");
  for (const auto& ext : extension_str) {
    fan::print_dbg_tag("VULKAN", "- ", ext);
  }
#endif

  return extension_str;
}
bool fan::vulkan::context_t::check_validation_layer_support() {
  std::uint32_t layerCount;
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
      fan::print_log(fan::log_level_e::warning, "VULKAN - validation layer:", "Missing Vulkan validation layer:", layerName);
      std::string msg = "Available Vulkan instance layers:\n";
      for (const auto& layerProperties : availableLayers) {
        msg += std::string(fan::tab) + layerProperties.layerName + '\n';
      }
      msg.pop_back();
      fan::print_log(fan::log_level_e::warning, "VULKAN - validation layer:", msg);
      return false;
    }
  }

  return true;
}
VKAPI_ATTR VkBool32 VKAPI_CALL fan::vulkan::context_t::debug_callback(
  VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
  VkDebugUtilsMessageTypeFlagsEXT messageType,
  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
  void* pUserData
) {
  if (pCallbackData->pMessageIdName && std::string_view(pCallbackData->pMessageIdName) == "Loader Message") {
    return VK_FALSE;
  }

  std::string_view msg = pCallbackData->pMessage ? pCallbackData->pMessage : "";

  if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) &&
    msg.find("has an Output value declared at Location") != std::string_view::npos &&
    msg.find("but there is no corresponding Input declared") != std::string_view::npos) {
    return VK_FALSE;
  }

  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    fan::print_log(fan::log_level_e::error, "VULKAN - Validation layer:", pCallbackData->pMessage);
  }
  else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    fan::print_log(fan::log_level_e::warning, "VULKAN - Validation layer:", pCallbackData->pMessage);
  }
  else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    fan::print_log(fan::log_level_e::info, "VULKAN - Validation layer:", pCallbackData->pMessage);
  }
  else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
    fan::print_log(fan::log_level_e::info, "VULKAN - Validation layer:", pCallbackData->pMessage);
  }
  return VK_FALSE;
}
#if defined(loco_window)
void fan::vulkan::context_t::set_vsync(fan::window_t* window, bool flag) {
  if (vsync == flag && !SwapChainRebuild) {
    return;
  }
  vsync = flag;
  SwapChainRebuild = true;
}

#endif
fan::graphics::context_functions_t fan::graphics::get_vk_context_functions() {
  fan::graphics::context_functions_t cf;
  cf.shader_create = [](void* context) {
    return VK_CTX->shader_create();
  };
  cf.shader_get = [](void* context, fan::graphics::shader_nr_t nr) {
    return (void*)&VK_CTX->shader_get(nr);
  };
  cf.shader_erase = [](void* context, fan::graphics::shader_nr_t nr) {
    VK_CTX->shader_erase(nr);
  };
  cf.shader_use = [](void* context, fan::graphics::shader_nr_t nr) {
    VK_CTX->shader_use(nr);
  };
  cf.shader_set_vertex = [](void* context, fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code) {
    VK_CTX->shader_set_vertex(nr, file_path, vertex_code);
  };
  cf.shader_set_fragment = [](void* context, fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code) {
    VK_CTX->shader_set_fragment(nr, file_path, fragment_code);
  };
  cf.shader_set_compute = [](void* context, fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& compute_code) {
    VK_CTX->shader_set_compute(nr, file_path, compute_code);
  };
  cf.shader_dispatch_compute = [](void* context, fan::graphics::shader_nr_t nr, uint32_t x, uint32_t y, uint32_t z) {
    VK_CTX->shader_dispatch_compute(nr, x, y, z);
  };
  cf.shader_compile = [](void* context, fan::graphics::shader_nr_t nr) {
    return VK_CTX->shader_compile(nr);
  };
  /*image*/
  cf.image_create = [](void* context) {
    return VK_CTX->image_create();
  };
  cf.image_get_handle = [](void* context, fan::graphics::image_nr_t nr) {
    return (std::uint64_t)VK_CTX->image_get_handle(nr);
  };
  cf.image_get = [](void* context, fan::graphics::image_nr_t nr) {
    return (void*)&VK_CTX->image_get(nr);
  };
  cf.image_erase = [](void* context, fan::graphics::image_nr_t nr) {
    VK_CTX->image_erase(nr);
  };
  cf.image_bind = [](void* context, fan::graphics::image_nr_t nr) {
    VK_CTX->image_bind(nr);
  };
  cf.image_bind_unit = [](void* context, fan::graphics::image_nr_t nr, std::uint32_t unit) {
    VK_CTX->image_bind(nr, unit);
  };
  cf.image_bind_params = [](void* context, fan::graphics::image_nr_t nr, std::uint32_t unit, std::uint32_t access, std::uint32_t format) {
    VK_CTX->image_bind(nr, unit, access, format);
  };
  cf.image_unbind = [](void* context, fan::graphics::image_nr_t nr) {
    VK_CTX->image_unbind(nr);
  };
  cf.image_get_settings = [](void* context, fan::graphics::image_nr_t nr) -> fan::graphics::image_load_properties_t& {
    return VK_CTX->image_get_settings(nr);
  };
  cf.image_set_settings = [](void* context, fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings) {
    VK_CTX->image_set_settings(nr, fan::graphics::format_converter::image_global_to_vulkan(settings));
  };
  cf.image_load_info = [](void* context, const fan::image::info_t& image_info) {
    return VK_CTX->image_load(image_info);
  };
  cf.image_load_info_props = [](void* context, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
    fan::image::info_t info = image_info;
    if (info.channels <= 0) {
      info.channels = fan::graphics::get_channel_amount(p.format);
    }
    return VK_CTX->image_load(info, fan::graphics::format_converter::image_global_to_vulkan(p));
  };
  cf.request_image_load_async = [](void* context, fan::str_view_t path, const fan::graphics::image_load_properties_t& p, std::function<void(const fan::graphics::decoded_image_payload_t&)> on_gpu_uploaded) {
    return VK_CTX->request_image_load_async(path, fan::graphics::format_converter::image_global_to_vulkan(p), on_gpu_uploaded);
  };
  cf.process_async_image_uploads = [](void* context) {
    VK_CTX->process_async_image_uploads();
  };
  cf.image_load_path = [](void* context, fan::str_view_t path, const std::source_location& callers_path = std::source_location::current()) {
    return VK_CTX->image_load(path, callers_path);
  };
  cf.image_load_path_props = [](void* context, fan::str_view_t path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
    return VK_CTX->image_load(path, fan::graphics::format_converter::image_global_to_vulkan(p), callers_path);
  };
  cf.image_load_colors = [](void* context, fan::color* colors, const fan::vec2ui& size_) {
    return VK_CTX->image_load(colors, size_);
  };
  cf.image_load_colors_props = [](void* context, fan::color* colors, const fan::vec2ui& size_, const fan::graphics::image_load_properties_t& p) {
    return VK_CTX->image_load(colors, size_, fan::graphics::format_converter::image_global_to_vulkan(p));
  };
  cf.image_unload = [](void* context, fan::graphics::image_nr_t nr) {
    VK_CTX->image_unload(nr);
  };
  cf.create_missing_texture = [](void* context) {
    return VK_CTX->create_missing_texture();
  };
  cf.create_transparent_texture = [](void* context) {
    return VK_CTX->create_transparent_texture();
  };
  cf.image_reload_image_info = [](void* context, fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) {
    image_load_properties_t lp;
    if (nr) lp = image_get_settings(nr);
    return VK_CTX->image_reload(nr, image_info, fan::graphics::format_converter::image_global_to_vulkan(lp));
  };
  cf.image_reload_image_info_props = [](void* context, fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
    fan::image::info_t info = image_info;
    if (info.channels <= 0) {
      info.channels = fan::graphics::get_channel_amount(p.format);
    }
    return VK_CTX->image_reload(nr, info, fan::graphics::format_converter::image_global_to_vulkan(p));
  };
  cf.image_reload_path = [](void* context, fan::graphics::image_nr_t nr, fan::str_view_t path, const std::source_location& callers_path = std::source_location::current()) {
    return VK_CTX->image_reload(nr, path, callers_path);
  };
  cf.image_reload_path_props = [](void* context, fan::graphics::image_nr_t nr, fan::str_view_t path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
    return VK_CTX->image_reload(nr, path, fan::graphics::format_converter::image_global_to_vulkan(p), callers_path);
  };
  cf.image_create_color = [](void* context, const fan::color& color) {
    return VK_CTX->image_create(color);
  };
  cf.image_create_color_props = [](void* context, const fan::color& color, const fan::graphics::image_load_properties_t& p) {
    return VK_CTX->image_create(color, fan::graphics::format_converter::image_global_to_vulkan(p));
  };
  cf.image_create_data = [](void* context, void* data, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
    fan::image::info_t info;
    info.data = data;
    info.size = size;
    info.channels = fan::graphics::get_channel_amount(p.format);
    return VK_CTX->image_load(info, fan::graphics::format_converter::image_global_to_vulkan(p));
  };
  /*camera*/
  cf.camera_create = [](void* context) {
    return VK_CTX->camera_create();
  };
  cf.camera_get = [](void* context, fan::graphics::camera_nr_t nr) -> decltype(auto) {
    return VK_CTX->camera_get(nr);
  };
  cf.camera_erase = [](void* context, fan::graphics::camera_nr_t nr) {
    VK_CTX->camera_erase(nr);
  };
  cf.camera_create_params = [](void* context, const fan::vec2& x, const fan::vec2& y) {
    return VK_CTX->camera_create(x, y);
  };
  cf.camera_get_position = [](void* context, fan::graphics::camera_nr_t nr) {
    return VK_CTX->camera_get_position(nr);
  };
  cf.camera_set_position = [](void* context, fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
    VK_CTX->camera_set_position(nr, cp);
  };
  cf.camera_get_center = [](void* context, fan::graphics::camera_nr_t nr) {
    return VK_CTX->camera_get_center(nr);
  };
  cf.camera_set_center = [](void* context, fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
    VK_CTX->camera_set_center(nr, cp);
  };
  cf.camera_get_size = [](void* context, fan::graphics::camera_nr_t nr) {
    return VK_CTX->camera_get_size(nr);
  };
  cf.camera_get_zoom = [](void* context, fan::graphics::camera_nr_t nr) {
    return VK_CTX->camera_get_zoom(nr);
  };
  cf.camera_set_zoom = [](void* context, fan::graphics::camera_nr_t nr, f32_t new_zoom) {
    VK_CTX->camera_set_zoom(nr, new_zoom);
  };
  cf.camera_set_ortho = [](void* context, fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
    VK_CTX->camera_set_ortho(nr, x, y);
  };
  cf.camera_set_perspective = [](void* context, fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
    VK_CTX->camera_set_perspective(nr, fov, window_size);
  };
  cf.camera_rotate = [](void* context, fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
    VK_CTX->camera_rotate(nr, offset);
  };
  /*viewport*/
  cf.viewport_create = [](void* context) {
    return VK_CTX->viewport_create();
  };
  cf.viewport_create_params = [](void* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
    auto vk_context = VK_CTX;
    auto nr = vk_context->viewport_create();
    vk_context->viewport_set(nr, viewport_position_, viewport_size_, window_size);
    return nr;
  };
  cf.viewport_get = [](void* context, fan::graphics::viewport_nr_t nr) -> fan::graphics::context_viewport_t& {
    return VK_CTX->viewport_get(nr);
  };
  cf.viewport_erase = [](void* context, fan::graphics::viewport_nr_t nr) {
    VK_CTX->viewport_erase(nr);
  };
  cf.viewport_get_position = [](void* context, fan::graphics::viewport_nr_t nr) {
    return VK_CTX->viewport_get_position(nr);
  };
  cf.viewport_get_size = [](void* context, fan::graphics::viewport_nr_t nr) {
    return VK_CTX->viewport_get_size(nr);
  };
  cf.viewport_set = [](void* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
    VK_CTX->viewport_set(viewport_position_, viewport_size_, window_size);
  };
  cf.viewport_set_nr = [](void* context, fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
    VK_CTX->viewport_set(nr, viewport_position_, viewport_size_, window_size);
  };
  cf.viewport_zero = [](void* context, fan::graphics::viewport_nr_t nr) {
    VK_CTX->viewport_zero(nr);
  };
  cf.viewport_inside = [](void* context, fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
    return VK_CTX->viewport_inside(nr, position);
  };
  cf.viewport_inside_wir = [](void* context, fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
    return VK_CTX->viewport_inside_wir(nr, position);
  };
  cf.image_get_pixel_data = [](void* context, fan::graphics::image_nr_t nr, std::uint32_t format, fan::vec2 uvp, fan::vec2 uvs) {
    return VK_CTX->image_get_pixel_data(nr, format, uvp, uvs);
  };
  cf.image_read_pixels = [](void* context, fan::graphics::image_nr_t nr, fan::vec2 uv_pos, fan::vec2 uv_size) {
    auto& vk = *(fan::vulkan::context_t*)context;
    auto img_settings = vk.image_get_settings(nr);
    return vk.image_get_pixel_data(nr, img_settings.format, uv_pos, uv_size);
  };
  return cf;
}
namespace fan::graphics {
  fan::vulkan::context_t& get_vk_context() {
    return (*static_cast<fan::vulkan::context_t*>(static_cast<void*>(fan::graphics::ctx())));
  }
}