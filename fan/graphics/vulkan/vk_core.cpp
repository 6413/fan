#include "vk_core.h"

#include <regex>

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

bool queue_family_indices_t::is_complete() {
  return graphicsFamily.has_value()
#if defined(loco_window)
    && presentFamily.has_value()
#endif
    ;
}

void fan::vulkan::validate(VkResult result) {
  if (result != VK_SUCCESS) {
    fan::throw_error("function failed");
  }
}

void fan::vulkan::create_image(fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
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
  allocInfo.memoryTypeIndex = fan::vulkan::context_t::find_memory_type(context, memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(context.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate image memory!");
  }

  vkBindImageMemory(context.device, image, imageMemory, 0);
}

std::vector<uint32_t> fan::vulkan::context_t::compile_file(
  const fan::string& source_name,
  shaderc_shader_kind kind,
  const fan::string& source) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  // Like -DMY_DEFINE=1
  //options.AddMacroDefinition("MY_DEFINE", "1");
#if fan_debug > 1
  options.SetOptimizationLevel(shaderc_optimization_level_zero);
#else
  options.SetOptimizationLevel(shaderc_optimization_level_performance);
#endif

  shaderc::SpvCompilationResult module =
    compiler.CompileGlslToSpv(source.c_str(), kind, source_name.c_str(), options);

  if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
    fan::throw_error(module.GetErrorMessage().c_str());
  }

  return { module.cbegin(), module.cend() };
}

fan::vulkan::context_t::shader_nr_t fan::vulkan::context_t::shader_create() {
  shader_nr_t nr = shader_list.NewNode();
  auto& shader = shader_get(nr);
  //TODO
  shader.projection_view_block.open(*this);
  for (uint32_t i = 0; i < fan::vulkan::max_camera; ++i) {
    shader.projection_view_block.push_ram_instance(*this, {});
  }
  return nr;
}

fan::vulkan::context_t::shader_t& fan::vulkan::context_t::shader_get(shader_nr_t nr) {
  return shader_list[nr];
}

void fan::vulkan::context_t::shader_erase(shader_nr_t nr) {
  shader_t& shader = shader_get(nr);
  vkDestroyShaderModule(device, shader.shader_stages[0].module, nullptr);
  vkDestroyShaderModule(device, shader.shader_stages[1].module, nullptr);
  //TODO
  shader.projection_view_block.close(*this);
  shader_list.Recycle(nr);
}

void fan::vulkan::context_t::shader_use(shader_nr_t nr) {
  shader_t& shader = shader_get(nr);
}

VkShaderModule create_shader_module(fan::vulkan::context_t& context, const std::vector<uint32_t>& code) {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size() * sizeof(typename std::remove_reference_t<decltype(code)>::value_type);
  createInfo.pCode = code.data();

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(context.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    fan::throw_error("failed to create shader module!");
  }

  return shaderModule;
}

void fan::vulkan::context_t::shader_set_vertex(shader_nr_t nr, const fan::string& vertex_code) {
  auto& shader = shader_get(nr);
  shader.svertex = vertex_code;
  // fan::print(
  //   "processed vertex shader:", path, "resulted in:",
  // preprocess_shader(shader_name.c_str(), shaderc_glsl_vertex_shader, shader_code);
  // );
}

void fan::vulkan::context_t::shader_set_fragment(shader_nr_t nr, const fan::string& fragment_code) {
  shader_t& shader = shader_get(nr);
  shader.sfragment = fragment_code;
  //fan::print(
    // "processed vertex shader:", path, "resulted in:",
  //preprocess_shader(shader_name.c_str(), shaderc_glsl_fragment_shader, shader_code);
  //);
}

bool fan::vulkan::context_t::shader_compile(shader_nr_t nr) {
  shader_t& shader = shader_get(nr);
  {
    auto spirv = compile_file(/*vertex_code.c_str()*/ "some vertex file", shaderc_glsl_vertex_shader, shader.svertex);

    auto module_vertex = create_shader_module(*this, spirv);

    VkPipelineShaderStageCreateInfo vert{};
    vert.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert.module = module_vertex;
    vert.pName = "main";

    shader.shader_stages[0] = vert;
  }
  {
    auto spirv = compile_file(/*shader_name.c_str()*/"some fragment file", shaderc_glsl_fragment_shader, shader.sfragment);

    auto module_fragment = create_shader_module(*this, spirv);

    VkPipelineShaderStageCreateInfo frag{};
    frag.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag.module = module_fragment;
    frag.pName = "main";

    shader.shader_stages[1] = frag;
  }

  std::regex uniformRegex(R"(uniform\s+(\w+)\s+(\w+)(\s*=\s*[\d\.]+)?;)");

  fan::string vertexData = shader.svertex;

  std::smatch match;
  while (std::regex_search(vertexData, match, uniformRegex)) {
    shader.uniform_type_table[match[2]] = match[1];
    vertexData = match.suffix().str();
  }

  fan::string fragmentData = shader.sfragment;

  while (std::regex_search(fragmentData, match, uniformRegex)) {
    shader.uniform_type_table[match[2]] = match[1];
    fragmentData = match.suffix().str();
  }

  return 0;
}

void fan::vulkan::context_t::open_no_window() {
  create_instance();
  setup_debug_messenger();
  create_instance();
  setup_debug_messenger();
  pick_physical_device();
  create_logical_device();
  create_command_pool();
  create_command_buffers();
  create_sync_objects();
}

void fan::vulkan::context_t::open(fan::window_t& window) {
  window.add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
    SwapChainRebuild = true;
    recreate_swap_chain(d.window, VK_ERROR_OUT_OF_DATE_KHR);
    });

  create_instance();
  setup_debug_messenger();
  create_surface(window);
  pick_physical_device();
  create_logical_device();
  create_swap_chain(window.get_size());
  create_image_views();
  create_render_pass();
  create_command_pool();
#if defined(loco_wboit)
  create_wboit_views();
#endif
  create_loco_framebuffer();
  create_depth_resources();
  create_framebuffers();
  create_command_buffers();
  create_sync_objects();
  descriptor_pool.open(*this);
  ImGuiSetupVulkanWindow();
}

void fan::vulkan::context_t::destroy_vulkan_soft() {
  vkDeviceWaitIdle(device);
  for (auto& i : vai_bitmap) {
    i.close(*this);
  }

  for (size_t i = 0; i < max_frames_in_flight; i++) {
    if (render_finished_semaphores.size())
      vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
    if (image_available_semaphores.size())
      vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
    if (in_flight_fences.size())
      vkDestroyFence(device, in_flight_fences[i], nullptr);
  }

  vkDestroyRenderPass(device, render_pass, nullptr);
  vkDestroyCommandPool(device, command_pool, nullptr);

#if fan_debug >= fan_debug_high
  if (supports_validation_layers) {
    DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
  }
#endif
}

void fan::vulkan::context_t::imgui_close() {
  vkFreeCommandBuffers(device, command_pool, command_buffers.size(), command_buffers.data());
  cleanup_swap_chain_dependencies();
  descriptor_pool.close(*this);
  destroy_vulkan_soft();
  ImGui_ImplVulkanH_DestroyWindow(instance, device, &MainWindowData, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}

void fan::vulkan::context_t::close() {
  cleanup_swap_chain();
  vkDestroySurfaceKHR(instance, surface, nullptr);
  destroy_vulkan_soft();
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}

void fan::vulkan::context_t::cleanup_swap_chain_dependencies() {
  vai_depth.close(*this);

  for (auto framebuffer : swap_chain_framebuffers) {
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  }

  for (auto imageView : swap_chain_image_views) {
    vkDestroyImageView(device, imageView, nullptr);
  }
}

void fan::vulkan::context_t::recreate_swap_chain_dependencies() {
  create_image_views();
  create_depth_resources();
  create_framebuffers();
}

// if swapchain changes, reque

void fan::vulkan::context_t::update_swapchain_dependencies() {
  cleanup_swap_chain_dependencies();
  uint32_t imageCount = MinImageCount + 1;
  vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, nullptr);
  swap_chain_images.resize(imageCount);
  vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, swap_chain_images.data());
  recreate_swap_chain_dependencies();
}

void fan::vulkan::context_t::recreate_swap_chain(fan::window_t* window, VkResult err) {
  if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
    int fb_width, fb_height;
    glfwGetFramebufferSize(*window, &fb_width, &fb_height);
    if (fb_width > 0 && fb_height > 0 && (SwapChainRebuild || MainWindowData.Width != fb_width || MainWindowData.Height != fb_height))
    {
      ImGui_ImplVulkan_SetMinImageCount(MinImageCount);
      ImGui_ImplVulkanH_CreateOrResizeWindow(instance, physicalDevice, device, &MainWindowData, queue_family, /*g_Allocator*/nullptr, fb_width, fb_height, MinImageCount);
      current_frame = MainWindowData.FrameIndex = 0;
      SwapChainRebuild = false;
      swap_chain = MainWindowData.Swapchain;
      swap_chain_size = fan::vec2(fb_width, fb_height);
      update_swapchain_dependencies();
    }
  }
  else if (err != VK_SUCCESS) {
    throw std::runtime_error("failed to present swap chain image");
  }
}

void fan::vulkan::context_t::recreate_swap_chain(const fan::vec2i& window_size) {
  vkDeviceWaitIdle(device);
  cleanup_swap_chain();
  create_swap_chain(window_size);
  recreate_swap_chain_dependencies();
  // need to recreate some imgui's swapchain dependencies
  MainWindowData.Swapchain = swap_chain;
}

void fan::vulkan::context_t::create_instance() {
#if fan_debug >= fan_debug_high
  if (!check_validation_layer_support()) {
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

  auto extensions = get_required_extensions();
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

    populate_debug_messenger_create_info(debugCreateInfo);
    createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
  }
#endif

  if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
    throw std::runtime_error("failed to create instance!");
  }
}

void fan::vulkan::context_t::populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
  createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debug_callback;
}

void fan::vulkan::context_t::setup_debug_messenger() {
#if fan_debug < fan_debug_high
  return;
#endif

  if (!supports_validation_layers) {
    return;
  }

  VkDebugUtilsMessengerCreateInfoEXT createInfo;
  populate_debug_messenger_create_info(createInfo);

  if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debug_messenger) != VK_SUCCESS) {
    throw std::runtime_error("failed to set up debug messenger!");
  }
}

void fan::vulkan::context_t::create_surface(GLFWwindow* window) {
  if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface!");
  }
}

void fan::vulkan::context_t::pick_physical_device() {
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

  if (deviceCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  for (const auto& device : devices) {
    if (is_device_suitable(device)) {
      physicalDevice = device;
      break;
    }
  }

  if (physicalDevice == VK_NULL_HANDLE) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
}

void fan::vulkan::context_t::create_logical_device() {
  queue_family_indices_t indices = find_queue_families(physicalDevice);

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

  vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphics_queue);
#if defined(loco_window)
  vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &present_queue);
#endif
}

void fan::vulkan::context_t::create_swap_chain(const fan::vec2ui& framebuffer_size) {
  swap_chain_support_details_t swapChainSupport = query_swap_chain_support(physicalDevice);

  surface_format = choose_swap_surface_format(swapChainSupport.formats);
  present_mode = choose_swap_present_mode(swapChainSupport.present_modes);
  VkExtent2D extent = choose_swap_extent(framebuffer_size, swapChainSupport.capabilities);

  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
  min_image_count = swapChainSupport.capabilities.minImageCount;
  image_count = imageCount;
  if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = surface;

  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surface_format.format;
  createInfo.imageColorSpace = surface_format.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

  queue_family_indices_t indices = find_queue_families(physicalDevice);
  queue_family = indices.graphicsFamily.value();
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
  createInfo.presentMode = present_mode;
  createInfo.clipped = VK_TRUE;
  //createInfo.imageUsage = ;

  if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swap_chain) != VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain!");
  }

  vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, nullptr);
  swap_chain_images.resize(imageCount);
  vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, swap_chain_images.data());

  swap_chain_image_format = surface_format.format;
  swap_chain_size = fan::vec2(extent.width, extent.height);
}

VkImageView fan::vulkan::context_t::create_image_view(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
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

void fan::vulkan::context_t::create_image_views() {
  swap_chain_image_views.resize(swap_chain_images.size());

  for (uint32_t i = 0; i < swap_chain_images.size(); i++) {
    swap_chain_image_views[i] = create_image_view(swap_chain_images[i], swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT);
  }
}

void fan::vulkan::context_t::create_render_pass() {
  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = swap_chain_image_format;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentDescription depthAttachment{};
  depthAttachment.format = find_depth_format();
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

  fbo_color_attachment[0].format = swap_chain_image_format;
  fbo_color_attachment[0].samples = VK_SAMPLE_COUNT_1_BIT;
  fbo_color_attachment[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  fbo_color_attachment[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  fbo_color_attachment[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  fbo_color_attachment[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  fbo_color_attachment[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  fbo_color_attachment[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  fbo_color_attachment[1].format = swap_chain_image_format;
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

  if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &render_pass) != VK_SUCCESS) {
    throw std::runtime_error("failed to create renderpass");
  }
}

void fan::vulkan::context_t::create_framebuffers() {
  swap_chain_framebuffers.resize(swap_chain_image_views.size());

  for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
    VkImageView attachments[] = {
      vai_bitmap[0].image_view,
      vai_bitmap[1].image_view,
      swap_chain_image_views[i],
      vai_bitmap[1].image_view,
      vai_depth.image_view,
    };

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = render_pass;
    framebufferInfo.attachmentCount = std::size(attachments);
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = swap_chain_size.x;
    framebufferInfo.height = swap_chain_size.y;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swap_chain_framebuffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }
}

void fan::vulkan::context_t::create_command_pool() {
  queue_family_indices_t queueFamilyIndices = find_queue_families(physicalDevice);

  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

  if (vkCreateCommandPool(device, &poolInfo, nullptr, &command_pool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics command pool!");
  }
}

void fan::vulkan::context_t::cleanup_swap_chain() {
  cleanup_swap_chain_dependencies();
  if (swap_chain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device, swap_chain, nullptr);
    swap_chain = VK_NULL_HANDLE;
  }
}

void fan::vulkan::context_t::create_depth_resources() {
  vai_t::properties_t p;
  p.swap_chain_size = swap_chain_size;
  p.format = find_depth_format();
  p.usage_flags = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  p.aspect_flags = VK_IMAGE_ASPECT_DEPTH_BIT;
  vai_depth.open(*this, p);
}


void fan::vulkan::context_t::pipeline_t::close(fan::vulkan::context_t& context) {
  vkDestroyPipeline(context.device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(context.device, m_layout, nullptr);
}

void fan::vulkan::context_t::pipeline_t::open(fan::vulkan::context_t& context, const properties_t& p) {
  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = p.shape_type;
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
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_FALSE;//p.enable_depth_test;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = p.depth_test_compare_op;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_NO_OP;
  colorBlending.attachmentCount = p.color_blend_attachment_count;
  colorBlending.pAttachments = p.color_blend_attachment;
  colorBlending.blendConstants[0] = 1.0f;
  colorBlending.blendConstants[1] = 1.0f;
  colorBlending.blendConstants[2] = 1.0f;
  colorBlending.blendConstants[3] = 1.0f;

  std::vector<VkDynamicState> dynamicStates = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR
  };
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = dynamicStates.size();
  dynamicState.pDynamicStates = dynamicStates.data();

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = p.descriptor_layout_count;
  pipelineLayoutInfo.pSetLayouts = p.descriptor_layout;

  VkPushConstantRange push_constant;
  push_constant.offset = 0;
  push_constant.size = p.push_constants_size;
  push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  pipelineLayoutInfo.pPushConstantRanges = &push_constant;
  pipelineLayoutInfo.pushConstantRangeCount = 1;

  if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &m_layout) != VK_SUCCESS) {
    fan::throw_error("failed to create pipeline layout!");
  }

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = context.shader_get(p.shader).shader_stages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = m_layout;
  pipelineInfo.renderPass = context.render_pass;
  pipelineInfo.subpass = p.subpass;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

  if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
    fan::throw_error("failed to create graphics pipeline");
  }
}


//  void* data;
//  vkMapMemory(device, uniform_block.common.memory[currentImage].device_memory, 0, sizeof(ubo), 0, &data);
//  memcpy(data, &ubo, sizeof(ubo));
//  vkUnmapMemory(device, uniform_block.common.memory[currentImage].device_memory);
//}


// assumes things are already bound

void fan::vulkan::context_t::bindless_draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_instance) {
  vkCmdDraw(command_buffers[current_frame], vertex_count, instance_count, 0, first_instance);
}

void fan::vulkan::context_t::draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_instance, const fan::vulkan::context_t::pipeline_t& pipeline, uint32_t descriptor_count, VkDescriptorSet* descriptor_sets) {
  bind_draw(pipeline, descriptor_count, descriptor_sets);
  bindless_draw(vertex_count, instance_count, first_instance);
}

void fan::vulkan::context_t::create_sync_objects() {
  image_available_semaphores.resize(max_frames_in_flight);
  render_finished_semaphores.resize(max_frames_in_flight);
  in_flight_fences.resize(max_frames_in_flight);

  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < max_frames_in_flight; i++) {
    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &image_available_semaphores[i]) != VK_SUCCESS ||
      vkCreateSemaphore(device, &semaphoreInfo, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS ||
      vkCreateFence(device, &fenceInfo, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create synchronization objects for a frame!");
    }
  }
}

void fan::vulkan::context_t::bind_draw(const fan::vulkan::context_t::pipeline_t& pipeline, uint32_t descriptor_count, VkDescriptorSet* descriptor_sets) {
  vkCmdBindPipeline(command_buffers[current_frame], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.m_pipeline);

  VkRect2D scissor{};
  scissor.offset = { 0, 0 };
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

VkFormat fan::vulkan::context_t::find_supported_foramt(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
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

VkFormat fan::vulkan::context_t::find_depth_format() {
  return find_supported_foramt(
    { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
    VK_IMAGE_TILING_OPTIMAL,
    VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
  );
}

bool fan::vulkan::context_t::has_stencil_component(VkFormat format) {
  return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

void fan::vulkan::context_t::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
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
  allocInfo.memoryTypeIndex = find_memory_type(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate buffer memory!");
  }

  vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

VkCommandBuffer fan::vulkan::context_t::begin_single_time_commands() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = command_pool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  return commandBuffer;
}

void fan::vulkan::context_t::end_single_time_commands(VkCommandBuffer commandBuffer) {
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue);

  vkFreeCommandBuffers(device, command_pool, 1, &commandBuffer);
}

void fan::vulkan::context_t::copy_buffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
  VkCommandBuffer commandBuffer = begin_single_time_commands();

  VkBufferCopy copyRegion{};
  copyRegion.size = size;
  vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

  end_single_time_commands(commandBuffer);
}

uint32_t fan::vulkan::context_t::find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

void fan::vulkan::context_t::create_command_buffers() {
  command_buffers.resize(max_frames_in_flight);

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = command_pool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = (uint32_t)command_buffers.size();

  if (vkAllocateCommandBuffers(device, &allocInfo, command_buffers.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }
}

void fan::vulkan::context_t::begin_render(const fan::color& clear_color) {
  vkWaitForFences(device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);
  vkResetFences(device, 1, &in_flight_fences[current_frame]);

  vkResetCommandBuffer(command_buffers[current_frame], /*VkCommandBufferResetFlagBits*/ 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  if (vkBeginCommandBuffer(command_buffers[current_frame], &beginInfo) != VK_SUCCESS) {
    fan::throw_error("failed to begin recording command buffer!");
  }

  command_buffer_in_use = true;

  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = render_pass;
  renderPassInfo.framebuffer = swap_chain_framebuffers[image_index];
  renderPassInfo.renderArea.offset = { 0, 0 };
  renderPassInfo.renderArea.extent.width = swap_chain_size.x;
  renderPassInfo.renderArea.extent.height = swap_chain_size.y;

  // TODO

  VkClearValue clearValues[
    5
  ]{};
    clearValues[0].color = { { clear_color.r, clear_color.g, clear_color.b, clear_color.a } };
    clearValues[1].color = { { clear_color.r, clear_color.g, clear_color.b, clear_color.a } };
    clearValues[2].color = { { clear_color.r, clear_color.g, clear_color.b, clear_color.a } };
    clearValues[3].color = { { clear_color.r, clear_color.g, clear_color.b, clear_color.a } };
    clearValues[4].depthStencil = { 1.0f, 0 };

    renderPassInfo.clearValueCount = std::size(clearValues);
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(command_buffers[current_frame], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    //TODO set viewport
    //fan::vulkan::viewport_t::set_viewport(0, swap_chain_size, swap_chain_size);

    {//TODO WRAP
      VkViewport viewport = {};
      viewport.x = 0.0f;
      viewport.y = 0.0f;
      viewport.width = swap_chain_size.x;
      viewport.height = swap_chain_size.y;
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;

      vkCmdSetViewport(command_buffers[current_frame], 0, 1, &viewport);

      VkRect2D scissor = {};
      scissor.offset = { 0, 0 };
      scissor.extent.width = swap_chain_size.x; // make operator vkextent
      scissor.extent.height = swap_chain_size.y;

      vkCmdSetScissor(command_buffers[current_frame], 0, 1, &scissor);
    }

    vkCmdNextSubpass(command_buffers[current_frame], VK_SUBPASS_CONTENTS_INLINE);
    vkCmdEndRenderPass(command_buffers[current_frame]);
}

void fan::vulkan::context_t::ImGuiSetupVulkanWindow() {
  MainWindowData.Surface = surface;
  MainWindowData.SurfaceFormat = surface_format;
  MainWindowData.Swapchain = swap_chain;
  MainWindowData.PresentMode = present_mode;

  IM_ASSERT(MinImageCount >= 2);
  ImGui_ImplVulkanH_CreateOrResizeWindow(instance, physicalDevice, device, &MainWindowData, queue_family, /*g_Allocator*/nullptr, swap_chain_size.x, swap_chain_size.y, MinImageCount);
  swap_chain = MainWindowData.Swapchain;
  update_swapchain_dependencies();
}

void fan::vulkan::context_t::ImGuiFrameRender(VkResult next_image_khr_err, fan::color clear_color) {
  MainWindowData.ClearValue.color.float32[0] = clear_color[0];
  MainWindowData.ClearValue.color.float32[1] = clear_color[1];
  MainWindowData.ClearValue.color.float32[2] = clear_color[2];
  MainWindowData.ClearValue.color.float32[3] = clear_color[3];
  ImGui_ImplVulkanH_Window* wd = &MainWindowData;
  VkResult err = next_image_khr_err;
  if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
    SwapChainRebuild = true;
  if (err == VK_ERROR_OUT_OF_DATE_KHR)
    return;
  if (err != VK_SUBOPTIMAL_KHR)
    fan::vulkan::validate(err);

  wd->FrameIndex = image_index;

  ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];

  VkRenderPassBeginInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  info.renderPass = wd->RenderPass;
  info.framebuffer = fd->Framebuffer;
  info.renderArea.extent.width = wd->Width;
  info.renderArea.extent.height = wd->Height;
  info.clearValueCount = 1;
  info.pClearValues = &wd->ClearValue;
  vkCmdBeginRenderPass(command_buffers[current_frame], &info, VK_SUBPASS_CONTENTS_INLINE);

  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffers[current_frame]);

  vkCmdEndRenderPass(command_buffers[current_frame]);
}

VkResult fan::vulkan::context_t::end_render() {
  //// render_fullscreen_pl loco fbo?
  if (!command_buffer_in_use) {
    return VK_SUCCESS;
  }
  if (vkEndCommandBuffer(command_buffers[current_frame]) != VK_SUCCESS) {
    fan::throw_error("failed to record command buffer!");
  }

  command_buffer_in_use = false;

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore waitSemaphores[] = { image_available_semaphores[current_frame] };
  VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffers[current_frame];

  VkSemaphore signalSemaphores[] = { render_finished_semaphores[current_frame] };
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  if (vkQueueSubmit(graphics_queue, 1, &submitInfo, in_flight_fences[current_frame]) != VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }
  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = signalSemaphores;

  VkSwapchainKHR swapChains[] = { swap_chain };
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;
  presentInfo.pImageIndices = &image_index;
  auto result = vkQueuePresentKHR(present_queue, &presentInfo);

  current_frame = (current_frame + 1) % max_frames_in_flight;
  return result;
}

VkSurfaceFormatKHR fan::vulkan::context_t::choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
  for (const auto& availableFormat : availableFormats) {
    // VK_FORMAT_B8G8R8A8_SRGB

    if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return availableFormat;
    }
  }

  return availableFormats[0];
}

VkPresentModeKHR fan::vulkan::context_t::choose_swap_present_mode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
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

VkExtent2D fan::vulkan::context_t::choose_swap_extent(const fan::vec2ui& framebuffer_size, const VkSurfaceCapabilitiesKHR& capabilities) {
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

void fan::vulkan::context_t::end_compute_shader() {
  if (vkEndCommandBuffer(command_buffers[current_frame]) != VK_SUCCESS) {
    fan::throw_error("failed to record command buffer!");
  }

  command_buffer_in_use = false;

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffers[current_frame];

  if (vkQueueSubmit(graphics_queue, 1, &submitInfo, in_flight_fences[current_frame]) != VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }
}

void fan::vulkan::context_t::begin_compute_shader() {
  //?
  //vkWaitForFences(device, 1, &inFlightFences[current_frame], VK_TRUE, UINT64_MAX);

  vkResetFences(device, 1, &in_flight_fences[current_frame]);

  vkResetCommandBuffer(command_buffers[current_frame], /*VkCommandBufferResetFlagBits*/ 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  if (vkBeginCommandBuffer(command_buffers[current_frame], &beginInfo) != VK_SUCCESS) {
    fan::throw_error("failed to begin recording command buffer!");
  }

  command_buffer_in_use = true;
}

swap_chain_support_details_t fan::vulkan::context_t::query_swap_chain_support(VkPhysicalDevice device) {
  swap_chain_support_details_t details;

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

queue_family_indices_t fan::vulkan::context_t::find_queue_families(VkPhysicalDevice device) {
  queue_family_indices_t indices;

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

    if (indices.is_complete()) {
      break;
    }

    i++;
  }

  return indices;
}

std::vector<fan::string> fan::vulkan::context_t::get_required_extensions() {

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

bool fan::vulkan::context_t::check_validation_layer_support() {
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

VKAPI_ATTR VkBool32 VKAPI_CALL fan::vulkan::context_t::debug_callback(
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
void fan::vulkan::context_t::set_vsync(fan::window_t* window, bool flag) {
  vsync = flag;
  recreate_swap_chain(window->get_size());
}
#endif

uint32_t fan::vulkan::makeAccessMaskPipelineStageFlags(uint32_t accessMask) {
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

void fan::vulkan::vai_t::open(auto& context, const properties_t& p) {
  fan::vulkan::create_image(
    context,
    p.swap_chain_size,
    p.format,
    VK_IMAGE_TILING_OPTIMAL,
    p.usage_flags,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    image,
    memory
  );
  image_view = context.create_image_view(image, p.format, p.aspect_flags);
  format = p.format;
}

void fan::vulkan::vai_t::close(auto& context) {
  vkDestroyImageView(context.device, image_view, nullptr);
  vkDestroyImage(context.device, image, nullptr);
  vkFreeMemory(context.device, memory, nullptr);
}

void fan::vulkan::vai_t::transition_image_layout(auto& context, VkImageLayout newLayout) {
  VkCommandBuffer commandBuffer = context.begin_single_time_commands();

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

  context.end_single_time_commands(commandBuffer);

  old_layout = newLayout;
}

void fan::vulkan::context_t::descriptor_pool_t::open(fan::vulkan::context_t& context) {
  VkDescriptorPoolSize pool_sizes[] =
  {
    { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE },
  };
  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 0;
  for (VkDescriptorPoolSize& pool_size : pool_sizes)
    pool_info.maxSets += pool_size.descriptorCount;
  pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;
  ;
  fan::vulkan::validate(vkCreateDescriptorPool(context.device, &pool_info, nullptr, &m_descriptor_pool));
}

void fan::vulkan::context_t::descriptor_pool_t::close(fan::vulkan::context_t& context) {
  vkDestroyDescriptorPool(context.device, m_descriptor_pool, nullptr);
}

void fan::vulkan::context_t::create_loco_framebuffer() {
  vai_t::properties_t p;
  p.format = swap_chain_image_format;
  p.swap_chain_size = swap_chain_size;
  p.usage_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  p.aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT;
  vai_bitmap[0].open(*this, p);
  vai_bitmap[0].transition_image_layout(*this, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  p.format = VK_FORMAT_B8G8R8A8_UNORM; // TODO should it be VK_FORMAT_R8_UINT?
  vai_bitmap[1].open(*this, p);
  vai_bitmap[1].transition_image_layout(*this, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}