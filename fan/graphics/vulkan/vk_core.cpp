#if defined(fan_vulkan)
#include "core.h"

#include <fan/physics/collision/rectangle.h>

using namespace fan::graphics;

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
  return graphics_family.has_value()
#if defined(loco_window)
    && present_family.has_value()
#endif
    ;
}

void fan::vulkan::validate(VkResult result) {
  if (result != VK_SUCCESS) {
    fan::throw_error("function failed");
  }
}

VkPipelineColorBlendAttachmentState fan::vulkan::get_default_color_blend() {
  VkPipelineColorBlendAttachmentState color_blend_attachment{};
  color_blend_attachment.colorWriteMask =
    VK_COLOR_COMPONENT_R_BIT |
    VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT |
    VK_COLOR_COMPONENT_A_BIT
  ;
  color_blend_attachment.blendEnable = VK_TRUE;
  color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
  color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
  return color_blend_attachment;
}


std::vector<uint32_t> fan::vulkan::context_t::compile_file(
  const std::string& source_name,
  shaderc_shader_kind kind,
  const std::string& source) {
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

fan::vulkan::context_t::shader_t& shader_get(fan::vulkan::context_t& context, shader_nr_t nr) {
  return *(fan::vulkan::context_t::shader_t*)__fan_internal_shader_list[nr].internal;
}

fan::graphics::shader_nr_t shader_create(fan::vulkan::context_t& context) {
  shader_nr_t nr = __fan_internal_shader_list.NewNode();
  __fan_internal_shader_list[nr].internal = new fan::vulkan::context_t::shader_t;
  auto& shader = shader_get(context, nr);
  //TODO
  shader.projection_view_block.open(context);
  for (uint32_t i = 0; i < fan::vulkan::max_camera; ++i) {
    shader.projection_view_block.push_ram_instance(context, {});
  }
  return nr;
}

void shader_erase(fan::vulkan::context_t& context, shader_nr_t nr, int recycle = 1) {
  auto& shader = shader_get(context, nr);
  if (shader.shader_stages[0].module) {
    vkDestroyShaderModule(context.device, shader.shader_stages[0].module, nullptr);
  }
  if (shader.shader_stages[1].module) {
    vkDestroyShaderModule(context.device, shader.shader_stages[1].module, nullptr);
  }
  //TODO
  shader.projection_view_block.close(context);
  delete static_cast<fan::vulkan::context_t::shader_t*>(__fan_internal_shader_list[nr].internal);
  if (recycle) {
    __fan_internal_shader_list.Recycle(nr);
  }
}

void shader_use(fan::vulkan::context_t& context, shader_nr_t nr) {
  auto& shader = shader_get(context, nr);
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

void shader_set_vertex(fan::vulkan::context_t& context, shader_nr_t nr, const std::string& vertex_code) {
  __fan_internal_shader_list[nr].svertex = vertex_code;
  // fan::print(
  //   "processed vertex shader:", path, "resulted in:",
  // preprocess_shader(shader_name.c_str(), shaderc_glsl_vertex_shader, shader_code);
  // );
}

void shader_set_fragment(fan::vulkan::context_t& context, shader_nr_t nr, const std::string& fragment_code) {
  auto& shader = shader_get(context, nr);
  __fan_internal_shader_list[nr].sfragment = fragment_code;
  //fan::print(
    // "processed vertex shader:", path, "resulted in:",
  //preprocess_shader(shader_name.c_str(), shaderc_glsl_fragment_shader, shader_code);
  //);
}

static void parse_uniforms(std::string& shaderData, std::unordered_map<std::string, std::string>& uniform_type_table) {
  size_t pos = 0;

  while ((pos = shaderData.find("uniform", pos)) != std::string::npos) {
    size_t endLine = shaderData.find(';', pos);
    if (endLine == std::string::npos) break;

    std::string line = shaderData.substr(pos, endLine - pos + 1);

    line = line.substr(7); 
    
    size_t start = line.find_first_not_of(" \t");
    if (start == std::string::npos) {
      pos = endLine + 1;
      continue;
    }
    line = line.substr(start);

    size_t space1 = line.find_first_of(" \t");
    if (space1 == std::string::npos) {
      pos = endLine + 1;
      continue;
    }

    std::string type = line.substr(0, space1);
    line = line.substr(space1);
    line = line.substr(line.find_first_not_of(" \t"));

    size_t varEnd = line.find_first_of("=;");
    std::string name = line.substr(0, varEnd);
    
    name.erase(0, name.find_first_not_of(" \t"));
    name.erase(name.find_last_not_of(" \t") + 1);

    uniform_type_table[name] = type;

    pos = endLine + 1;
  }
}

bool shader_compile(fan::vulkan::context_t& context, shader_nr_t nr) {
  auto& shader = shader_get(context, nr);
  {
    auto spirv = context.compile_file(/*vertex_code.c_str()*/ "some vertex file", shaderc_glsl_vertex_shader, __fan_internal_shader_list[nr].svertex);

    auto module_vertex = create_shader_module(context, spirv);

    VkPipelineShaderStageCreateInfo vert{};
    vert.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert.module = module_vertex;
    vert.pName = "main";

    shader.shader_stages[0] = vert;
  }
  {
    auto spirv = context.compile_file(/*shader_name.c_str()*/"some fragment file", shaderc_glsl_fragment_shader, __fan_internal_shader_list[nr].sfragment);

    auto module_fragment = create_shader_module(context, spirv);

    VkPipelineShaderStageCreateInfo frag{};
    frag.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag.module = module_fragment;
    frag.pName = "main";

    shader.shader_stages[1] = frag;
  }

  std::string vertexData = __fan_internal_shader_list[nr].svertex;
  parse_uniforms(vertexData, __fan_internal_shader_list[nr].uniform_type_table);

  std::string fragmentData = __fan_internal_shader_list[nr].sfragment;
  parse_uniforms(fragmentData, __fan_internal_shader_list[nr].uniform_type_table);

  return 0;
}


void fan::vulkan::image_create(fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
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
  allocInfo.memoryTypeIndex = context.find_memory_type(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(context.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate image memory!");
  }

  vkBindImageMemory(context.device, image, imageMemory, 0);
}

void fan::vulkan::context_t::transition_image_layout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
  VkCommandBuffer command_buffer = begin_single_time_commands();

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
      command_buffer,
      sourceStage, destinationStage,
      0,
      0, nullptr,
      0, nullptr,
      1, &barrier
  );

  end_single_time_commands(command_buffer);
}

constexpr static uint32_t get_image_multiplier(VkFormat format) {
  switch (format) {
  case fan::vulkan::context_t::image_format::b8g8r8a8_unorm: {
    return 4;
  }
  case fan::vulkan::context_t::image_format::r8_unorm: {
    return 4; // 1?
  }
  case fan::vulkan::context_t::image_format::r8g8b8a8_srgb: {
    return 4;
  }
  case fan::vulkan::context_t::image_format::r8b8g8a8_unorm: {
    return 4;
  }
  default: {// removes warning
    break;
  }
  }
  fan::throw_error("failed to find format for image multiplier");
  return {};
}

void fan::vulkan::context_t::copy_buffer_to_image(VkBuffer buffer, VkImage image, VkFormat format, const fan::vec2ui& size, const fan::vec2ui& stride) {
  VkCommandBuffer command_buffer = begin_single_time_commands();

  uint32_t block_width = get_image_multiplier(format);
  uint32_t block_x = (block_width - 1) / block_width;
  uint32_t block_y = (block_width - 1) / block_width;
  uint32_t block_h = std::max(1u, (size.y + block_width - 1) / block_width);
  // Flush CPU and GPU caches if not coherent mapping.
  VkDeviceSize buffer_flush_offset = block_y * stride.x;
  VkDeviceSize buffer_flush_size = block_h * stride.x;

  /*
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
  size.x,
  size.y,
  1
  };
  */

  VkBufferImageCopy region = {
      block_y * stride.x + block_x,// VkDeviceSize             bufferOffset
      size.x,                                        // uint32_t                 bufferRowLength
      0,                                              // uint32_t                 bufferImageHeight
      { 0, 0, 0, 1 },                  // VkImageSubresourceLayers imageSubresource
      { 0, 0, 0 },  // VkOffset3D               imageOffset
      { size.x, size.y, 1 }                              // VkExtent3D               imageExtent
  };

  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;

  vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  end_single_time_commands(command_buffer);
}

void fan::vulkan::context_t::create_texture_sampler(VkSampler& sampler, const image_load_properties_t& lp) {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device, &properties);

  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = lp.mag_filter;
  samplerInfo.minFilter = lp.min_filter;
  samplerInfo.addressModeU = lp.visual_output;
  samplerInfo.addressModeV = lp.visual_output;
  samplerInfo.addressModeW = lp.visual_output;
  samplerInfo.anisotropyEnable = VK_TRUE;
  samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
    throw std::runtime_error("failed to create texture sampler!");
  }
}

fan::graphics::image_nr_t image_create(fan::vulkan::context_t& context) {
  image_nr_t nr = __fan_internal_image_list.NewNode();
  __fan_internal_image_list[nr].internal = new fan::vulkan::context_t::image_t;
  return nr;
}

uint64_t image_get_handle(fan::vulkan::context_t& context, image_nr_t nr) {
  fan::throw_error("invalid call");
  return 0;
}

fan::vulkan::context_t::image_t& image_get(fan::vulkan::context_t& context, image_nr_t nr) {
  return *(fan::vulkan::context_t::image_t*)__fan_internal_image_list[nr].internal;
}

void image_erase(fan::vulkan::context_t& context, image_nr_t nr, int recycle = 1) {
  fan::vulkan::context_t::image_t& image = image_get(context, nr);
  vkDestroySampler(context.device, image.sampler, nullptr);
  vkDestroyBuffer(context.device, image.staging_buffer, nullptr);
  vkFreeMemory(context.device, image.staging_buffer_memory, nullptr);
  vkDestroyImage(context.device, image.image_index, 0);
  vkDestroyImageView(context.device, image.image_view, 0);
  vkFreeMemory(context.device, image.image_memory, nullptr);
  delete static_cast<fan::vulkan::context_t::image_t*>(__fan_internal_image_list[nr].internal);
  if (recycle) {
    __fan_internal_image_list.Recycle(nr);
  }
}

void image_bind(fan::vulkan::context_t& context, image_nr_t nr) {
  
}

void image_unbind(fan::vulkan::context_t& context, image_nr_t nr) {
  
}

fan::graphics::image_load_properties_t& image_get_settings(fan::vulkan::context_t& context, image_nr_t nr) {
  return __fan_internal_image_list[nr].image_settings;
}

void image_set_settings(fan::vulkan::context_t& context, const fan::vulkan::context_t::image_load_properties_t& p) {

}

fan::graphics::image_nr_t image_load(fan::vulkan::context_t& context, const fan::image::info_t& image_info, const fan::vulkan::context_t::image_load_properties_t& p) {
  image_nr_t nr = image_create(context);

  fan::vulkan::context_t::image_t& image = image_get(context, nr);
  auto& image_data = __fan_internal_image_list[nr];
  image_data.size = image_info.size;
  __fan_internal_image_list[nr].image_path = "";

  auto image_multiplier = get_image_multiplier(p.format);

  VkDeviceSize image_size_bytes = image_info.size.multiply() * image_multiplier;

  context.create_buffer(
    image_size_bytes,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    //VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    image.staging_buffer,
    image.staging_buffer_memory
  );

  vkMapMemory(context.device, image.staging_buffer_memory, 0, image_size_bytes, 0, &image.data);
  memcpy(image.data, image_info.data, image_size_bytes); // TODO  / 4 in yuv420p

  fan::vulkan::image_create(
    context,
    image_info.size,
    p.format,
    VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    image.image_index,
    image.image_memory
  );
  image.image_view = context.create_image_view(image.image_index, p.format, VK_IMAGE_ASPECT_COLOR_BIT);
  context.create_texture_sampler(image.sampler, p);

  context.transition_image_layout(image.image_index, p.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  context.copy_buffer_to_image(image.staging_buffer, image.image_index, p.format, image_info.size);
  context.transition_image_layout(image.image_index, p.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  return nr;
}

fan::graphics::image_nr_t image_load(fan::vulkan::context_t& context, const fan::image::info_t& image_info) {
  return image_load(context, image_info, fan::vulkan::context_t::image_load_properties_t());
}

fan::graphics::image_nr_t image_load(fan::vulkan::context_t& context, fan::color* colors, const fan::vec2ui& size_, const fan::vulkan::context_t::image_load_properties_t& p) {

  fan::image::info_t ii;
  ii.data = colors;
  ii.size = size_;
  ii.channels = 4;
  image_nr_t nr = image_load(context, ii, p);

  image_set_settings(context, p);

  auto& image_data = __fan_internal_image_list[nr];
  image_data.size = size_;

  return nr;
}

fan::graphics::image_nr_t image_load(fan::vulkan::context_t& context, fan::color* colors, const fan::vec2ui& size_) {
  return image_load(context, colors, size_, fan::vulkan::context_t::image_load_properties_t());
}

fan::graphics::image_nr_t create_missing_texture(fan::vulkan::context_t& context) {
  fan::vulkan::context_t::image_load_properties_t p;

  fan::vec2i image_size = fan::vec2i(2, 2);
  image_nr_t nr = image_load(context, (fan::color*)fan::image::missing_texture_pixels, image_size, p);

  auto& image_data = __fan_internal_image_list[nr];
  image_data.size = image_size;
  __fan_internal_image_list[nr].image_settings = image_global_to_vulkan(p);
  return nr;
}
fan::graphics::image_nr_t create_transparent_texture(fan::vulkan::context_t& context) {
  fan::vulkan::context_t::image_load_properties_t p;

  fan::vec2i image_size = fan::vec2i(2, 2);
  image_nr_t nr = image_load(context, (fan::color*)fan::image::transparent_texture_pixels, image_size, p);

  auto& image_data = __fan_internal_image_list[nr];
  image_data.size = image_size;

  return nr;
}

fan::graphics::image_nr_t image_load(fan::vulkan::context_t& context, const std::string& path, const fan::vulkan::context_t::image_load_properties_t& p) {

#if fan_assert_if_same_path_loaded_multiple_times

  static std::unordered_map<std::string, bool> existing_images;

  if (existing_images.find(path) != existing_images.end()) {
    fan::throw_error("image already existing " + path);
  }

  existing_images[path] = 0;

#endif

  fan::image::info_t image_info;
  if (fan::image::load(path, &image_info)) {
    return create_missing_texture(context);
  }
  image_nr_t nr = image_load(context, image_info, p);
  __fan_internal_image_list[nr].image_path = path;
  fan::image::free(&image_info);
  return nr;
}

fan::graphics::image_nr_t image_load(fan::vulkan::context_t& context, const std::string& path) {
  return image_load(context, path, fan::vulkan::context_t::image_load_properties_t());
}

void image_unload(fan::vulkan::context_t& context, image_nr_t nr) {
  image_erase(context, nr);
}

void image_reload(fan::vulkan::context_t& context, image_nr_t nr, const fan::image::info_t& image_info, const fan::vulkan::context_t::image_load_properties_t& p) {
  auto image_multiplier = get_image_multiplier(p.format);

  VkDeviceSize image_size = image_info.size.multiply() * image_multiplier;

  fan::vulkan::context_t::image_t& image = image_get(context, nr);
  auto& image_data = __fan_internal_image_list[nr];
  image_data.size = image_info.size;

  if (image.image_index == 0) {
    VkDeviceSize image_size_bytes = image_info.size.multiply() * image_multiplier;

    context.create_buffer(
      image_size_bytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      //VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      image.staging_buffer,
      image.staging_buffer_memory
    );

    vkMapMemory(context.device, image.staging_buffer_memory, 0, image_size_bytes, 0, &image.data);
    memcpy(image.data, image_info.data, image_size_bytes); // TODO  / 4 in yuv420p

    fan::vulkan::image_create(
      context,
      image_info.size,
      p.format,
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      image.image_index,
      image.image_memory
    );
    image.image_view = context.create_image_view(image.image_index, p.format, VK_IMAGE_ASPECT_COLOR_BIT);
    context.create_texture_sampler(image.sampler, p);
  }

  memcpy(image.data, image_info.data, image_size / 4);


  context.transition_image_layout(image.image_index, p.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  context.copy_buffer_to_image(image.staging_buffer, image.image_index, p.format, image_info.size);
  context.transition_image_layout(image.image_index, p.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void image_reload(fan::vulkan::context_t& context, image_nr_t nr, const fan::image::info_t& image_info) {
  image_reload(context, nr, image_info, fan::vulkan::context_t::image_load_properties_t());
}

void image_reload(fan::vulkan::context_t& context, image_nr_t nr, const std::string& path, const fan::vulkan::context_t::image_load_properties_t& p) {
  fan::image::info_t image_info;
  if (fan::image::load(path, &image_info)) {
    image_info.data = (void*)fan::image::missing_texture_pixels;
    image_info.size = 2;
    image_info.channels = 4;
    image_info.type = -1; // ignore free
  }
  image_reload(context, nr, image_info, p);
  __fan_internal_image_list[nr].image_path = path;
  fan::image::free(&image_info);
}

void image_reload(fan::vulkan::context_t& context, image_nr_t nr, const std::string& path) {
  image_reload(context, nr, path, fan::vulkan::context_t::image_load_properties_t());
}

// creates single colored text size.x*size.y sized
fan::graphics::image_nr_t image_create(fan::vulkan::context_t& context, const fan::color& color, const fan::vulkan::context_t::image_load_properties_t& p) {

  uint8_t pixels[4];
  for (uint32_t p = 0; p < fan::color::size(); p++) {
    pixels[p] = color[p] * 255;
  }

  fan::image::info_t ii;

  ii.data = (void*)&color.r;
  ii.size = 1;
  ii.channels = 4;
  image_nr_t nr = image_load(context, ii, p);

  image_bind(context, nr);

  image_set_settings(context, p);

  return nr;
}

fan::graphics::image_nr_t image_create(fan::vulkan::context_t& context, const fan::color& color) {
  return image_create(context, color, fan::vulkan::context_t::image_load_properties_t());
}

fan::graphics::camera_nr_t camera_create(fan::vulkan::context_t& context) {
  return __fan_internal_camera_list.NewNode();
}

fan::graphics::context_camera_t& camera_get(fan::vulkan::context_t& context, camera_nr_t nr) {
  return __fan_internal_camera_list[nr];
}

void camera_erase(fan::vulkan::context_t& context, camera_nr_t nr) {
  __fan_internal_camera_list.Recycle(nr);
}

void camera_set_ortho(fan::vulkan::context_t& context, camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
  fan::graphics::context_camera_t& camera = camera_get(context, nr);

  camera.coordinates.left = x.x;
  camera.coordinates.right = x.y;
  camera.coordinates.down = y.y;
  camera.coordinates.up = y.x;

  camera.m_projection = fan::math::ortho<fan::mat4>(
    camera.coordinates.left,
    camera.coordinates.right,
    camera.coordinates.up,
    camera.coordinates.down,
    -fan::graphics::znearfar / 2,
    fan::graphics::znearfar / 2
  );

  camera.m_view[3][0] = 0;
  camera.m_view[3][1] = 0;
  camera.m_view[3][2] = 0;
  camera.m_view = camera.m_view.translate(camera.position);
  fan::vec3 position = camera.m_view.get_translation();
  constexpr fan::vec3 front(0, 0, 1);

  camera.m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);

  //auto it = gloco->m_viewport_resize_callback.GetNodeFirst();

  //while (it != gloco->m_viewport_resize_callback.dst) {

  //  gloco->m_viewport_resize_callback.StartSafeNext(it);

  //  resize_cb_data_t cbd;
  //  cbd.camera = this;
  //  cbd.position = get_position();
  //  cbd.size = get_camera_size();
  //  gloco->m_viewport_resize_callback[it].data(cbd);

  //  it = gloco->m_viewport_resize_callback.EndSafeNext();
  //}
}

fan::graphics::camera_nr_t camera_create(fan::vulkan::context_t& context, const fan::vec2& x, const fan::vec2& y) {
  camera_nr_t nr = camera_create(context);
  camera_set_ortho(context, nr, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
  return nr;
}

fan::vec3 camera_get_position(fan::vulkan::context_t& context, camera_nr_t nr) {
  return camera_get(context, nr).position;
}

void camera_set_position(fan::vulkan::context_t& context, camera_nr_t nr, const fan::vec3& cp) {
  fan::graphics::context_camera_t& camera = camera_get(context, nr);
  camera.position = cp;


  camera.m_view[3][0] = 0;
  camera.m_view[3][1] = 0;
  camera.m_view[3][2] = 0;
  camera.m_view = camera.m_view.translate(camera.position);
  fan::vec3 position = camera.m_view.get_translation();
  constexpr fan::vec3 front(0, 0, 1);

  camera.m_view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
}

fan::vec2 camera_get_size(fan::vulkan::context_t& context, camera_nr_t nr) {
  fan::graphics::context_camera_t& camera = camera_get(context, nr);
  return fan::vec2(std::abs(camera.coordinates.right - camera.coordinates.left), std::abs(camera.coordinates.down - camera.coordinates.up));
}

void camera_set_perspective(fan::vulkan::context_t& context, camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
  fan::graphics::context_camera_t& camera = camera_get(context, nr);

  camera.m_projection = fan::math::perspective<fan::mat4>(fan::math::radians(fov), (f32_t)window_size.x / (f32_t)window_size.y, camera.znear, camera.zfar);

  camera.update_view();

  camera.m_view = camera.get_view_matrix();

  //auto it = gloco->m_viewport_resize_callback.GetNodeFirst();

  //while (it != gloco->m_viewport_resize_callback.dst) {

  //  gloco->m_viewport_resize_callback.StartSafeNext(it);

  //  resize_cb_data_t cbd;
  //  cbd.camera = this;
  //  cbd.position = get_position();
  //  cbd.size = get_camera_size();
  //  gloco->m_viewport_resize_callback[it].data(cbd);

  //  it = gloco->m_viewport_resize_callback.EndSafeNext();
  //}
}

void camera_rotate(fan::vulkan::context_t& context, camera_nr_t nr, const fan::vec2& offset) {
  fan::graphics::context_camera_t& camera = camera_get(context, nr);
  camera.rotate_camera(offset);
  camera.m_view = camera.get_view_matrix();
}

void viewport_set(fan::vulkan::context_t& context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  VkViewport viewport{};
  viewport.x = viewport_position_.x;
  viewport.y = viewport_position_.y;
  viewport.width = viewport_size_.x;
  viewport.height = viewport_size_.y;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  if (!context.command_buffer_in_use) {
    VkResult result = vkGetFenceStatus(context.device, context.in_flight_fences[context.current_frame]);
    if (result == VK_NOT_READY) {
      vkDeviceWaitIdle(context.device);
    }

    if (vkBeginCommandBuffer(context.command_buffers[context.current_frame], &beginInfo) != VK_SUCCESS) {
      fan::throw_error("failed to begin recording command buffer!");
    }
  }
  vkCmdSetViewport(context.command_buffers[context.current_frame], 0, 1, &viewport);

  if (!context.command_buffer_in_use) {
    if (vkEndCommandBuffer(context.command_buffers[context.current_frame]) != VK_SUCCESS) {
      fan::throw_error("failed to record command buffer!");
    }
    context.command_buffer_in_use = false;
  }
}

fan::graphics::context_viewport_t& viewport_get(fan::vulkan::context_t& context, viewport_nr_t nr) {
  return __fan_internal_viewport_list[nr];
}

void viewport_set(fan::vulkan::context_t& context, viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  fan::graphics::context_viewport_t& viewport = viewport_get(context, nr);
  viewport.viewport_position = viewport_position_;
  viewport.viewport_size = viewport_size_;

  viewport_set(context, viewport_position_, viewport_size_, window_size);
}

fan::graphics::viewport_nr_t viewport_create(fan::vulkan::context_t& context)
{
  auto nr = __fan_internal_viewport_list.NewNode();

  viewport_set(
    context,
    nr,
    0, 1, 0
  );
  return nr;
}

void viewport_erase(fan::vulkan::context_t& context, viewport_nr_t nr) {
  __fan_internal_viewport_list.Recycle(nr);
}

fan::vec2 viewport_get_position(fan::vulkan::context_t& context, viewport_nr_t nr) {
  return viewport_get(context, nr).viewport_position;
}

fan::vec2 viewport_get_size(fan::vulkan::context_t& context, viewport_nr_t nr) {
  return viewport_get(context, nr).viewport_size;
}

void viewport_zero(fan::vulkan::context_t& context, viewport_nr_t nr) {
  auto& viewport = viewport_get(context, nr);
  viewport.viewport_position = 0;
  viewport.viewport_size = 0;
  viewport_set(context, 0, 0, 0); // window_size not used
}

bool viewport_inside(fan::vulkan::context_t& context, viewport_nr_t nr, const fan::vec2& position) {
  fan::graphics::context_viewport_t& viewport = viewport_get(context, nr);
  return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport.viewport_position + viewport.viewport_size / 2, viewport.viewport_size / 2);
}

bool viewport_inside_wir(fan::vulkan::context_t& context, viewport_nr_t nr, const fan::vec2& position) {
  fan::graphics::context_viewport_t& viewport = viewport_get(context, nr);
  return fan_2d::collision::rectangle::point_inside_no_rotation(position, viewport.viewport_size / 2, viewport.viewport_size / 2);
}
#include <fan/time/time.h>
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

  create_command_pool();
  create_image_views();
  create_render_pass();
  create_framebuffers();
  create_command_buffers();
  create_sync_objects();
  descriptor_pool.open(*this);
#if defined(fan_gui)
  ImGuiSetupVulkanWindow();
#endif

  //{
  //  VkImageMemoryBarrier barrier = {};
  //  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  //  barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Layout after the first render pass
  //  barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Layout for the second render pass
  //  barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // Access in the first pass
  //  barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // Access in the second pass
  //  barrier.image = swap_chain; // Your color attachment image
  //  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // For color attachments
  //  barrier.subresourceRange.baseMipLevel = 0;
  //  barrier.subresourceRange.levelCount = 1;
  //  barrier.subresourceRange.baseArrayLayer = 0;
  //  barrier.subresourceRange.layerCount = 1;

  //  // Insert the pipeline barrier
  //  vkCmdPipelineBarrier(commandBuffer,
  //    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // Source stage
  //    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // Destination stage
  //    0, // No dependency flags
  //    0, nullptr, // No memory barriers
  //    0, nullptr, // No buffer barriers
  //    1, &barrier); // One image barrier
  //}

}

void close_vais(fan::vulkan::context_t& c, std::vector<fan::vulkan::vai_t>& v) {
  for (auto& e : v) {
    e.close(c);
  }
}

void fan::vulkan::context_t::destroy_vulkan_soft() {
  vkDeviceWaitIdle(device);
  fan::vulkan::context_t& context = *this;
  {
    fan::graphics::shader_list_t::nrtra_t nrtra;
    fan::graphics::shader_list_t::nr_t nr;
    nrtra.Open(&__fan_internal_shader_list, &nr);
    while(nrtra.Loop(&__fan_internal_shader_list, &nr)) {
      shader_erase(*this, nr, 0);
    }
    nrtra.Close(&__fan_internal_shader_list);
  }
  {
    fan::graphics::image_list_t::nrtra_t nrtra;
    fan::graphics::image_list_t::nr_t nr;
    nrtra.Open(&__fan_internal_image_list, &nr);
    while(nrtra.Loop(&__fan_internal_image_list, &nr)) {
      image_erase(*this, nr, 0);
    }
    nrtra.Close(&__fan_internal_image_list);
  }

  close_vais(*this, mainColorImageViews);
  close_vais(*this, postProcessedColorImageViews);
  close_vais(*this, depthImageViews);
  close_vais(*this, downscaleImageViews1);
  close_vais(*this, upscaleImageViews1);
  close_vais(*this, vai_depth);
  
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
  #if defined(fan_gui)
  ImGui_ImplVulkanH_DestroyWindow(instance, device, &MainWindowData, nullptr);
#endif

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}

void fan::vulkan::context_t::close() {
  vkDeviceWaitIdle(device);
  cleanup_swap_chain();
  vkDestroySurfaceKHR(instance, surface, nullptr);
  destroy_vulkan_soft();
  fan::vulkan::context_t& context = *this;
  {
    fan::graphics::camera_list_t::nrtra_t nrtra;
    fan::graphics::camera_list_t::nr_t nr;
    nrtra.Open(&__fan_internal_camera_list, &nr);
    while(nrtra.Loop(&__fan_internal_camera_list, &nr)) {
      camera_erase(context, nr);
    }
    nrtra.Close(&__fan_internal_camera_list);
  }
  {
    fan::graphics::shader_list_t::nrtra_t nrtra;
    fan::graphics::shader_list_t::nr_t nr;
    nrtra.Open(&__fan_internal_shader_list, &nr);
    while(nrtra.Loop(&__fan_internal_shader_list, &nr)) {
      shader_erase(context, nr);
    }
    nrtra.Close(&__fan_internal_shader_list);
  }
  {
    fan::graphics::image_list_t::nrtra_t nrtra;
    fan::graphics::image_list_t::nr_t nr;
    nrtra.Open(&__fan_internal_image_list, &nr);
    while(nrtra.Loop(&__fan_internal_image_list, &nr)) {
      image_erase(context, nr);
    }
    nrtra.Close(&__fan_internal_image_list);
  }
  {
    fan::graphics::viewport_list_t::nrtra_t nrtra;
    fan::graphics::viewport_list_t::nr_t nr;
    nrtra.Open(&__fan_internal_viewport_list, &nr);
    while(nrtra.Loop(&__fan_internal_viewport_list, &nr)) {
      viewport_erase(context, nr);
    }
    nrtra.Close(&__fan_internal_viewport_list);
  }
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}

void fan::vulkan::context_t::cleanup_swap_chain_dependencies() {
  vkDeviceWaitIdle(device);
  close_vais(*this, mainColorImageViews);
  close_vais(*this, postProcessedColorImageViews);
  close_vais(*this, depthImageViews);
  close_vais(*this, downscaleImageViews1);
  close_vais(*this, upscaleImageViews1);
  close_vais(*this, vai_depth);
  
  for (auto framebuffer : swap_chain_framebuffers) {
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  }
  for (auto& i : swap_chain_image_views) {
    vkDestroyImageView(device, i, nullptr);
  }
}

void fan::vulkan::context_t::recreate_swap_chain_dependencies() {
  create_image_views();
  create_framebuffers();
}

// if swapchain changes, reque

void fan::vulkan::context_t::update_swapchain_dependencies() {
  
  uint32_t imageCount = 
    #if defined(fan_gui)
    MinImageCount + 1
#else 
    min_image_count + 1
#endif
    ;
  vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, nullptr);
  swap_chain_images.resize(imageCount);
  mainColorImageViews.resize(imageCount);
  postProcessedColorImageViews.resize(imageCount);
  depthImageViews.resize(imageCount);
  downscaleImageViews1.resize(imageCount);
  upscaleImageViews1.resize(imageCount);
  vai_depth.resize(imageCount);

  vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, swap_chain_images.data());
  recreate_swap_chain_dependencies();
}

void fan::vulkan::context_t::recreate_swap_chain(fan::window_t* window, VkResult err) {
  if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR || SwapChainRebuild) {
    int fb_width, fb_height;
    glfwGetFramebufferSize(*window, &fb_width, &fb_height);
    if (fb_width > 0 && fb_height > 0 && 
      #if defined(fan_gui)
      (
#endif
      SwapChainRebuild 
      #if defined(fan_gui)
      || MainWindowData.Width != fb_width || 
      MainWindowData.Height != fb_height)
      #endif
      )
    {
      
      #if defined(fan_gui)
      ImGui_ImplVulkan_SetMinImageCount(MinImageCount);
      ImGui_ImplVulkanH_CreateOrResizeWindow(instance, physical_device, device, &MainWindowData, queue_family, /*g_Allocator*/nullptr, fb_width, fb_height, MinImageCount);
      current_frame = MainWindowData.FrameIndex = 0;
#endif
      SwapChainRebuild = false;
      #if defined(fan_gui)
      swap_chain = MainWindowData.Swapchain;
#endif
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
  #if defined(fan_gui)
  MainWindowData.Swapchain = swap_chain;
#endif
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
      physical_device = device;
      break;
    }
  }

  if (physical_device == VK_NULL_HANDLE) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
}

void fan::vulkan::context_t::create_logical_device() {
  queue_family_indices_t indices = find_queue_families(physical_device);

  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<uint32_t> uniqueQueueFamilies = {
    indices.graphics_family.value(),
#if defined(loco_window)
    indices.present_family.value()
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

  VkPhysicalDeviceProperties2 deviceProperties{};
  deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  vkGetPhysicalDeviceProperties2(physical_device, &deviceProperties);

  VkPhysicalDeviceFeatures deviceFeatures{};
  deviceFeatures.samplerAnisotropy = VK_TRUE;

  //deviceFeatures.vertexPipelineStoresAndAtomics = VK_TRUE;

  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.queueCreateInfoCount = queueCreateInfos.size();
  createInfo.pQueueCreateInfos = queueCreateInfos.data();

  if (deviceProperties.properties.apiVersion >= VK_API_VERSION_1_2) {
    // Use Vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.runtimeDescriptorArray = VK_TRUE;
    vulkan12Features.descriptorIndexing = VK_TRUE;
    vulkan12Features.descriptorBindingVariableDescriptorCount = VK_TRUE;
    vulkan12Features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    
    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.features = deviceFeatures;
    deviceFeatures2.pNext = &vulkan12Features;
    
    createInfo.pNext = &deviceFeatures2;
    createInfo.pEnabledFeatures = nullptr;
  } else {
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT indexingFeatures{};
    indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
    indexingFeatures.runtimeDescriptorArray = VK_TRUE;
    indexingFeatures.descriptorBindingVariableDescriptorCount = VK_TRUE;
    indexingFeatures.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    
    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.features = deviceFeatures;
    deviceFeatures2.pNext = &indexingFeatures;
    
    createInfo.pNext = &deviceFeatures2;
    createInfo.pEnabledFeatures = nullptr;
  }
  
  createInfo.enabledExtensionCount = deviceExtensions.size();
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();

#if fan_debug >= fan_debug_high
  if (supports_validation_layers) {
    createInfo.enabledLayerCount = validationLayers.size();
    createInfo.ppEnabledLayerNames = validationLayers.data();
  }
#endif

  if (vkCreateDevice(physical_device, &createInfo, nullptr, &device) != VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }

  vkGetDeviceQueue(device, indices.graphics_family.value(), 0, &graphics_queue);
#if defined(loco_window)
  vkGetDeviceQueue(device, indices.present_family.value(), 0, &present_queue);
#endif
}

void fan::vulkan::context_t::create_swap_chain(const fan::vec2ui& framebuffer_size) {
  swap_chain_support_details_t swapChainSupport = query_swap_chain_support(physical_device);

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

  queue_family_indices_t indices = find_queue_families(physical_device);
  queue_family = indices.graphics_family.value();
  uint32_t queueFamilyIndices[] = { indices.graphics_family.value(), indices.present_family.value() };

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

  fan::vulkan::vai_t::properties_t vp;
  vp.format = swap_chain_image_format;
  vp.swap_chain_size = swap_chain_size;
  vp.usage_flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  vp.aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT;

  // Resize vectors to hold image views for each swap chain image
  mainColorImageViews.resize(swap_chain_image_views.size());
  postProcessedColorImageViews.resize(swap_chain_image_views.size());
  depthImageViews.resize(swap_chain_image_views.size());
  downscaleImageViews1.resize(swap_chain_image_views.size());
  upscaleImageViews1.resize(swap_chain_image_views.size());

  for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
    mainColorImageViews[i].open(*this, vp);
    mainColorImageViews[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

    postProcessedColorImageViews[i].open(*this, vp);
    postProcessedColorImageViews[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

    fan::vulkan::vai_t::properties_t depth_vp = vp;
    depth_vp.aspect_flags = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth_vp.format = find_depth_format();
    depth_vp.usage_flags = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depthImageViews[i].open(*this, depth_vp);
    depthImageViews[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, depth_vp.aspect_flags);

    downscaleImageViews1[i].open(*this, vp);
    downscaleImageViews1[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);

    upscaleImageViews1[i].open(*this, vp);
    upscaleImageViews1[i].transition_image_layout(*this, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
  }

  for (uint32_t i = 0; i < swap_chain_images.size(); i++) {
    swap_chain_image_views[i] = create_image_view(swap_chain_images[i], swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT);
  }
}

void fan::vulkan::context_t::create_render_pass() {
  //--------------attachment description--------------

  VkAttachmentDescription mainColorAttachment{};
  mainColorAttachment.format = swap_chain_image_format;
  mainColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  mainColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  mainColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  mainColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  mainColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  mainColorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  mainColorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // For post-process input

  VkAttachmentDescription postProcessedColorAttachment{};
  postProcessedColorAttachment.format = swap_chain_image_format;
  postProcessedColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  postProcessedColorAttachment.loadOp = shapes_top ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  postProcessedColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  postProcessedColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  postProcessedColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  postProcessedColorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  postProcessedColorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentDescription depthAttachment{};
  depthAttachment.format = find_depth_format();
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  //--------------attachment description--------------

  VkAttachmentReference mainSceneColorRef{};
  mainSceneColorRef.attachment = 0;
  mainSceneColorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference postProcessInputRef{};
  postProcessInputRef.attachment = 0;
  postProcessInputRef.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkAttachmentReference postProcessOutputRef{};
  postProcessOutputRef.attachment = 1;
  postProcessOutputRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthRef{};
  depthRef.attachment = 2;
  depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription mainSceneSubpass{};
  mainSceneSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  mainSceneSubpass.colorAttachmentCount = 1;
  mainSceneSubpass.pColorAttachments = &mainSceneColorRef;
  mainSceneSubpass.pDepthStencilAttachment = &depthRef;

  VkSubpassDescription postProcessSubpass{};
  postProcessSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  postProcessSubpass.inputAttachmentCount = 1;
  postProcessSubpass.pInputAttachments = &postProcessInputRef;
  postProcessSubpass.colorAttachmentCount = 1;
  postProcessSubpass.pColorAttachments = &postProcessOutputRef;

  VkSubpassDependency extToMainDep{};
  extToMainDep.srcSubpass = VK_SUBPASS_EXTERNAL;
  extToMainDep.dstSubpass = 0;
  extToMainDep.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  extToMainDep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  extToMainDep.srcAccessMask = 0;
  extToMainDep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkSubpassDependency mainToPostDep{};
  mainToPostDep.srcSubpass = 0;
  mainToPostDep.dstSubpass = 1;
  mainToPostDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  mainToPostDep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  mainToPostDep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  mainToPostDep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  VkSubpassDependency postToExtDep{};
  postToExtDep.srcSubpass = 1;
  postToExtDep.dstSubpass = VK_SUBPASS_EXTERNAL;
  postToExtDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  postToExtDep.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  postToExtDep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  postToExtDep.dstAccessMask = 0;

  VkAttachmentDescription attachments[] = {
    mainColorAttachment,
    postProcessedColorAttachment,
    depthAttachment
  };

  VkSubpassDescription subpasses[] = {
    mainSceneSubpass,
    postProcessSubpass
  };

  VkSubpassDependency dependencies[] = {
    extToMainDep,
    mainToPostDep,
    postToExtDep
  };

  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = std::size(attachments);
  renderPassInfo.pAttachments = attachments;
  renderPassInfo.subpassCount = std::size(subpasses);
  renderPassInfo.pSubpasses = subpasses;
  renderPassInfo.dependencyCount = std::size(dependencies);
  renderPassInfo.pDependencies = dependencies;

  if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &render_pass) != VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass");
  }
}


void fan::vulkan::context_t::create_framebuffers() {
  swap_chain_framebuffers.resize(swap_chain_image_views.size());

  for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
    VkImageView attachments[] = {
      mainColorImageViews[i].image_view,
      swap_chain_image_views[i],
      depthImageViews[i].image_view,
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
  queue_family_indices_t queueFamilyIndices = find_queue_families(physical_device);

  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = queueFamilyIndices.graphics_family.value();

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
  depthStencil.depthTestEnable = VK_TRUE;//p.enable_depth_test;
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
  pipelineLayoutInfo.pushConstantRangeCount = p.push_constants_size == 0 ? 0 : 1;

  if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &m_layout) != VK_SUCCESS) {
    fan::throw_error("failed to create pipeline layout!");
  }

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shader_get(context, p.shader).shader_stages;
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

  shader_nr = p.shader;

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
    vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

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

#if defined(fan_gui)

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

void fan::vulkan::context_t::ImGuiFrameRender(VkResult next_image_khr_err, fan::color clear_color) {
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

  vkCmdBeginRenderPass(command_buffers[current_frame], &info, VK_SUBPASS_CONTENTS_INLINE);
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffers[current_frame]);

  vkCmdEndRenderPass(command_buffers[current_frame]);
}

#endif

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

  std::vector<std::string> extension_str(available_extensions.size());

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
  if (pCallbackData->pMessageIdName && std::string(pCallbackData->pMessageIdName) == "Loader Message") {
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
  fan::vulkan::image_create(
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
  if (image_view != 0) {
    vkDestroyImageView(context.device, image_view, nullptr);
    image_view = 0;
  }
  if (image != 0) {
    vkDestroyImage(context.device, image, nullptr);
    image = 0;
  }
  if (memory != 0) {
    vkFreeMemory(context.device, memory, nullptr);
    memory = 0;
  }
}

void fan::vulkan::vai_t::transition_image_layout(auto& context, VkImageLayout newLayout, VkImageAspectFlags aspectFlags) {
  if (old_layout == newLayout) {
        return;
    }

  VkCommandBuffer commandBuffer = context.begin_single_time_commands();

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = aspectFlags;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags sourceStage;
  VkPipelineStageFlags destinationStage;

  if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  }
  else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  }
  else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
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

  context.end_single_time_commands(commandBuffer);

  old_layout = newLayout;
}



void fan::vulkan::context_t::descriptor_pool_t::open(fan::vulkan::context_t& context) {
  VkDescriptorPoolSize pool_sizes[] =
  {
    
    { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 
    #if defined(fan_gui)
    IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE +
#endif
    5 * max_frames_in_flight},
  };
  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 0;
  for (VkDescriptorPoolSize& pool_size : pool_sizes)
    pool_info.maxSets += max_frames_in_flight * pool_size.descriptorCount;
  pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;
  ;
  fan::vulkan::validate(vkCreateDescriptorPool(context.device, &pool_info, nullptr, &m_descriptor_pool));
}

void fan::vulkan::context_t::descriptor_pool_t::close(fan::vulkan::context_t& context) {
  vkDestroyDescriptorPool(context.device, m_descriptor_pool, nullptr);
}

uint32_t fan::vulkan::context_t::find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physical_device, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	fan::throw_error("failed to find suitable memory type!");
  return {};
}

void fan::vulkan::context_t::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = usage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	fan::vulkan::validate(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = find_memory_type(memRequirements.memoryTypeBits, properties);

	fan::vulkan::validate(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory));
	fan::vulkan::validate(vkBindBufferMemory(device, buffer, bufferMemory, 0));
}

VkFormat global_to_vulkan_format(uintptr_t format) {
  if (format == image_format::b8g8r8a8_unorm) return VK_FORMAT_B8G8R8A8_UNORM;
  if (format == image_format::r8b8g8a8_unorm) return VK_FORMAT_R8G8B8A8_UNORM;
  if (format == image_format::r8_unorm) return VK_FORMAT_R8_UNORM;
  if (format == image_format::r8_uint) return VK_FORMAT_R8_UINT;
  if (format == image_format::r8g8b8a8_srgb) return VK_FORMAT_R8G8B8A8_SRGB;
  if (format == image_format::rgba_unorm) return VK_FORMAT_R8G8B8A8_UNORM;
#if fan_debug >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return VK_FORMAT_R8G8B8A8_UNORM;
}

VkSamplerAddressMode global_to_vulkan_address_mode(uintptr_t mode) {
  if (mode == image_sampler_address_mode::repeat) return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  if (mode == image_sampler_address_mode::mirrored_repeat) return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  if (mode == image_sampler_address_mode::clamp_to_edge) return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  if (mode == image_sampler_address_mode::clamp_to_border) return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  if (mode == image_sampler_address_mode::mirrored_clamp_to_edge) return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
#if fan_debug >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkFilter global_to_vulkan_filter(uintptr_t filter) {
  if (filter == image_filter::nearest) return VK_FILTER_NEAREST;
  if (filter == image_filter::linear) return VK_FILTER_LINEAR;
#if fan_debug >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return VK_FILTER_NEAREST;
}

fan::vulkan::context_t::image_load_properties_t image_global_to_vulkan(const fan::graphics::image_load_properties_t& p) {
  return {
    .visual_output = global_to_vulkan_address_mode(p.visual_output),
    .format = global_to_vulkan_format(p.format),
    .min_filter = global_to_vulkan_filter(p.min_filter),
    .mag_filter = global_to_vulkan_filter(p.mag_filter),
  };
}

fan::graphics::context_functions_t fan::graphics::get_vk_context_functions() {
	fan::graphics::context_functions_t cf;
  cf.shader_create = [](void* context) { 
    return shader_create(*(fan::vulkan::context_t*)context);
  }; 
  cf.shader_get = [](void* context, shader_nr_t nr) { 
    return (void*)&shader_get(*(fan::vulkan::context_t*)context, nr);
  }; 
  cf.shader_erase = [](void* context, shader_nr_t nr) { 
    shader_erase(*(fan::vulkan::context_t*)context,nr); 
  }; 
  cf.shader_use = [](void* context, shader_nr_t nr) { 
    shader_use(*(fan::vulkan::context_t*)context,nr); 
  }; 
  cf.shader_set_vertex = [](void* context, shader_nr_t nr, const std::string& vertex_code) { 
    shader_set_vertex(*(fan::vulkan::context_t*)context, nr, vertex_code); 
  }; 
  cf.shader_set_fragment = [](void* context, shader_nr_t nr, const std::string& fragment_code) { 
    shader_set_fragment(*(fan::vulkan::context_t*)context,nr, fragment_code); 
  }; 
  cf.shader_compile = [](void* context, shader_nr_t nr) { 
    return shader_compile(*(fan::vulkan::context_t*)context,nr); 
  }; 
    /*image*/
  cf.image_create = [](void* context) {
    return image_create(*(fan::vulkan::context_t*)context);
  }; 
  cf.image_get_handle = [](void* context, image_nr_t nr) { 
    return (uint64_t)image_get_handle(*(fan::vulkan::context_t*)context,nr); 
  }; 
  cf.image_get = [](void* context, image_nr_t nr) {
    return (void*)&image_get(*(fan::vulkan::context_t*)context, nr);
  }; 
  cf.image_erase = [](void* context, image_nr_t nr) { 
    ::image_erase(*(fan::vulkan::context_t*)context,nr); 
  }; 
  cf.image_bind = [](void* context, image_nr_t nr) { 
    ::image_bind(*(fan::vulkan::context_t*)context,nr); 
  }; 
  cf.image_unbind = [](void* context, image_nr_t nr) { 
    ::image_unbind(*(fan::vulkan::context_t*)context,nr); 
  }; 
  cf.image_get_settings = [](void* context, fan::graphics::image_nr_t nr) -> fan::graphics::image_load_properties_t& {
    return ::image_get_settings(*(fan::vulkan::context_t*)context, nr);
  };
  cf.image_set_settings = [](void* context, fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings) { 
    ::image_bind(*(fan::vulkan::context_t*)context, nr);
    ::image_set_settings(*(fan::vulkan::context_t*)context, image_global_to_vulkan(settings));
  }; 
  cf.image_load_info = [](void* context, const fan::image::info_t& image_info) { 
    return ::image_load(*(fan::vulkan::context_t*)context, image_info);
  }; 
  cf.image_load_info_props = [](void* context, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) { 
    return ::image_load(*(fan::vulkan::context_t*)context, image_info, image_global_to_vulkan(p));
  }; 
  cf.image_load_path = [](void* context, const std::string& path) { 
    return ::image_load(*(fan::vulkan::context_t*)context, path);
  }; 
  cf.image_load_path_props = [](void* context, const std::string& path, const fan::graphics::image_load_properties_t& p) { 
    return ::image_load(*(fan::vulkan::context_t*)context, path, image_global_to_vulkan(p));
  }; 
  cf.image_load_colors = [](void* context, fan::color* colors, const fan::vec2ui& size_) { 
    return ::image_load(*(fan::vulkan::context_t*)context, colors, size_);
  }; 
  cf.image_load_colors_props = [](void* context, fan::color* colors, const fan::vec2ui& size_, const fan::graphics::image_load_properties_t& p) { 
    return ::image_load(*(fan::vulkan::context_t*)context, colors, size_, image_global_to_vulkan(p));
  }; 
  cf.image_unload = [](void* context, image_nr_t nr) { 
    ::image_unload(*(fan::vulkan::context_t*)context, nr); 
  }; 
  cf.create_missing_texture = [](void* context) { 
    return ::create_missing_texture(*(fan::vulkan::context_t*)context);
  }; 
  cf.create_transparent_texture = [](void* context) { 
    return ::create_transparent_texture(*(fan::vulkan::context_t*)context);
  }; 
  cf.image_reload_image_info = [](void* context, image_nr_t nr, const fan::image::info_t& image_info) { 
    return ::image_reload(*(fan::vulkan::context_t*)context, nr, image_info); 
  }; 
  cf.image_reload_image_info_props = [](void* context, image_nr_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) { 
    return ::image_reload(*(fan::vulkan::context_t*)context, nr, image_info, image_global_to_vulkan(p)); 
  }; 
  cf.image_reload_path = [](void* context, image_nr_t nr, const std::string& path) { 
    return ::image_reload(*(fan::vulkan::context_t*)context, nr, path); 
  }; 
  cf.image_reload_path_props = [](void* context, image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p) { 
    return ::image_reload(*(fan::vulkan::context_t*)context, nr, path, image_global_to_vulkan(p)); 
  };
  cf.image_create_color = [](void* context, const fan::color& color) { 
    return ::image_create(*(fan::vulkan::context_t*)context, color);
  }; 
  cf.image_create_color_props = [](void* context, const fan::color& color, const fan::graphics::image_load_properties_t& p) { 
    return ::image_create(*(fan::vulkan::context_t*)context, color, image_global_to_vulkan(p));
  };
  /*camera*/
  cf.camera_create = [](void* context) {
    return camera_create(*(fan::vulkan::context_t*)context);
  };
  cf.camera_get = [](void* context, fan::graphics::camera_nr_t nr) -> decltype(auto) {
    return camera_get(*(fan::vulkan::context_t*)context, nr);
  };
  cf.camera_erase = [](void* context, camera_nr_t nr) { 
    camera_erase(*(fan::vulkan::context_t*)context, nr); 
  };
  cf.camera_create = [](void* context, const fan::vec2& x, const fan::vec2& y) {
    return camera_create(*(fan::vulkan::context_t*)context, x, y);
  };
  cf.camera_get_position = [](void* context, camera_nr_t nr) { 
    return camera_get_position(*(fan::vulkan::context_t*)context, nr); 
  };
  cf.camera_set_position = [](void* context, camera_nr_t nr, const fan::vec3& cp) { 
    camera_set_position(*(fan::vulkan::context_t*)context, nr, cp); 
  };
  cf.camera_get_size = [](void* context, camera_nr_t nr) { 
    return camera_get_size(*(fan::vulkan::context_t*)context, nr); 
  };
  cf.camera_set_ortho = [](void* context, camera_nr_t nr, fan::vec2 x, fan::vec2 y) { 
    camera_set_ortho(*(fan::vulkan::context_t*)context, nr, x, y); 
  };
  cf.camera_set_perspective = [](void* context, camera_nr_t nr, f32_t fov, const fan::vec2& window_size) { 
    camera_set_perspective(*(fan::vulkan::context_t*)context, nr, fov, window_size); 
  };
  cf.camera_rotate = [](void* context, camera_nr_t nr, const fan::vec2& offset) { 
    camera_rotate(*(fan::vulkan::context_t*)context, nr, offset); 
  };
  /*viewport*/
  cf.viewport_create = [](void* context) {
    return viewport_create(*(fan::vulkan::context_t*)context);
  };
  cf.viewport_get = [](void* context, viewport_nr_t nr) -> fan::graphics::context_viewport_t&{ 
    return viewport_get(*(fan::vulkan::context_t*)context, nr);
  };
  cf.viewport_erase = [](void* context, viewport_nr_t nr) { 
    viewport_erase(*(fan::vulkan::context_t*)context, nr); 
  };
  cf.viewport_get_position = [](void* context, viewport_nr_t nr) { 
    return viewport_get_position(*(fan::vulkan::context_t*)context, nr); 
  };
  cf.viewport_get_size = [](void* context, viewport_nr_t nr) { 
    return viewport_get_size(*(fan::vulkan::context_t*)context, nr); 
  };
  cf.viewport_set = [](void* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) { 
    viewport_set(*(fan::vulkan::context_t*)context, viewport_position_, viewport_size_, window_size); 
  };
  cf.viewport_set_nr = [](void* context, viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) { 
    viewport_set(*(fan::vulkan::context_t*)context, nr, viewport_position_, viewport_size_, window_size); 
  };
  cf.viewport_zero = [](void* context, viewport_nr_t nr) { 
    viewport_zero(*(fan::vulkan::context_t*)context, nr); 
  };
  cf.viewport_inside = [](void* context, viewport_nr_t nr, const fan::vec2& position) { 
    return viewport_inside(*(fan::vulkan::context_t*)context, nr, position); 
  };
  cf.viewport_inside_wir = [](void* context, viewport_nr_t nr, const fan::vec2& position) { 
    return viewport_inside_wir(*(fan::vulkan::context_t*)context, nr, position); 
  };
  return cf;
}

uint32_t fan::vulkan::core::get_draw_mode(uint8_t draw_mode) {
  switch (draw_mode) {
  case primitive_topology_t::points:
    return fan::vulkan::context_t::primitive_topology_t::points;
  case primitive_topology_t::lines:
    return fan::vulkan::context_t::primitive_topology_t::lines;
  case primitive_topology_t::line_strip:
    return fan::vulkan::context_t::primitive_topology_t::line_strip;
  case primitive_topology_t::triangles:
    return fan::vulkan::context_t::primitive_topology_t::triangles;
  case primitive_topology_t::triangle_strip:
    return fan::vulkan::context_t::primitive_topology_t::triangle_strip;
  case primitive_topology_t::triangle_fan:
    return fan::vulkan::context_t::primitive_topology_t::triangle_fan;
  case primitive_topology_t::lines_with_adjacency:
    return fan::vulkan::context_t::primitive_topology_t::lines_with_adjacency;
  case primitive_topology_t::line_strip_with_adjacency:
    return fan::vulkan::context_t::primitive_topology_t::line_strip_with_adjacency;
  case primitive_topology_t::triangles_with_adjacency:
    return fan::vulkan::context_t::primitive_topology_t::triangles_with_adjacency;
  case primitive_topology_t::triangle_strip_with_adjacency:
    return fan::vulkan::context_t::primitive_topology_t::triangle_strip_with_adjacency;
  default:
    fan::throw_error("invalid draw mode");
    return -1;
  }
}
#endif