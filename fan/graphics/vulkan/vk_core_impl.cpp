module;

#if defined(FAN_VULKAN)
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
#include <shaderc/shaderc.hpp>
#if defined(fan_platform_windows)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define GLFW_NATIVE_INCLUDE_NONE
#endif
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#endif

module fan.graphics.vulkan.core;

import std;

#if defined(FAN_VULKAN)

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

#if defined(fan_compiler_msvc)
#pragma comment(lib, "vulkan-1.lib")
#pragma comment(lib, "shaderc_combined_mt.lib")
#endif

#define ENABLE_RAYTRACING_DEPENDENCIES


const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
#if defined(ENABLE_RAYTRACING_DEPENDENCIES)
  // RT
  VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
  VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
  VK_KHR_SPIRV_1_4_EXTENSION_NAME,
  VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
  VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
  VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
#endif
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

VkFormat fan::graphics::format_converter::global_to_vulkan_format(std::uintptr_t format) {
  if (format == image_format_e::bgra) return VK_FORMAT_B8G8R8A8_UNORM;
  if (format == image_format_e::rgba) return VK_FORMAT_R8G8B8A8_UNORM;
  if (format == image_format_e::r8_unorm) return VK_FORMAT_R8_UNORM;
  if (format == image_format_e::r8_uint) return VK_FORMAT_R8_UINT;
  if (format == image_format_e::r8g8b8a8_srgb) return VK_FORMAT_R8G8B8A8_SRGB;
  if (format == image_format_e::rgba_unorm) return VK_FORMAT_R8G8B8A8_UNORM;
#if FAN_DEBUG >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return VK_FORMAT_R8G8B8A8_UNORM;
}

VkSamplerAddressMode fan::graphics::format_converter::global_to_vulkan_address_mode(std::uintptr_t mode) {
  if (mode == image_sampler_address_mode_e::repeat) return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  if (mode == image_sampler_address_mode_e::mirrored_repeat) return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  if (mode == image_sampler_address_mode_e::clamp_to_edge) return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  if (mode == image_sampler_address_mode_e::clamp_to_border) return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  if (mode == image_sampler_address_mode_e::mirrored_clamp_to_edge) return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
#if FAN_DEBUG >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkFilter fan::graphics::format_converter::global_to_vulkan_filter(std::uintptr_t filter) {
  if (filter == image_filter_e::nearest) return VK_FILTER_NEAREST;
  if (filter == image_filter_e::linear) return VK_FILTER_LINEAR;
#if FAN_DEBUG >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return VK_FILTER_NEAREST;
}

std::uint32_t fan::graphics::format_converter::vulkan_to_global_format(VkFormat format) {
  if (format == VK_FORMAT_B8G8R8A8_UNORM) return fan::graphics::image_format_e::bgra;
  if (format == VK_FORMAT_R8G8B8A8_UNORM) return fan::graphics::image_format_e::rgba;
  if (format == VK_FORMAT_R8_UNORM) return fan::graphics::image_format_e::r8_unorm;
  if (format == VK_FORMAT_R8_UINT) return fan::graphics::image_format_e::r8_uint;
  if (format == VK_FORMAT_R8G8B8A8_SRGB) return fan::graphics::image_format_e::r8g8b8a8_srgb;
#if FAN_DEBUG >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return fan::graphics::image_format_e::rgba_unorm;
}

std::uint32_t fan::graphics::format_converter::vulkan_to_global_address_mode(VkSamplerAddressMode mode) {
  if (mode == VK_SAMPLER_ADDRESS_MODE_REPEAT) return fan::graphics::image_sampler_address_mode_e::repeat;
  if (mode == VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT) return fan::graphics::image_sampler_address_mode_e::mirrored_repeat;
  if (mode == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE) return fan::graphics::image_sampler_address_mode_e::clamp_to_edge;
  if (mode == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) return fan::graphics::image_sampler_address_mode_e::clamp_to_border;
  if (mode == VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE) return fan::graphics::image_sampler_address_mode_e::mirrored_clamp_to_edge;
#if FAN_DEBUG >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return fan::graphics::image_sampler_address_mode_e::repeat;
}

std::uint32_t fan::graphics::format_converter::vulkan_to_global_filter(VkFilter filter) {
  if (filter == VK_FILTER_NEAREST) return fan::graphics::image_filter_e::nearest;
  if (filter == VK_FILTER_LINEAR) return fan::graphics::image_filter_e::linear;
#if FAN_DEBUG >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return fan::graphics::image_filter_e::nearest;
}

void fan::vulkan::validate(VkResult result) {
  if (result != VK_SUCCESS) {
    fan::throw_error("function failed");
  }
}

fan::vulkan::context_t::shader_t& fan::vulkan::context_t::shader_get(fan::graphics::shader_nr_t nr) {
  return *(fan::vulkan::context_t::shader_t*)__fan_internal_shader_list[nr].internal;
}
std::vector<std::uint32_t> fan::vulkan::context_t::compile_file(const std::string& source_name,
  shaderc_shader_kind kind,
  const std::string& source) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

  // Like -DMY_DEFINE=1
  //options.AddMacroDefinition("MY_DEFINE", "1");
#if FAN_DEBUG > 1
  options.SetOptimizationLevel(shaderc_optimization_level_zero);
#else
  options.SetOptimizationLevel(shaderc_optimization_level_performance);
#endif

  shaderc::SpvCompilationResult module =
    compiler.CompileGlslToSpv(source.c_str(), kind, source_name.c_str(), options);

  if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
    fan::throw_error(module.GetErrorMessage().c_str());
  }

  return {module.cbegin(), module.cend()};
}
fan::graphics::shader_nr_t fan::vulkan::context_t::shader_create() {
  fan::graphics::shader_nr_t nr = __fan_internal_shader_list.NewNode();
  __fan_internal_shader_list[nr].internal = new fan::vulkan::context_t::shader_t;
  auto& shader = shader_get(nr);
  shader.projection_view_block = new std::remove_pointer_t<decltype(shader.projection_view_block)>;
  //TODO
  shader.projection_view_block->open(*this);
  for (std::uint32_t i = 0; i < fan::vulkan::max_camera; ++i) {
    shader.projection_view_block->push_ram_instance(*this, {});
  }
  return nr;
}
void fan::vulkan::context_t::shader_erase(fan::graphics::shader_nr_t nr, int recycle) {
  auto& shader = shader_get(nr);
  if (shader.shader_stages[0].module) {
    vkDestroyShaderModule(device, shader.shader_stages[0].module, nullptr);
  }
  if (shader.shader_stages[1].module) {
    vkDestroyShaderModule(device, shader.shader_stages[1].module, nullptr);
  }
  if (shader.shader_stages[2].module) {
    vkDestroyShaderModule(device, shader.shader_stages[2].module, nullptr);
  }
  //TODO
  shader.projection_view_block->close(*this);
  delete shader.projection_view_block;
  delete static_cast<fan::vulkan::context_t::shader_t*>(__fan_internal_shader_list[nr].internal);
  if (recycle) {
    __fan_internal_shader_list.Recycle(nr);
  }
}
void fan::vulkan::context_t::shader_use(fan::graphics::shader_nr_t nr) {
  // TODO - required?
}
VkShaderModule fan::vulkan::context_t::create_shader_module(const std::vector<std::uint32_t>& code) {
  VkShaderModuleCreateInfo createInfo {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size() * sizeof(typename std::remove_reference_t<decltype(code)>::value_type);
  createInfo.pCode = code.data();

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    fan::throw_error("failed to create shader module!");
  }

  return shaderModule;
}
void fan::vulkan::context_t::shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code) {
  __fan_internal_shader_list[nr].path_vertex = file_path;
  __fan_internal_shader_list[nr].svertex = vertex_code;
  // fan::print_impl(
  //   "processed vertex shader:", path, "resulted in:",
  // preprocess_shader(shader_name.c_str(), shaderc_glsl_vertex_shader, shader_code);
  // );
}
void fan::vulkan::context_t::shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code) {
  shader_set_vertex(nr, {}, vertex_code);
}
void fan::vulkan::context_t::shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code) {
  auto& shader = shader_get(nr);
  __fan_internal_shader_list[nr].path_fragment = file_path;
  __fan_internal_shader_list[nr].sfragment = fragment_code;
  //fan::print_impl(
    // "processed vertex shader:", path, "resulted in:",
  //preprocess_shader(shader_name.c_str(), shaderc_glsl_fragment_shader, shader_code);
  //);
}
void fan::vulkan::context_t::shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code) {
  shader_set_fragment(nr, {}, fragment_code);
}
void fan::vulkan::context_t::shader_set_compute(
  fan::graphics::shader_nr_t nr,
  const std::string_view file_path,
  const std::string& compute_code
) {
  __fan_internal_shader_list[nr].path_compute = file_path;
  __fan_internal_shader_list[nr].scompute = compute_code;
}
void fan::vulkan::context_t::shader_set_camera(fan::graphics::shader_nr_t nr, fan::graphics::camera_nr_t camera_nr) {
  auto& shader = shader_get(nr);
  auto& camera = camera_get(camera_nr);

  std::uint32_t camera_index = camera_nr.gint();

#if FAN_DEBUG >= fan_debug_medium
  if (camera_index >= fan::vulkan::max_camera) {
    fan::throw_error("vulkan camera index exceeds max_camera");
  }
#endif

  shader.projection_view_block->edit_instance(
    *this,
    camera_index,
    &fan::vulkan::view_projection_t::projection,
    camera.projection
  );

  shader.projection_view_block->edit_instance(
    *this,
    camera_index,
    &fan::vulkan::view_projection_t::view,
    camera.view
  );
}
void fan::vulkan::context_t::shader_dispatch_compute(
  fan::graphics::shader_nr_t nr,
  std::uint32_t x,
  std::uint32_t y,
  std::uint32_t z
) {
  fan::throw_error("vulkan compute dispatch is not implemented");
}
void fan::vulkan::context_t::parse_uniforms(std::string& shaderData, std::unordered_map<std::string, std::string>& uniform_type_table) {
  std::size_t pos = 0;

  while ((pos = shaderData.find("uniform", pos)) != std::string::npos) {
    std::size_t endLine = shaderData.find(';', pos);
    if (endLine == std::string::npos) break;

    std::string line = shaderData.substr(pos, endLine - pos + 1);

    line = line.substr(7);

    std::size_t start = line.find_first_not_of(" \t");
    if (start == std::string::npos) {
      pos = endLine + 1;
      continue;
    }
    line = line.substr(start);

    std::size_t space1 = line.find_first_of(" \t");
    if (space1 == std::string::npos) {
      pos = endLine + 1;
      continue;
    }

    std::string type = line.substr(0, space1);
    line = line.substr(space1);
    line = line.substr(line.find_first_not_of(" \t"));

    std::size_t varEnd = line.find_first_of("=;");
    std::string name = line.substr(0, varEnd);

    name.erase(0, name.find_first_not_of(" \t"));
    name.erase(name.find_last_not_of(" \t") + 1);

    uniform_type_table[name] = type;

    pos = endLine + 1;
  }
}
bool fan::vulkan::context_t::shader_compile(fan::graphics::shader_nr_t nr) {
  auto& shader = shader_get(nr);
  bool has_vertex = !__fan_internal_shader_list[nr].svertex.empty();
  bool has_fragment = !__fan_internal_shader_list[nr].sfragment.empty();
  bool has_compute = !__fan_internal_shader_list[nr].scompute.empty();

  if (has_compute && (has_vertex || has_fragment)) {
    fan::print_impl("compute shader cannot be linked with graphics shaders");
    return false;
  }

  if (has_vertex) {
    auto spirv = compile_file(std::string(__fan_internal_shader_list[nr].path_vertex.c_str()), shaderc_glsl_vertex_shader, __fan_internal_shader_list[nr].svertex);

    auto module_vertex = create_shader_module(spirv);

    VkPipelineShaderStageCreateInfo vert {};
    vert.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert.module = module_vertex;
    vert.pName = "main";

    shader.shader_stages[0] = vert;
  }
  if (has_fragment) {
    auto spirv = compile_file(std::string(__fan_internal_shader_list[nr].path_fragment.c_str()), shaderc_glsl_fragment_shader, __fan_internal_shader_list[nr].sfragment);

    auto module_fragment = create_shader_module(spirv);

    VkPipelineShaderStageCreateInfo frag {};
    frag.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag.module = module_fragment;
    frag.pName = "main";

    shader.shader_stages[1] = frag;
  }
  if (has_compute) {
    auto spirv = compile_file(std::string(__fan_internal_shader_list[nr].path_compute.c_str()), shaderc_glsl_compute_shader, __fan_internal_shader_list[nr].scompute);

    auto module_compute = create_shader_module(spirv);

    VkPipelineShaderStageCreateInfo compute {};
    compute.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute.module = module_compute;
    compute.pName = "main";

    shader.shader_stages[0] = compute;
  }

  std::string vertexData = __fan_internal_shader_list[nr].svertex;
  parse_uniforms(vertexData, __fan_internal_shader_list[nr].uniform_type_table);

  std::string fragmentData = __fan_internal_shader_list[nr].sfragment;
  parse_uniforms(fragmentData, __fan_internal_shader_list[nr].uniform_type_table);

  std::string computeData = __fan_internal_shader_list[nr].scompute;
  parse_uniforms(computeData, __fan_internal_shader_list[nr].uniform_type_table);

  return true;
}
void fan::vulkan::context_t::transition_image_layout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
  VkCommandBuffer command_buffer = begin_single_time_commands();

  VkImageMemoryBarrier barrier {};
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
    fan::throw_error("unsupported layout transition!");
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
void fan::vulkan::context_t::copy_buffer_to_image(
  VkBuffer buffer,
  VkImage image,
  VkFormat format,
  const fan::vec2ui& size,
  const fan::vec2ui& stride) {
  VkCommandBuffer command_buffer = begin_single_time_commands();

  VkBufferImageCopy region {};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;      // tightly packed
  region.bufferImageHeight = 0;    // tightly packed

  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;

  region.imageOffset = {0, 0, 0};
  region.imageExtent = {size.x, size.y, 1};

  vkCmdCopyBufferToImage(
    command_buffer,
    buffer,
    image,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    1,
    &region
  );

  end_single_time_commands(command_buffer);
}
void fan::vulkan::context_t::create_texture_sampler(VkSampler& sampler, const image_load_properties_t& lp) {
  VkPhysicalDeviceProperties properties {};
  vkGetPhysicalDeviceProperties(physical_device, &properties);

  VkSamplerCreateInfo samplerInfo {};
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
    fan::throw_error("failed to create texture sampler!");
  }
}
VkFormat fan::vulkan::context_t::get_format_from_channels(int channels) {
  switch (channels) {
    case 1: return VK_FORMAT_R8_UNORM;
    case 2: return VK_FORMAT_R8G8_UNORM;
    case 3: return VK_FORMAT_R8G8B8_UNORM;
    case 4: return VK_FORMAT_R8G8B8A8_UNORM;
    default: return VK_FORMAT_R8G8B8A8_UNORM;
  }
}
// for draw

fan::graphics::image_nr_t fan::vulkan::context_t::image_create() {
  fan::graphics::image_nr_t nr = __fan_internal_image_list.NewNode();
  __fan_internal_image_list[nr].internal = new fan::vulkan::context_t::image_t;
  return nr;
}
std::uint64_t fan::vulkan::context_t::image_get_handle(fan::graphics::image_nr_t nr) {
  fan::throw_error("invalid call");
  return 0;
}
fan::vulkan::context_t::image_t& fan::vulkan::context_t::image_get(fan::graphics::image_nr_t nr) {
  return *(fan::vulkan::context_t::image_t*)__fan_internal_image_list[nr].internal;
}
void fan::vulkan::context_t::image_erase(fan::graphics::image_nr_t nr, int recycle) {
  auto& node = __fan_internal_image_list[nr];
  auto& img = image_get(nr);

  if (img.sampler != VK_NULL_HANDLE) {
    vkDestroySampler(device, img.sampler, nullptr);
    img.sampler = VK_NULL_HANDLE;
  }
  if (img.staging_buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device, img.staging_buffer, nullptr);
    img.staging_buffer = VK_NULL_HANDLE;
  }
  if (img.staging_buffer_memory != VK_NULL_HANDLE) {
    vkFreeMemory(device, img.staging_buffer_memory, nullptr);
    img.staging_buffer_memory = VK_NULL_HANDLE;
  }
  if (img.image_index != VK_NULL_HANDLE) {
    vkDestroyImage(device, img.image_index, nullptr);
    img.image_index = VK_NULL_HANDLE;
  }
  if (img.image_view != VK_NULL_HANDLE) {
    vkDestroyImageView(device, img.image_view, nullptr);
    img.image_view = VK_NULL_HANDLE;
  }
  if (img.image_memory != VK_NULL_HANDLE) {
    vkFreeMemory(device, img.image_memory, nullptr);
    img.image_memory = VK_NULL_HANDLE;
  }

  delete node.internal;
  node.internal = nullptr;

  if (recycle) {
    __fan_internal_image_list.Recycle(nr);
  }
}
void fan::vulkan::context_t::image_bind(fan::graphics::image_nr_t nr) {

}
void fan::vulkan::context_t::image_bind(fan::graphics::image_nr_t nr, std::uint32_t unit) {
  image_bind(nr);
}
void fan::vulkan::context_t::image_bind(
  fan::graphics::image_t nr,
  uint32_t unit,
  std::uint32_t access,
  std::uint32_t format
) {
  image_bind(nr);
}
void fan::vulkan::context_t::image_unbind(fan::graphics::image_nr_t nr) {

}
fan::graphics::image_load_properties_t& fan::vulkan::context_t::image_get_settings(fan::graphics::image_nr_t nr) {
  return __fan_internal_image_list[nr].image_settings;
}
void fan::vulkan::context_t::image_set_settings(fan::graphics::image_nr_t nr, const fan::vulkan::context_t::image_load_properties_t& p) {
  __fan_internal_image_list[nr].image_settings = fan::graphics::format_converter::image_vulkan_to_global(p);
}
void fan::vulkan::context_t::image_set_settings(const fan::vulkan::context_t::image_load_properties_t& p) {

}
fan::graphics::image_nr_t fan::vulkan::context_t::image_load(const fan::image::info_t& image_info, const fan::vulkan::context_t::image_load_properties_t& p) {
  fan::graphics::image_nr_t nr = image_create();

  fan::vulkan::context_t::image_t& image = image_get(nr);
  auto& image_data = __fan_internal_image_list[nr];
  image_data.size = image_info.size;
  image_data.image_settings = fan::graphics::format_converter::image_vulkan_to_global(p);
  __fan_internal_image_list[nr].image_path = "";

  auto lp = p;
  int src_channels = image_info.channels;
  int format_channels = 0;

  if (lp.format == image_load_properties_defaults::format) {
    if (src_channels <= 0) {
      fan::throw_error("image_load: unknown channel count with default format");
    }
    if (src_channels == 1) {
      lp.format = get_format_from_channels(1);
      format_channels = 1;
    }
    else {
      lp.format = get_format_from_channels(4);
      format_channels = 4;
    }
  }
  else {
    format_channels = fan::graphics::get_channel_amount(
      fan::graphics::format_converter::vulkan_to_global_format(lp.format)
    );
    if (src_channels <= 0) {
      src_channels = format_channels;
    }
  }

  VkDeviceSize image_size_bytes = image_info.size.multiply() * format_channels;

  create_buffer(
    image_size_bytes,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    image.staging_buffer,
    image.staging_buffer_memory
  );

  vkMapMemory(device, image.staging_buffer_memory, 0, image_size_bytes, 0, &image.data);

  const std::uint8_t* src = static_cast<const std::uint8_t*>(image_info.data);
  std::uint8_t* dst = static_cast<std::uint8_t*>(image.data);
  std::uint64_t pixel_count = image_info.size.multiply();

  if (src_channels == format_channels) {
    memcpy(dst, src, image_size_bytes);
  }
  else if (src_channels == 3 && format_channels == 4) {
    for (std::uint64_t i = 0; i < pixel_count; ++i) {
      dst[0] = src[0];
      dst[1] = src[1];
      dst[2] = src[2];
      dst[3] = 255;
      src += 3;
      dst += 4;
    }
  }
  else {
    vkUnmapMemory(device, image.staging_buffer_memory);
    fan::throw_error("image_load: unsupported channel/format combination");
  }

  vkUnmapMemory(device, image.staging_buffer_memory);

  fan::vulkan::image_create(
    *this,
    image_info.size,
    lp.format,
    VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    image.image_index,
    image.image_memory
  );
  image.image_view = create_image_view(image.image_index, lp.format, VK_IMAGE_ASPECT_COLOR_BIT);
  create_texture_sampler(image.sampler, lp);

  transition_image_layout(image.image_index, lp.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  copy_buffer_to_image(image.staging_buffer, image.image_index, lp.format, image_info.size);
  transition_image_layout(image.image_index, lp.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  return nr;
}
fan::graphics::image_nr_t fan::vulkan::context_t::image_load(const fan::image::info_t& image_info) {
  return image_load(image_info, fan::vulkan::context_t::image_load_properties_t());
}
fan::graphics::image_nr_t fan::vulkan::context_t::image_load(fan::color* colors, const fan::vec2ui& size_, const fan::vulkan::context_t::image_load_properties_t& p) {

  fan::image::info_t ii;
  ii.data = colors;
  ii.size = size_;
  ii.channels = 4;
  fan::graphics::image_nr_t nr = image_load(ii, p);

  image_set_settings(nr, p);

  auto& image_data = __fan_internal_image_list[nr];
  image_data.size = size_;

  return nr;
}
fan::graphics::image_nr_t fan::vulkan::context_t::image_load(fan::color* colors, const fan::vec2ui& size_) {
  return image_load(colors, size_, fan::vulkan::context_t::image_load_properties_t());
}
fan::graphics::image_nr_t fan::vulkan::context_t::create_missing_texture() {
  fan::vulkan::context_t::image_load_properties_t p;

  fan::vec2i image_size = fan::vec2i(2, 2);
  fan::graphics::image_nr_t nr = image_load((fan::color*)fan::image::missing_texture_pixels, image_size, p);

  auto& image_data = __fan_internal_image_list[nr];
  image_data.size = image_size;
  __fan_internal_image_list[nr].image_settings = fan::graphics::format_converter::image_vulkan_to_global(p);
  return nr;
}
fan::graphics::image_nr_t fan::vulkan::context_t::create_transparent_texture() {
  fan::vulkan::context_t::image_load_properties_t p;

  fan::vec2i image_size = fan::vec2i(2, 2);
  fan::graphics::image_nr_t nr = image_load((fan::color*)fan::image::transparent_texture_pixels, image_size, p);

  auto& image_data = __fan_internal_image_list[nr];
  image_data.size = image_size;

  return nr;
}
fan::graphics::image_nr_t fan::vulkan::context_t::image_load(fan::str_view_t path, const fan::vulkan::context_t::image_load_properties_t& p, const std::source_location& callers_path) {

#if fan_assert_if_same_path_loaded_multiple_times

  static std::unordered_map<std::string, bool> existing_images;

  if (existing_images.find(path) != existing_images.end()) {
    fan::throw_error("image already existing " + path);
  }

  existing_images[path] = 0;

#endif

  fan::image::info_t image_info;
  if (fan::image::load(path, &image_info, callers_path)) {
    return create_missing_texture();
  }
  fan::graphics::image_nr_t nr = image_load(image_info, p);
  __fan_internal_image_list[nr].image_path = path;
  fan::image::free(&image_info);
  return nr;
}
fan::graphics::image_nr_t fan::vulkan::context_t::image_load(fan::str_view_t path, const std::source_location& callers_path) {
  return image_load(path, fan::vulkan::context_t::image_load_properties_t(), callers_path);
}
void fan::vulkan::context_t::image_unload(fan::graphics::image_nr_t nr) {
  image_erase(nr);
}
void fan::vulkan::context_t::image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::vulkan::context_t::image_load_properties_t& p) {
  auto image_multiplier = get_image_multiplier(p.format);
  VkDeviceSize image_size = image_info.size.multiply() * image_multiplier;

  fan::vulkan::context_t::image_t& image = image_get(nr);
  auto& image_data = __fan_internal_image_list[nr];
  image_data.size = image_info.size;

  VkDeviceSize image_size_bytes = image_size;

  if (image.image_index == 0) {
    create_buffer(
      image_size_bytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      image.staging_buffer,
      image.staging_buffer_memory
    );

    vkMapMemory(device, image.staging_buffer_memory, 0, image_size_bytes, 0, &image.data);

    fan::vulkan::image_create(
      *this,
      image_info.size,
      p.format,
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      image.image_index,
      image.image_memory
    );

    image.image_view = create_image_view(image.image_index, p.format, VK_IMAGE_ASPECT_COLOR_BIT);
    create_texture_sampler(image.sampler, p);
  }

  memcpy(image.data, image_info.data, image_size_bytes);

  transition_image_layout(image.image_index, p.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  copy_buffer_to_image(image.staging_buffer, image.image_index, p.format, image_info.size);
  transition_image_layout(image.image_index, p.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
void fan::vulkan::context_t::image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) {
  image_reload(nr, image_info, fan::vulkan::context_t::image_load_properties_t());
}
void fan::vulkan::context_t::image_reload(fan::graphics::image_nr_t nr, fan::str_view_t path, const fan::vulkan::context_t::image_load_properties_t& p, const std::source_location& callers_path) {
  fan::image::info_t image_info;
  if (fan::image::load(path, &image_info, callers_path)) {
    image_info.data = (void*)fan::image::missing_texture_pixels;
    image_info.size = 2;
    image_info.channels = 4;
    image_info.type = -1; // ignore free
  }
  image_reload(nr, image_info, p);
  __fan_internal_image_list[nr].image_path = path;
  fan::image::free(&image_info);
}
void fan::vulkan::context_t::image_reload(fan::graphics::image_nr_t nr, fan::str_view_t path, const std::source_location& callers_path) {
  image_reload(nr, path, fan::vulkan::context_t::image_load_properties_t(), callers_path);
}
// creates single colored text size.x*size.y sized
fan::graphics::image_nr_t fan::vulkan::context_t::image_create(const fan::color& color, const fan::vulkan::context_t::image_load_properties_t& p) {

  std::uint8_t pixels[4];
  for (std::uint32_t p = 0; p < fan::color::size(); p++) {
    pixels[p] = color[p] * 255;
  }

  fan::image::info_t ii;

  ii.data = (void*)&color.r;
  ii.size = 1;
  ii.channels = 4;
  fan::graphics::image_nr_t nr = image_load(ii, p);

  image_bind(nr);

  image_set_settings(nr, p);

  return nr;
}
fan::graphics::image_nr_t fan::vulkan::context_t::image_create(const fan::color& color) {
  return image_create(color, fan::vulkan::context_t::image_load_properties_t());
}
fan::graphics::image_nr_t fan::vulkan::context_t::image_create(void* data, const fan::vec2ui& size, const fan::vulkan::context_t::image_load_properties_t& p) {
  fan::image::info_t info;
  info.data = data;
  info.size = size;
  info.channels = fan::graphics::get_channel_amount(fan::graphics::format_converter::vulkan_to_global_format(p.format));
  return image_load(info, p);
}
std::vector<std::uint8_t> fan::vulkan::context_t::image_get_pixel_data(fan::graphics::image_nr_t nr, std::uint32_t format, fan::vec2 uvp, fan::vec2 uvs) {
  return {};
}
fan::graphics::image_nr_t fan::vulkan::context_t::image_create_from_view(
  VkImageView view,
  VkImage image,
  fan::vec2ui size,
  VkFormat format
) {
  fan::graphics::image_nr_t img_nr = image_create();
  auto& img = image_get(img_nr);

  img.image_index = image;
  img.image_view = view;
  create_texture_sampler(img.sampler, {});

  return img_nr;
}
//-----------------------------image-----------------------------

      //-----------------------------camera-----------------------------

fan::graphics::camera_nr_t fan::vulkan::context_t::camera_create() {
  return __fan_internal_camera_list.NewNode();
}
fan::graphics::context_camera_t& fan::vulkan::context_t::camera_get(fan::graphics::camera_nr_t nr) {
  return __fan_internal_camera_list[nr];
}
void fan::vulkan::context_t::camera_erase(fan::graphics::camera_nr_t nr) {
  __fan_internal_camera_list.Recycle(nr);
}
void fan::vulkan::context_t::camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
  camera_get(nr).coordinates.v = fan::vec4(x, y);
  camera_update_projection(nr);
  camera_update_view(nr);
}
void fan::vulkan::context_t::camera_update_projection(fan::graphics::camera_nr_t nr) {
  auto& camera = camera_get(nr);

  camera.projection = fan::math::ortho<fan::mat4>(
    camera.coordinates.left / camera.zoom,
    camera.coordinates.right / camera.zoom,
    camera.coordinates.bottom / camera.zoom,
    camera.coordinates.top / camera.zoom,
    -fan::graphics::znearfar / 2,
    fan::graphics::znearfar / 2
  );
}
void fan::vulkan::context_t::camera_update_view(fan::graphics::camera_nr_t nr) {
  auto& camera = camera_get(nr);
  camera.view[3][0] = 0;
  camera.view[3][1] = 0;
  camera.view[3][2] = 0;
  camera.view = camera.view.translate(camera.position);
  fan::vec3 position = camera.view.get_translation();
  constexpr fan::vec3 front(0, 0, 1);
  camera.view = fan::math::look_at_left<fan::mat4, fan::vec3>(position, position + front, fan::camera::world_up);
}
fan::graphics::camera_nr_t fan::vulkan::context_t::camera_create(const fan::vec2& x, const fan::vec2& y) {
  fan::graphics::camera_nr_t nr = camera_create();
  camera_set_ortho(nr, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
  return nr;
}
fan::vec3 fan::vulkan::context_t::camera_get_position(fan::graphics::camera_nr_t nr) {
  return camera_get(nr).position;
}
void fan::vulkan::context_t::camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
  auto& camera = camera_get(nr);
  camera.position = cp;
  camera_update_view(nr);
}
fan::vec3 fan::vulkan::context_t::camera_get_center(fan::graphics::camera_nr_t nr) {
  auto& c = camera_get(nr);
  fan::vec2 center_offset = fan::vec2(
    c.coordinates.left + c.coordinates.right,
    c.coordinates.top + c.coordinates.bottom
  ) / (2.f * c.zoom);
  return fan::vec2(c.position.x, c.position.y) + center_offset;
}
void fan::vulkan::context_t::camera_set_center(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
  auto& c = camera_get(nr);
  fan::vec2 center_offset = fan::vec2(
    c.coordinates.left + c.coordinates.right,
    c.coordinates.top + c.coordinates.bottom
  ) / (2.f * c.zoom);

  camera_set_position(nr, fan::vec3(cp.xy() - center_offset, cp.z));
}
fan::vec2 fan::vulkan::context_t::camera_get_size(fan::graphics::camera_nr_t nr) {
  fan::graphics::context_camera_t& camera = camera_get(nr);
  return fan::vec2(std::abs(camera.coordinates.right - camera.coordinates.left), std::abs(camera.coordinates.bottom - camera.coordinates.top));
}
f32_t fan::vulkan::context_t::camera_get_zoom(fan::graphics::camera_nr_t nr) {
  return camera_get(nr).zoom;
}
void fan::vulkan::context_t::camera_set_zoom(fan::graphics::camera_nr_t nr, f32_t new_zoom) {
  camera_get(nr).zoom = new_zoom;
  camera_update_projection(nr);
  camera_update_view(nr);
}
void fan::vulkan::context_t::camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
  fan::graphics::context_camera_t& camera = camera_get(nr);

  camera.fov = fov;
  camera.projection = fan::math::perspective<fan::mat4>(fan::math::radians(camera.fov), (f32_t)window_size.x / (f32_t)window_size.y, camera.znear, camera.zfar);

  camera.update_view();

  camera.view = camera.get_view_matrix();

  //auto it = gloco()->m_viewport_resize_callback.GetNodeFirst();

  //while (it != gloco()->m_viewport_resize_callback.dst) {

  //  gloco()->m_viewport_resize_callback.StartSafeNext(it);

  //  resize_cb_data_t cbd;
  //  cbd.camera = this;
  //  cbd.position = get_position();
  //  cbd.size = get_camera_size();
  //  gloco()->m_viewport_resize_callback[it].data(cbd);

  //  it = gloco()->m_viewport_resize_callback.EndSafeNext();
  //}
}
void fan::vulkan::context_t::camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
  fan::graphics::context_camera_t& camera = camera_get(nr);
  camera.rotate_camera(offset);
  camera.view = camera.get_view_matrix();
}
//-----------------------------camera-----------------------------

      //-----------------------------viewport-----------------------------

void fan::vulkan::context_t::viewport_set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  VkViewport viewport {};
  viewport.x = viewport_position_.x;
  viewport.y = viewport_position_.y;
  viewport.width = viewport_size_.x;
  viewport.height = viewport_size_.y;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkCommandBufferBeginInfo beginInfo {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  if (!command_buffer_in_use) {
    VkResult result = vkGetFenceStatus(device, in_flight_fences[current_frame]);
    if (result == VK_NOT_READY) {
      vkDeviceWaitIdle(device);
    }

    if (vkBeginCommandBuffer(command_buffers[current_frame], &beginInfo) != VK_SUCCESS) {
      fan::throw_error("failed to begin recording command buffer!");
    }
  }
  vkCmdSetViewport(command_buffers[current_frame], 0, 1, &viewport);

  if (!command_buffer_in_use) {
    if (vkEndCommandBuffer(command_buffers[current_frame]) != VK_SUCCESS) {
      fan::throw_error("failed to record command buffer!");
    }
    command_buffer_in_use = false;
  }
}
fan::graphics::context_viewport_t& fan::vulkan::context_t::viewport_get(fan::graphics::viewport_nr_t nr) {
  return __fan_internal_viewport_list[nr];
}
void fan::vulkan::context_t::viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  fan::graphics::context_viewport_t& viewport = viewport_get(nr);
  viewport.position = viewport_position_;
  viewport.size = viewport_size_;

  viewport_set(viewport_position_, viewport_size_, window_size);
}
fan::graphics::viewport_nr_t fan::vulkan::context_t::viewport_create() {
  auto nr = __fan_internal_viewport_list.NewNode();

  viewport_set(nr, 0, 1, 0);
  return nr;
}
void fan::vulkan::context_t::viewport_erase(fan::graphics::viewport_nr_t nr) {
  __fan_internal_viewport_list.Recycle(nr);
}
fan::vec2 fan::vulkan::context_t::viewport_get_position(fan::graphics::viewport_nr_t nr) {
  return viewport_get(nr).position;
}
fan::vec2 fan::vulkan::context_t::viewport_get_size(fan::graphics::viewport_nr_t nr) {
  return viewport_get(nr).size;
}
void fan::vulkan::context_t::viewport_zero(fan::graphics::viewport_nr_t nr) {
  auto& viewport = viewport_get(nr);
  viewport.position = 0;
  viewport.size = 0;
  viewport_set(0, 0, 0); // window_size not used
}
bool fan::vulkan::context_t::viewport_inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
  fan::graphics::context_viewport_t& viewport = viewport_get(nr);
  return fan::math::d2::aabb_point_inside(position, viewport.position + viewport.size / 2, viewport.size / 2);
}
bool fan::vulkan::context_t::viewport_inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
  fan::graphics::context_viewport_t& viewport = viewport_get(nr);
  return fan::math::d2::aabb_point_inside(position, viewport.size / 2, viewport.size / 2);
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
#if defined(loco_window)

void fan::vulkan::context_t::open(fan::window_t& window) {
  window_resize_handle = window.add_resize_callback([&](const fan::window_t::resize_data_t& d) {
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
#if defined(FAN_GUI)
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

#endif
void fan::vulkan::context_t::close_vais(std::vector<fan::vulkan::vai_t>& v) {
  for (auto& e : v) {
    e.close(*this);
  }
}
void fan::vulkan::context_t::destroy_vulkan_soft() {
  vkDeviceWaitIdle(device);
  fan::vulkan::context_t& context = *this;
  {
    fan::graphics::shader_list_t::nrtra_t nrtra;
    fan::graphics::shader_list_t::nr_t nr;
    nrtra.Open(&__fan_internal_shader_list, &nr);
    while (nrtra.Loop(&__fan_internal_shader_list, &nr)) {
      shader_erase(nr, 0);
    }
    nrtra.Close(&__fan_internal_shader_list);
  }
  {
    fan::graphics::image_list_t::nrtra_t nrtra;
    fan::graphics::image_list_t::nr_t nr;
    nrtra.Open(&__fan_internal_image_list, &nr);
    while (nrtra.Loop(&__fan_internal_image_list, &nr)) {
      image_erase(nr, 0);
    }
    nrtra.Close(&__fan_internal_image_list);
  }

  close_vais(mainColorImageViews);
  close_vais(postProcessedColorImageViews);
  close_vais(depthImageViews);
  close_vais(downscaleImageViews1);
  close_vais(upscaleImageViews1);
  close_vais(vai_depth);

  for (std::size_t i = 0; i < max_frames_in_flight; i++) {
    if (render_finished_semaphores.size())
      vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
    if (image_available_semaphores.size())
      vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
    if (in_flight_fences.size())
      vkDestroyFence(device, in_flight_fences[i], nullptr);
  }

  vkDestroyRenderPass(device, render_pass, nullptr);
  vkDestroyCommandPool(device, command_pool, nullptr);

#if FAN_DEBUG >= fan_debug_high
  if (supports_validation_layers) {
    DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
  }
#endif
}
void fan::vulkan::context_t::gui_close() {
  vkFreeCommandBuffers(device, command_pool, command_buffers.size(), command_buffers.data());
  cleanup_swap_chain_dependencies();
  descriptor_pool.close(*this);
  destroy_vulkan_soft();
#if defined(FAN_GUI)
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
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
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

#if FAN_DEBUG >= fan_debug_high
  if (!check_validation_layer_support()) {
    fan::print_warning("validation layers not supported");
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

#if fan_debug >= 2
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
  if (deviceProperties.apiVersion < VK_API_VERSION_1_2) {
    fan::throw_error("unsupported Vulkan apiVersion:", appInfo.apiVersion);
  }
#endif

  VkInstanceCreateInfo createInfo {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  auto extensions = get_required_extensions();
  createInfo.enabledExtensionCount = extensions.size();
  std::vector<char*> extension_names(extensions.size() + 1);

  for (std::uint32_t i = 0; i < extensions.size(); ++i) {
    extension_names[i] = new char[extensions[i].size() + 1];
    memcpy(extension_names[i], extensions[i].data(), extensions[i].size() + 1);
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

#if defined(ENABLE_RAYTRACING_DEPENDENCIES)
  vulkan12.bufferDeviceAddress = VK_TRUE;

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel {};
  accel.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
  accel.accelerationStructure = VK_TRUE;

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
    fan::print_impl("vkCreateDevice failed with code:", (int)r);
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

  for (std::size_t i = 0; i < swap_chain_image_views.size(); i++) {
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

  for (std::uint32_t i = 0; i < swap_chain_images.size(); i++) {
    swap_chain_image_views[i] = create_image_view(swap_chain_images[i], swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT);
  }
}
void fan::vulkan::context_t::create_render_pass() {
  //--------------attachment description--------------

  VkAttachmentDescription mainColorAttachment {};
  mainColorAttachment.format = swap_chain_image_format;
  mainColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  mainColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  mainColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  mainColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  mainColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  mainColorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  mainColorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkAttachmentDescription postProcessedColorAttachment {};
  postProcessedColorAttachment.format = swap_chain_image_format;
  postProcessedColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  postProcessedColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  postProcessedColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  postProcessedColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  postProcessedColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  postProcessedColorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  postProcessedColorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentDescription depthAttachment {};
  depthAttachment.format = find_depth_format();
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  //--------------attachment description--------------

  VkAttachmentReference mainSceneColorRef {};
  mainSceneColorRef.attachment = 0;
  mainSceneColorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference postProcessInputRef {};
  postProcessInputRef.attachment = 0;
  postProcessInputRef.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkAttachmentReference postProcessOutputRef {};
  postProcessOutputRef.attachment = 1;
  postProcessOutputRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthRef {};
  depthRef.attachment = 2;
  depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription mainSceneSubpass {};
  mainSceneSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  mainSceneSubpass.colorAttachmentCount = 1;
  mainSceneSubpass.pColorAttachments = &mainSceneColorRef;
  mainSceneSubpass.pDepthStencilAttachment = &depthRef;

  VkSubpassDescription postProcessSubpass {};
  postProcessSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  postProcessSubpass.inputAttachmentCount = 1;
  postProcessSubpass.pInputAttachments = &postProcessInputRef;
  postProcessSubpass.colorAttachmentCount = 1;
  postProcessSubpass.pColorAttachments = &postProcessOutputRef;

  VkSubpassDependency extToMainDep {};
  extToMainDep.srcSubpass = VK_SUBPASS_EXTERNAL;
  extToMainDep.dstSubpass = 0;
  extToMainDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  extToMainDep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  extToMainDep.srcAccessMask = 0;
  extToMainDep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  extToMainDep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  VkSubpassDependency mainToPostDep {};
  mainToPostDep.srcSubpass = 0;
  mainToPostDep.dstSubpass = 1;
  mainToPostDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  mainToPostDep.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  mainToPostDep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  mainToPostDep.dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
  mainToPostDep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  VkSubpassDependency postToExtDep {};
  postToExtDep.srcSubpass = 1;
  postToExtDep.dstSubpass = VK_SUBPASS_EXTERNAL;
  postToExtDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  postToExtDep.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  postToExtDep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  postToExtDep.dstAccessMask = 0;
  postToExtDep.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;


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
  swap_chain_framebuffers.resize(swap_chain_image_views.size());

  for (std::size_t i = 0; i < swap_chain_image_views.size(); i++) {
    VkImageView attachments[] = {
      mainColorImageViews[i].image_view,
      swap_chain_image_views[i],
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
void fan::vulkan::context_t::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
  VkBufferCreateInfo bufferInfo {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  fan::vulkan::validate(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

  VkMemoryAllocateFlagsInfo allocFlags {};
  allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  allocFlags.flags = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;
  allocFlags.pNext = nullptr;

  VkMemoryAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = find_memory_type(memRequirements.memoryTypeBits, properties);
  allocInfo.pNext = (allocFlags.flags != 0) ? &allocFlags : nullptr;

  fan::vulkan::validate(vkAllocateMemory(device, &allocInfo, nullptr, &buffer_memory));
  fan::vulkan::validate(vkBindBufferMemory(device, buffer, buffer_memory, 0));
}
VkCommandBuffer fan::vulkan::context_t::begin_single_time_commands() {
  VkCommandBufferAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = command_pool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  return commandBuffer;
}
void fan::vulkan::context_t::end_single_time_commands(VkCommandBuffer command_buffer) {
  vkEndCommandBuffer(command_buffer);

  VkSubmitInfo submitInfo {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffer;

  vkQueueSubmit(graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue);

  vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}
void fan::vulkan::context_t::copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size) {
  VkCommandBuffer commandBuffer = begin_single_time_commands();

  VkBufferCopy copyRegion {};
  copyRegion.size = size;
  vkCmdCopyBuffer(commandBuffer, src_buffer, dst_buffer, 1, &copyRegion);

  end_single_time_commands(commandBuffer);
}
std::uint32_t fan::vulkan::context_t::find_memory_type(std::uint32_t type_filter, VkMemoryPropertyFlags properties) const {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &memProperties);

  for (std::uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  fan::throw_error("failed to find suitable memory type!");
  return {};
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
  image_available_semaphores.resize(max_frames_in_flight);
  render_finished_semaphores.resize(max_frames_in_flight);
  in_flight_fences.resize(max_frames_in_flight);

  VkSemaphoreCreateInfo semaphoreInfo {};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceInfo {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (std::size_t i = 0; i < max_frames_in_flight; i++) {
    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &image_available_semaphores[i]) != VK_SUCCESS ||
      vkCreateSemaphore(device, &semaphoreInfo, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS ||
      vkCreateFence(device, &fenceInfo, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
      fan::throw_error("failed to create synchronization objects for a frame!");
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
VkResult fan::vulkan::context_t::end_render() {
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

  VkSemaphore waitSemaphores[] = {image_available_semaphores[current_frame]};
  VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &command_buffers[current_frame];

  VkSemaphore signalSemaphores[] = {render_finished_semaphores[current_frame]};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  VkResult submit_result = vkQueueSubmit(graphics_queue, 1, &submitInfo, in_flight_fences[current_frame]);
  if (submit_result != VK_SUCCESS) {
    fan::print_impl("vkQueueSubmit error:", (int)submit_result);
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

  std::uint32_t extensions_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extensions_count, nullptr);
  if (extensions_count == 0) {
    fan::throw_error("Could not get the number of Instance extensions.");
  }

  std::vector<VkExtensionProperties> available_extensions;

  available_extensions.resize(extensions_count);

  vkEnumerateInstanceExtensionProperties(nullptr, &extensions_count, &available_extensions[0]);

  if (extensions_count == 0) {
    fan::throw_error("Could not enumerate Instance extensions.");
  }

  std::vector<std::string> extension_str(available_extensions.size());

  for (int i = 0; i < available_extensions.size(); i++) {
    extension_str[i] = available_extensions[i].extensionName;
  }

#if FAN_DEBUG >= fan_debug_high
  if (supports_validation_layers) {
    extension_str.push_back((char*)VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
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
      fan::print_warning("missing Vulkan validation layer:", layerName);
      fan::print_warning("available Vulkan instance layers:");
      for (const auto& layerProperties : availableLayers) {
        fan::print_warning("  ", layerProperties.layerName);
      }
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
  fan::print_impl("validation layer:", pCallbackData->pMessage);
  // system("pause");
//  exit(0);

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

auto fan::graphics::format_converter::image_global_to_vulkan(const fan::graphics::image_load_properties_t& p) {
  return fan::vulkan::context_t::image_load_properties_t {
    .visual_output = global_to_vulkan_address_mode(p.visual_output),
    .format = global_to_vulkan_format(p.format),
    .min_filter = global_to_vulkan_filter(p.min_filter),
    .mag_filter = global_to_vulkan_filter(p.mag_filter),
  };
}
void fan::vulkan::image_create(const fan::vulkan::context_t& context, const fan::vec2ui& image_size, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
  VkImageCreateInfo imageInfo {};
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
    fan::throw_error("failed to create image!");
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(context.device, image, &memRequirements);

  VkMemoryAllocateInfo allocInfo {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = context.find_memory_type(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(context.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
    fan::throw_error("failed to allocate image memory!");
  }

  vkBindImageMemory(context.device, image, imageMemory, 0);
}
void fan::vulkan::vai_t::open(fan::vulkan::context_t& context, const properties_t& p) {
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
void fan::vulkan::vai_t::close(fan::vulkan::context_t& context) {
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
fan::graphics::context_functions_t fan::graphics::get_vk_context_functions() {
  fan::graphics::context_functions_t cf;
  cf.shader_create = [](void* context) {
    return ((fan::vulkan::context_t*)context)->shader_create();
  };
  cf.shader_get = [](void* context, fan::graphics::shader_nr_t nr) {
    return (void*)&((fan::vulkan::context_t*)context)->shader_get(nr);
  };
  cf.shader_erase = [](void* context, fan::graphics::shader_nr_t nr) {
    ((fan::vulkan::context_t*)context)->shader_erase(nr);
  };
  cf.shader_use = [](void* context, fan::graphics::shader_nr_t nr) {
    ((fan::vulkan::context_t*)context)->shader_use(nr);
  };
  cf.shader_set_vertex = [](void* context, fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code) {
    ((fan::vulkan::context_t*)context)->shader_set_vertex(nr, file_path, vertex_code);
  };
  cf.shader_set_fragment = [](void* context, fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code) {
    ((fan::vulkan::context_t*)context)->shader_set_fragment(nr, file_path, fragment_code);
  };
  cf.shader_set_compute = [](void* context, fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& compute_code) {
    ((fan::vulkan::context_t*)context)->shader_set_compute(nr, file_path, compute_code);
  };
  cf.shader_dispatch_compute = [](void* context, fan::graphics::shader_nr_t nr, uint32_t x, uint32_t y, uint32_t z) {
    ((fan::vulkan::context_t*)context)->shader_dispatch_compute(nr, x, y, z);
  };
  cf.shader_compile = [](void* context, fan::graphics::shader_nr_t nr) {
    return ((fan::vulkan::context_t*)context)->shader_compile(nr);
  };
  /*image*/
  cf.image_create = [](void* context) {
    return ((fan::vulkan::context_t*)context)->image_create();
  };
  cf.image_get_handle = [](void* context, fan::graphics::image_nr_t nr) {
    return (std::uint64_t)((fan::vulkan::context_t*)context)->image_get_handle(nr);
  };
  cf.image_get = [](void* context, fan::graphics::image_nr_t nr) {
    return (void*)&((fan::vulkan::context_t*)context)->image_get(nr);
  };
  cf.image_erase = [](void* context, fan::graphics::image_nr_t nr) {
    ((fan::vulkan::context_t*)context)->image_erase(nr);
  };
  cf.image_bind = [](void* context, fan::graphics::image_nr_t nr) {
    ((fan::vulkan::context_t*)context)->image_bind(nr);
  };
  cf.image_bind_unit = [](void* context, fan::graphics::image_nr_t nr, std::uint32_t unit) {
    ((fan::vulkan::context_t*)context)->image_bind(nr, unit);
  };
  cf.image_bind_params = [](void* context, fan::graphics::image_nr_t nr, std::uint32_t unit, std::uint32_t access, std::uint32_t format) {
    ((fan::vulkan::context_t*)context)->image_bind(nr, unit, access, format);
  };
  cf.image_unbind = [](void* context, fan::graphics::image_nr_t nr) {
    ((fan::vulkan::context_t*)context)->image_unbind(nr);
  };
  cf.image_get_settings = [](void* context, fan::graphics::image_nr_t nr) -> fan::graphics::image_load_properties_t& {
    return ((fan::vulkan::context_t*)context)->image_get_settings(nr);
  };
  cf.image_set_settings = [](void* context, fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings) {
    ((fan::vulkan::context_t*)context)->image_set_settings(nr, fan::graphics::format_converter::image_global_to_vulkan(settings));
  };
  cf.image_load_info = [](void* context, const fan::image::info_t& image_info) {
    return ((fan::vulkan::context_t*)context)->image_load(image_info);
  };
  cf.image_load_info_props = [](void* context, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
    return ((fan::vulkan::context_t*)context)->image_load(image_info, fan::graphics::format_converter::image_global_to_vulkan(p));
  };
  cf.image_load_path = [](void* context, fan::str_view_t path, const std::source_location& callers_path = std::source_location::current()) {
    return ((fan::vulkan::context_t*)context)->image_load(path, callers_path);
  };
  cf.image_load_path_props = [](void* context, fan::str_view_t path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
    return ((fan::vulkan::context_t*)context)->image_load(path, fan::graphics::format_converter::image_global_to_vulkan(p), callers_path);
  };
  cf.image_load_colors = [](void* context, fan::color* colors, const fan::vec2ui& size_) {
    return ((fan::vulkan::context_t*)context)->image_load(colors, size_);
  };
  cf.image_load_colors_props = [](void* context, fan::color* colors, const fan::vec2ui& size_, const fan::graphics::image_load_properties_t& p) {
    return ((fan::vulkan::context_t*)context)->image_load(colors, size_, fan::graphics::format_converter::image_global_to_vulkan(p));
  };
  cf.image_unload = [](void* context, fan::graphics::image_nr_t nr) {
    ((fan::vulkan::context_t*)context)->image_unload(nr);
  };
  cf.create_missing_texture = [](void* context) {
    return ((fan::vulkan::context_t*)context)->create_missing_texture();
  };
  cf.create_transparent_texture = [](void* context) {
    return ((fan::vulkan::context_t*)context)->create_transparent_texture();
  };
  cf.image_reload_image_info = [](void* context, fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) {
    return ((fan::vulkan::context_t*)context)->image_reload(nr, image_info);
  };
  cf.image_reload_image_info_props = [](void* context, fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
    return ((fan::vulkan::context_t*)context)->image_reload(nr, image_info, fan::graphics::format_converter::image_global_to_vulkan(p));
  };
  cf.image_reload_path = [](void* context, fan::graphics::image_nr_t nr, fan::str_view_t path, const std::source_location& callers_path = std::source_location::current()) {
    return ((fan::vulkan::context_t*)context)->image_reload(nr, path, callers_path);
  };
  cf.image_reload_path_props = [](void* context, fan::graphics::image_nr_t nr, fan::str_view_t path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path = std::source_location::current()) {
    return ((fan::vulkan::context_t*)context)->image_reload(nr, path, fan::graphics::format_converter::image_global_to_vulkan(p), callers_path);
  };
  cf.image_create_color = [](void* context, const fan::color& color) {
    return ((fan::vulkan::context_t*)context)->image_create(color);
  };
  cf.image_create_color_props = [](void* context, const fan::color& color, const fan::graphics::image_load_properties_t& p) {
    return ((fan::vulkan::context_t*)context)->image_create(color, fan::graphics::format_converter::image_global_to_vulkan(p));
  };
  cf.image_create_data = [](void* context, void* data, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
    return ((fan::vulkan::context_t*)context)->image_create(data, size, fan::graphics::format_converter::image_global_to_vulkan(p));
  };
  /*camera*/
  cf.camera_create = [](void* context) {
    return ((fan::vulkan::context_t*)context)->camera_create();
  };
  cf.camera_get = [](void* context, fan::graphics::camera_nr_t nr) -> decltype(auto) {
    return ((fan::vulkan::context_t*)context)->camera_get(nr);
  };
  cf.camera_erase = [](void* context, fan::graphics::camera_nr_t nr) {
    ((fan::vulkan::context_t*)context)->camera_erase(nr);
  };
  cf.camera_create_params = [](void* context, const fan::vec2& x, const fan::vec2& y) {
    return ((fan::vulkan::context_t*)context)->camera_create(x, y);
  };
  cf.camera_get_position = [](void* context, fan::graphics::camera_nr_t nr) {
    return ((fan::vulkan::context_t*)context)->camera_get_position(nr);
  };
  cf.camera_set_position = [](void* context, fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
    ((fan::vulkan::context_t*)context)->camera_set_position(nr, cp);
  };
  cf.camera_get_center = [](void* context, fan::graphics::camera_nr_t nr) {
    return ((fan::vulkan::context_t*)context)->camera_get_center(nr);
  };
  cf.camera_set_center = [](void* context, fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
    ((fan::vulkan::context_t*)context)->camera_set_center(nr, cp);
  };
  cf.camera_get_size = [](void* context, fan::graphics::camera_nr_t nr) {
    return ((fan::vulkan::context_t*)context)->camera_get_size(nr);
  };
  cf.camera_get_zoom = [](void* context, fan::graphics::camera_nr_t nr) {
    return ((fan::vulkan::context_t*)context)->camera_get_zoom(nr);
  };
  cf.camera_set_zoom = [](void* context, fan::graphics::camera_nr_t nr, f32_t new_zoom) {
    ((fan::vulkan::context_t*)context)->camera_set_zoom(nr, new_zoom);
  };
  cf.camera_set_ortho = [](void* context, fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
    ((fan::vulkan::context_t*)context)->camera_set_ortho(nr, x, y);
  };
  cf.camera_set_perspective = [](void* context, fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
    ((fan::vulkan::context_t*)context)->camera_set_perspective(nr, fov, window_size);
  };
  cf.camera_rotate = [](void* context, fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
    ((fan::vulkan::context_t*)context)->camera_rotate(nr, offset);
  };
  /*viewport*/
  cf.viewport_create = [](void* context) {
    return ((fan::vulkan::context_t*)context)->viewport_create();
  };
  cf.viewport_create_params = [](void* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
    auto vk_context = ((fan::vulkan::context_t*)context);
    auto nr = vk_context->viewport_create();
    vk_context->viewport_set(nr, viewport_position_, viewport_size_, window_size);
    return nr;
  };
  cf.viewport_get = [](void* context, fan::graphics::viewport_nr_t nr) -> fan::graphics::context_viewport_t& {
    return ((fan::vulkan::context_t*)context)->viewport_get(nr);
  };
  cf.viewport_erase = [](void* context, fan::graphics::viewport_nr_t nr) {
    ((fan::vulkan::context_t*)context)->viewport_erase(nr);
  };
  cf.viewport_get_position = [](void* context, fan::graphics::viewport_nr_t nr) {
    return ((fan::vulkan::context_t*)context)->viewport_get_position(nr);
  };
  cf.viewport_get_size = [](void* context, fan::graphics::viewport_nr_t nr) {
    return ((fan::vulkan::context_t*)context)->viewport_get_size(nr);
  };
  cf.viewport_set = [](void* context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
    ((fan::vulkan::context_t*)context)->viewport_set(viewport_position_, viewport_size_, window_size);
  };
  cf.viewport_set_nr = [](void* context, fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
    ((fan::vulkan::context_t*)context)->viewport_set(nr, viewport_position_, viewport_size_, window_size);
  };
  cf.viewport_zero = [](void* context, fan::graphics::viewport_nr_t nr) {
    ((fan::vulkan::context_t*)context)->viewport_zero(nr);
  };
  cf.viewport_inside = [](void* context, fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
    return ((fan::vulkan::context_t*)context)->viewport_inside(nr, position);
  };
  cf.viewport_inside_wir = [](void* context, fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
    return ((fan::vulkan::context_t*)context)->viewport_inside_wir(nr, position);
  };
  cf.image_get_pixel_data = [](void* context, fan::graphics::image_nr_t nr, std::uint32_t format, fan::vec2 uvp, fan::vec2 uvs) {
    return ((fan::vulkan::context_t*)context)->image_get_pixel_data(nr, format, uvp, uvs);
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
#endif
