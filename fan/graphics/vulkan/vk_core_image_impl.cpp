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
#include <vulkan/vk_enum_string_helper.h>

module fan.graphics.vulkan.core;

import std;

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
import fan.graphics.webp;

#include <fan/stb/stb_image.h>

import fan.math;
import fan.math.intersection;

#define __fan_internal_camera_list (*fan::graphics::ctx().camera_list)
#define __fan_internal_shader_list (*fan::graphics::ctx().shader_list)
#define __fan_internal_image_list (*fan::graphics::ctx().image_list)
#define __fan_internal_viewport_list (*fan::graphics::ctx().viewport_list)

#define ENABLE_RAYTRACING_DEPENDENCIES

#define VK_CTX ((fan::vulkan::context_t*)context)

VkFormat fan::graphics::format_converter::global_to_vulkan_format(std::uintptr_t format) {
  switch (format) {
    case image_format_e::bgra: return VK_FORMAT_B8G8R8A8_UNORM;
    case image_format_e::rgba: return VK_FORMAT_R8G8B8A8_UNORM;
    case image_format_e::r8_unorm: return VK_FORMAT_R8_UNORM;
    case image_format_e::r8_uint: return VK_FORMAT_R8_UINT;
    case image_format_e::rg8_unorm: return VK_FORMAT_R8G8_UNORM;
    case image_format_e::rgb_unorm: return VK_FORMAT_R8G8B8A8_UNORM;
    case image_format_e::r8g8b8a8_srgb: return VK_FORMAT_R8G8B8A8_SRGB;
    case image_format_e::rgba_unorm: return VK_FORMAT_R8G8B8A8_UNORM;
  }
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
  switch (format) {
    case VK_FORMAT_B8G8R8A8_UNORM: return fan::graphics::image_format_e::bgra;
    case VK_FORMAT_R8G8B8A8_UNORM: return fan::graphics::image_format_e::rgba;
    case VK_FORMAT_R8_UNORM: return fan::graphics::image_format_e::r8_unorm;
    case VK_FORMAT_R8_UINT: return fan::graphics::image_format_e::r8_uint;
    case VK_FORMAT_R8G8_UNORM: return fan::graphics::image_format_e::rg8_unorm;
    case VK_FORMAT_R8G8B8_UNORM: return fan::graphics::image_format_e::rgb_unorm;
    case VK_FORMAT_R8G8B8A8_SRGB: return fan::graphics::image_format_e::r8g8b8a8_srgb;
    default: break;
  }
#if FAN_DEBUG >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return fan::graphics::image_format_e::rgba_unorm;
}

std::uint32_t fan::graphics::format_converter::vulkan_to_global_address_mode(VkSamplerAddressMode mode) {
  switch (mode) {
    case VK_SAMPLER_ADDRESS_MODE_REPEAT: return fan::graphics::image_sampler_address_mode_e::repeat;
    case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return fan::graphics::image_sampler_address_mode_e::mirrored_repeat;
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: return fan::graphics::image_sampler_address_mode_e::clamp_to_edge;
    case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: return fan::graphics::image_sampler_address_mode_e::clamp_to_border;
    case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: return fan::graphics::image_sampler_address_mode_e::mirrored_clamp_to_edge;
    default: break;
  }
#if FAN_DEBUG >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return fan::graphics::image_sampler_address_mode_e::repeat;
}

std::uint32_t fan::graphics::format_converter::vulkan_to_global_filter(VkFilter filter) {
  switch (filter) {
    case VK_FILTER_NEAREST: return fan::graphics::image_filter_e::nearest;
    case VK_FILTER_LINEAR: return fan::graphics::image_filter_e::linear;
    default: break;
  }
#if FAN_DEBUG >= fan_debug_high
  fan::throw_error("invalid format");
#endif
  return fan::graphics::image_filter_e::nearest;
}

void fan::vulkan::validate(VkResult result) {
  if (result != VK_SUCCESS) {
    fan::throw_error("function failed with:", string_VkResult(result));
  }
}
void fan::vulkan::context_t::transition_image_layout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
  VkCommandBuffer command_buffer = begin_single_time_commands();
  transition_image_layout_cmd(command_buffer, image, format, oldLayout, newLayout);
  end_single_time_commands(command_buffer);
}
void fan::vulkan::context_t::transition_image_layout_cmd(VkCommandBuffer command_buffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {

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
  else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
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
}
void fan::vulkan::context_t::copy_buffer_to_image(
  VkBuffer buffer,
  VkImage image,
  VkFormat format,
  const fan::vec2ui& size,
  VkDeviceSize buffer_offset) {
  VkCommandBuffer command_buffer = begin_single_time_commands();
  copy_buffer_to_image_cmd(command_buffer, buffer, image, format, size, buffer_offset);
  end_single_time_commands(command_buffer);
}
void fan::vulkan::context_t::copy_buffer_to_image_cmd(
  VkCommandBuffer command_buffer,
  VkBuffer buffer,
  VkImage image,
  VkFormat format,
  const fan::vec2ui& size,
  VkDeviceSize buffer_offset) {

  VkBufferImageCopy region {};
  region.bufferOffset = buffer_offset;
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
    case 3: return VK_FORMAT_R8G8B8A8_UNORM;
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
#if defined(FAN_GUI)
  auto& img = image_get(nr);
  if (img.image_view == VK_NULL_HANDLE || img.sampler == VK_NULL_HANDLE) {
    return 0;
  }
  if (img.gui_descriptor_set == VK_NULL_HANDLE ||
    img.gui_image_view != img.image_view ||
    img.gui_sampler != img.sampler) {
    if (img.gui_descriptor_set != VK_NULL_HANDLE) {
      ImGui_ImplVulkan_RemoveTexture(img.gui_descriptor_set);
    }
    img.gui_descriptor_set = ImGui_ImplVulkan_AddTexture(
      img.sampler,
      img.image_view,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );
    img.gui_image_view = img.image_view;
    img.gui_sampler = img.sampler;
  }
  return (std::uint64_t)img.gui_descriptor_set;
#else
  return 0;
#endif
}
fan::vulkan::context_t::image_t& fan::vulkan::context_t::image_get(fan::graphics::image_nr_t nr) {
  return *(fan::vulkan::context_t::image_t*)__fan_internal_image_list[nr].internal;
}
void fan::vulkan::context_t::image_erase(fan::graphics::image_nr_t nr, int recycle) {
  auto& node = __fan_internal_image_list[nr];
  auto& img = image_get(nr);

  VkDevice dev = device;
  VmaAllocator alloc = allocator;
  
  auto to_destroy = [
    dev, alloc, 
    gui_ds = img.gui_descriptor_set,
    sampler = img.sampler,
    view = (img.image_view && img.owns_image_view) ? img.image_view : VK_NULL_HANDLE,
    image = (img.image_index && img.owns_image) ? img.image_index : VK_NULL_HANDLE,
    alloc_handle = (img.image_index && img.owns_image) ? img.image_allocation : VK_NULL_HANDLE,
    staging_buf = img.staging_buffer,
    staging_alloc = img.staging_allocation
  ]() mutable {
    #if defined(FAN_GUI)
      if (gui_ds != VK_NULL_HANDLE) ImGui_ImplVulkan_RemoveTexture(gui_ds);
    #endif
    if (sampler != VK_NULL_HANDLE) vkDestroySampler(dev, sampler, nullptr);
    if (view != VK_NULL_HANDLE) vkDestroyImageView(dev, view, nullptr);
    if (image != VK_NULL_HANDLE) vmaDestroyImage(alloc, image, alloc_handle);
    if (staging_buf != VK_NULL_HANDLE) {
        // You would need to make sure destroy_buffer logic is accessible here
        // or just use vmaDestroyBuffer directly
        vmaDestroyBuffer(alloc, staging_buf, staging_alloc);
    }
  };

  get_current_deletion_queue().push_function(std::move(to_destroy));

  img.gui_descriptor_set = VK_NULL_HANDLE;
  img.sampler = VK_NULL_HANDLE;
  img.image_view = VK_NULL_HANDLE;
  img.image_index = VK_NULL_HANDLE;
  img.staging_buffer = VK_NULL_HANDLE;

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

  auto alloc = staging_ring_buffer.allocate(image_size_bytes);

  image.staging_size = image_size_bytes;

  const std::uint8_t* src = static_cast<const std::uint8_t*>(image_info.data);
  std::uint8_t* dst = static_cast<std::uint8_t*>(alloc.mapped_ptr);
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
    fan::throw_error("image_load: unsupported channel/format combination");
  }

  fan::vulkan::image_create(
    *this,
    image_info.size,
    lp.format,
    VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    image.image_index,
    image.image_allocation
  );
  image.image_view = create_image_view(image.image_index, lp.format, VK_IMAGE_ASPECT_COLOR_BIT);
  create_texture_sampler(image.sampler, lp);

  VkCommandBuffer cmd = begin_async_transfer_commands();
  transition_image_layout_cmd(cmd, image.image_index, lp.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  copy_buffer_to_image_cmd(cmd, alloc.buffer, image.image_index, lp.format, image_info.size, alloc.offset);
  transition_image_layout_cmd(cmd, image.image_index, lp.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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

fan::graphics::image_nr_t fan::vulkan::context_t::request_image_load_async(
  fan::str_view_t path,
  const fan::vulkan::context_t::image_load_properties_t& p,
  std::function<void(const fan::vulkan::decoded_image_payload_t&)> on_gpu_uploaded
) {
  fan::graphics::image_nr_t nr = image_create();
  
  fan::vec2ui size{0, 0};
  if (fan::webp::validate(path)) {
    fan::webp::get_image_size(path, &size);
  } else {
    int x, y, comp;
    ::stbi_info(path.data(), &x, &y, &comp);
    size = {static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y)};
  }
  
  __fan_internal_image_list[nr].size = size;
  __fan_internal_image_list[nr].image_settings = fan::graphics::format_converter::image_vulkan_to_global(p);
  __fan_internal_image_list[nr].image_path = path;

  std::string path_str(path);
  std::jthread([this, path_str, nr, p, on_gpu_uploaded]() {
    auto owned = fan::image::load_owned(path_str);
    if (!owned.valid()) { return; }

    std::lock_guard lock(async_image_mutex);
    pending_image_uploads.emplace_back(
      fan::vulkan::decoded_image_payload_t{
        .filename = path_str,
        .size = owned.size,
        .channels = owned.channels,
        .raw_pixels = std::vector<std::uint8_t>(owned.data.get(), owned.data.get() + owned.data_size),
        .properties = fan::graphics::format_converter::image_vulkan_to_global(p),
        .target_nr = nr
      },
      on_gpu_uploaded
    );
  }).detach();

  return nr;
}

void fan::vulkan::context_t::process_async_image_uploads() {
  std::vector<std::pair<fan::vulkan::decoded_image_payload_t, std::function<void(const fan::vulkan::decoded_image_payload_t&)>>> ready_uploads;
  
  {
      std::lock_guard<std::mutex> lock(async_image_mutex);
      if (pending_image_uploads.empty()) return;
      ready_uploads.swap(pending_image_uploads);
  }

  for (auto& [payload, callback] : ready_uploads) {
      fan::image::info_t ii;
      ii.data = payload.raw_pixels.data();
      ii.size = payload.size;
      ii.channels = payload.channels;
      
      fan::vulkan::context_t::image_t& image = image_get(payload.target_nr);
      
      auto lp = fan::graphics::format_converter::image_global_to_vulkan(payload.properties);
      int src_channels = ii.channels;
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

      VkDeviceSize image_size_bytes = ii.size.multiply() * format_channels;
      auto alloc = staging_ring_buffer.allocate(image_size_bytes);
      image.staging_size = image_size_bytes;

      const std::uint8_t* src = static_cast<const std::uint8_t*>(ii.data);
      std::uint8_t* dst = static_cast<std::uint8_t*>(alloc.mapped_ptr);
      std::uint64_t pixel_count = ii.size.multiply();

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
        fan::throw_error("image_load: unsupported channel/format combination");
      }

      fan::vulkan::image_create(
        *this,
        ii.size,
        lp.format,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        image.image_index,
        image.image_allocation
      );
      image.image_view = create_image_view(image.image_index, lp.format, VK_IMAGE_ASPECT_COLOR_BIT);
      create_texture_sampler(image.sampler, lp);

      VkCommandBuffer cmd = begin_async_transfer_commands();
      transition_image_layout_cmd(cmd, image.image_index, lp.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      copy_buffer_to_image_cmd(cmd, alloc.buffer, image.image_index, lp.format, ii.size, alloc.offset);
      transition_image_layout_cmd(cmd, image.image_index, lp.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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

      if (callback) callback(payload);
  }
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
  if (fan::image::load(path, &image_info, 0, callers_path)) {
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
  auto format_channels = get_image_multiplier(p.format);
  auto src_channels = image_info.channels <= 0 ? format_channels : image_info.channels;
  VkDeviceSize image_size = image_info.size.multiply() * format_channels;

  fan::vulkan::context_t::image_t& image = image_get(nr);
  auto& image_data = __fan_internal_image_list[nr];

  auto new_settings = fan::graphics::format_converter::image_vulkan_to_global(p);
  bool image_was_null = image.image_index == VK_NULL_HANDLE;
  bool recreate_image = image.image_index != VK_NULL_HANDLE && (
    image_data.size != image_info.size ||
    image_data.image_settings.format != new_settings.format
  );
  if (recreate_image) {
  #if defined(FAN_GUI)
    if (image.gui_descriptor_set != VK_NULL_HANDLE) {
      ImGui_ImplVulkan_RemoveTexture(image.gui_descriptor_set);
      image.gui_descriptor_set = VK_NULL_HANDLE;
      image.gui_image_view = VK_NULL_HANDLE;
      image.gui_sampler = VK_NULL_HANDLE;
    }
  #endif
    if (image.sampler != VK_NULL_HANDLE) {
      vkDestroySampler(device, image.sampler, nullptr);
      image.sampler = VK_NULL_HANDLE;
    }
    if (image.image_view != VK_NULL_HANDLE && image.owns_image_view) {
      vkDestroyImageView(device, image.image_view, nullptr);
      image.image_view = VK_NULL_HANDLE;
    }
    if (image.image_index != VK_NULL_HANDLE && image.owns_image) {
      vmaDestroyImage(allocator, image.image_index, image.image_allocation);
      image.image_index = VK_NULL_HANDLE;
      image.image_allocation = VK_NULL_HANDLE;
    }
  }

  image_data.size = image_info.size;
  image_data.image_settings = new_settings;

  VkDeviceSize image_size_bytes = image_size;

  if (image.staging_buffer == VK_NULL_HANDLE || image.staging_size < image_size_bytes) {
    if (image.staging_buffer != VK_NULL_HANDLE) {
      destroy_buffer(image.staging_buffer, image.staging_allocation);
    }
    create_buffer(
      image_size_bytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      image.staging_buffer,
      image.staging_allocation
    );
    image.staging_size = image_size_bytes;
  }

  if (image.image_index == VK_NULL_HANDLE) {
    fan::vulkan::image_create(
      *this,
      image_info.size,
      p.format,
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      image.image_index,
      image.image_allocation
    );

    image.image_view = create_image_view(image.image_index, p.format, VK_IMAGE_ASPECT_COLOR_BIT);
    create_texture_sampler(image.sampler, p);
  }

  vmaMapMemory(allocator, image.staging_allocation, &image.data);

  const std::uint8_t* src = static_cast<const std::uint8_t*>(image_info.data);
  std::uint8_t* dst = static_cast<std::uint8_t*>(image.data);
  std::uint64_t pixel_count = image_info.size.multiply();

  if (src_channels == format_channels) {
    std::memcpy(dst, src, image_size_bytes);
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
    vmaUnmapMemory(allocator, image.staging_allocation);
    fan::throw_error("image_reload: unsupported channel/format combination");
  }

  vmaUnmapMemory(allocator, image.staging_allocation);

  transition_image_layout(
    image.image_index,
    p.format,
    (recreate_image || image_was_null) ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
  );
  copy_buffer_to_image(image.staging_buffer, image.image_index, p.format, image_info.size);
  transition_image_layout(image.image_index, p.format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
void fan::vulkan::context_t::image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) {
  image_reload(nr, image_info, fan::vulkan::context_t::image_load_properties_t());
}
void fan::vulkan::context_t::image_reload(fan::graphics::image_nr_t nr, fan::str_view_t path, const fan::vulkan::context_t::image_load_properties_t& p, const std::source_location& callers_path) {
  fan::image::info_t image_info;
  if (fan::image::load(path, &image_info, 0, callers_path)) {
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

  ii.data = pixels;
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
  img.owns_image = false;
  img.owns_image_view = false;
  create_texture_sampler(img.sampler, {});

  return img_nr;
}
//-----------------------------image-----------------------------

      //-----------------------------camera-----------------------------
fan::vulkan::context_t::image_load_properties_t fan::graphics::format_converter::image_global_to_vulkan(const fan::graphics::image_load_properties_t& p) {
  return fan::vulkan::context_t::image_load_properties_t {
    .visual_output = global_to_vulkan_address_mode(p.visual_output),
    .format = global_to_vulkan_format(p.format),
    .min_filter = global_to_vulkan_filter(p.min_filter),
    .mag_filter = global_to_vulkan_filter(p.mag_filter),
  };
}
void fan::vulkan::image_create(
  const fan::vulkan::context_t& context,
  const fan::vec2ui& image_size,
  VkFormat format,
  VkImageTiling tiling,
  VkImageUsageFlags usage,
  VkMemoryPropertyFlags properties,
  VkImage& image,
  VmaAllocation& allocation
) {
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

  VmaAllocationCreateInfo allocation_info{};
  allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
  allocation_info.requiredFlags = properties;
  if (vmaCreateImage(context.allocator, &imageInfo, &allocation_info, &image, &allocation, nullptr) != VK_SUCCESS) {
    fan::throw_error("failed to create image!");
  }
}

void fan::vulkan::vai_t::open(fan::vulkan::context_t& context, const properties_t& p) {
  old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
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
    vmaDestroyImage(context.allocator, image, memory);
    image = 0;
    memory = 0;
  }
  old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
}