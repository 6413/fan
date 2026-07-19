module;

#if defined(FAN_2D)

#if defined(fan_platform_windows)
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(fan_platform_unix)
#define VK_USE_PLATFORM_XLIB_KHR
#endif
#if defined(FAN_GUI)
  #include <fan/imgui/imgui_impl_vulkan.h>
#endif
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

module fan.graphics.vulkan.core;

import std;

import fan.print;
import fan.print.error;

void fan::vulkan::context_t::flush_deletion_queues() {
  vkDeviceWaitIdle(device);
  for (std::uint32_t i = 0; i < max_frames_in_flight; ++i) {
    get_current_deletion_queue(i).flush();
  }
  pending_deletion_queue.flush();
  main_deletion_queue.flush();
}

fan::vulkan::descriptor_t::descriptor_t(descriptor_t&& other) noexcept {
  *this = std::move(other);
}
fan::vulkan::descriptor_t& fan::vulkan::descriptor_t::operator=(descriptor_t&& other) noexcept {
  m_properties = std::move(other.m_properties);
  m_layout = other.m_layout;
  std::memcpy(m_descriptor_set, other.m_descriptor_set, sizeof(m_descriptor_set));
  other.m_layout = VK_NULL_HANDLE;
  std::memset(other.m_descriptor_set, 0, sizeof(other.m_descriptor_set));
  return *this;
}
void fan::vulkan::descriptor_t::open(fan::vulkan::context_t& context, const properties_t& properties) {
  m_properties = properties;
  std::vector<VkDescriptorSetLayoutBinding> uboLayoutBinding(properties.size());
  for (std::uint16_t i = 0; i < properties.size(); ++i) {
    uboLayoutBinding[i].binding = properties[i].binding;
    uboLayoutBinding[i].descriptorCount = m_properties[i].descriptor_count;
    if (uboLayoutBinding[i].descriptorCount == 0) {
      uboLayoutBinding[i].descriptorCount = m_properties[i].use_image ?
        max_textures : 1;
    }
    uboLayoutBinding[i].descriptorType = properties[i].type;
    uboLayoutBinding[i].stageFlags = properties[i].flags;
  }

  std::vector<VkDescriptorBindingFlags> binding_flags(uboLayoutBinding.size());
  bool has_update_after_bind = false;
  for (std::uint32_t i = 0; i < (std::uint32_t)uboLayoutBinding.size(); ++i) {
    bool is_bindless =
      uboLayoutBinding[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
      uboLayoutBinding[i].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
      uboLayoutBinding[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
      uboLayoutBinding[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
      uboLayoutBinding[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    binding_flags[i] = is_bindless ?
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
      VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0;
    has_update_after_bind |= is_bindless;
  }

  VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{};
  binding_flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
  binding_flags_info.bindingCount = (std::uint32_t)binding_flags.size();
  binding_flags_info.pBindingFlags = binding_flags.data();

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.pNext = has_update_after_bind ? &binding_flags_info : nullptr;
  layoutInfo.flags = has_update_after_bind ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0;
  layoutInfo.bindingCount = std::size(uboLayoutBinding);
  layoutInfo.pBindings = uboLayoutBinding.data();

  validate(vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &m_layout));

  std::array<VkDescriptorSetLayout, max_frames_in_flight> layouts;
  for (std::uint32_t i = 0; i < max_frames_in_flight; ++i) {
    layouts[i] = m_layout;
  }
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = context.descriptor_pool.m_descriptor_pool;
  allocInfo.descriptorSetCount = layouts.size();
  allocInfo.pSetLayouts = layouts.data();

  validate(vkAllocateDescriptorSets(context.device, &allocInfo, m_descriptor_set));
}
void fan::vulkan::descriptor_t::close(fan::vulkan::context_t& context) {
  if (m_layout == VK_NULL_HANDLE) { return; }
  vkDestroyDescriptorSetLayout(context.device, m_layout, 0);
  m_layout = VK_NULL_HANDLE;
}
void fan::vulkan::descriptor_t::update(
  fan::vulkan::context_t& context,
  std::uint32_t n,
  std::uint32_t begin,
  std::uint32_t texture_n,
  std::uint32_t texture_begin
) {
  m_buffer_infos.resize(n);
  m_descriptor_writes.resize(n);

  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint32_t j = begin + i;
    auto& write = m_descriptor_writes[i];
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptor_set[context.current_frame];
    write.dstBinding = m_properties[j].dst_binding;
    write.dstArrayElement = texture_begin;
    write.descriptorType = m_properties[j].type;
    if (m_properties[j].use_image) {
      std::uint32_t descriptor_n = texture_n;
      if (texture_n == max_textures && m_properties[j].descriptor_count != 0) {
        descriptor_n = m_properties[j].descriptor_count;
      }
      if (texture_begin + descriptor_n > m_properties[j].image_infos.size()) {
        fan::throw_error("descriptor image update out of range");
      }
      write.descriptorCount = descriptor_n;
      write.pImageInfo = m_properties[j].image_infos.data() + texture_begin;
    }
    else {
      m_buffer_infos[i].buffer = m_properties[j].buffer;
      m_buffer_infos[i].offset = 0;
      m_buffer_infos[i].range = m_properties[j].range;
      write.descriptorCount = 1;
      write.pBufferInfo = &m_buffer_infos[i];
    }
  }

  vkUpdateDescriptorSets(context.device, n, m_descriptor_writes.data(), 0, nullptr);
}

void fan::vulkan::context_t::descriptor_pool_t::open(fan::vulkan::context_t& context) {
  VkDescriptorPoolSize pool_sizes[] = {
    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024 },
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1024 },
    { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,

    #if defined(FAN_GUI)
      IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE +
    #endif
      fan::vulkan::max_textures * 32
    },
    { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 64 },
  };
  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
  pool_info.maxSets = 0;
  for (VkDescriptorPoolSize& pool_size : pool_sizes)
    pool_info.maxSets += max_frames_in_flight * pool_size.descriptorCount;
  pool_info.poolSizeCount = (std::uint32_t)std::size(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;
  fan::vulkan::validate(vkCreateDescriptorPool(context.device, &pool_info, nullptr, &m_descriptor_pool));
}
void fan::vulkan::context_t::descriptor_pool_t::close(fan::vulkan::context_t& context) {
  vkDestroyDescriptorPool(context.device, m_descriptor_pool, nullptr);
}

void fan::vulkan::pipeline_t::open(fan::vulkan::context_t& context, const properties_t& p) {
  properties = p;

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = static_cast<std::uint32_t>(properties.descriptor_layouts.size());
  pipelineLayoutInfo.pSetLayouts = properties.descriptor_layouts.data();

  VkPushConstantRange push_constant;
  push_constant.offset = 0;
  push_constant.size = properties.push_constants_size;
  push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  pipelineLayoutInfo.pPushConstantRanges = &push_constant;
  pipelineLayoutInfo.pushConstantRangeCount = properties.push_constants_size == 0 ? 0 : 1;

  if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &m_layout) != VK_SUCCESS) {
    fan::throw_error("failed to create pipeline layout!");
  }

  auto shader_ref = context.shaders.shader_get(properties.shader);
  shader_nr = properties.shader;

  m_shaders[0] = VK_NULL_HANDLE;
  m_shaders[1] = VK_NULL_HANDLE;

  bool has_stages[2] = { false, false };
  int ref_indices[2] = { -1, -1 };

  for (int i = 0; i < 2; ++i) {
    VkShaderStageFlagBits stage = (i == 0) ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
    for (int j = 0; j < 3; ++j) {
      if (shader_ref.shader_stages[j].stage == stage && shader_ref.shader_stages[j].module != VK_NULL_HANDLE) {
        has_stages[i] = true;
        ref_indices[i] = j;
        break;
      }
    }
  }

  std::uint32_t shader_count = (has_stages[0] ? 1 : 0) + (has_stages[1] ? 1 : 0);
  VkShaderCreateInfoEXT shader_infos[2] = {};
  bool link = shader_count > 1;

  for (int i = 0, idx = 0; i < 2; ++i) {
    if (!has_stages[i]) { continue; }
    auto& spirv = shader_ref.spirv_stages[ref_indices[i]];
    if (spirv.empty()) { continue; }

    auto& info = shader_infos[idx];
    info.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
    info.flags = link ? VK_SHADER_CREATE_LINK_STAGE_BIT_EXT : 0;
    info.stage = (i == 0) ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
    info.nextStage = (i == 0 && has_stages[1]) ? VK_SHADER_STAGE_FRAGMENT_BIT : 0;
    info.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
    info.pCode = spirv.data();
    info.codeSize = spirv.size() * sizeof(std::uint32_t);
    info.pName = "main";
    info.setLayoutCount = (std::uint32_t)properties.descriptor_layouts.size();
    info.pSetLayouts = properties.descriptor_layouts.data();
    info.pushConstantRangeCount = properties.push_constants_size ? 1 : 0;
    info.pPushConstantRanges = properties.push_constants_size ? &push_constant : nullptr;
    ++idx;
  }

  if (shader_count > 0) {
    if (fan_vkCreateShadersEXT(context.device, shader_count, shader_infos, nullptr, m_shaders) != VK_SUCCESS) {
      fan::throw_error("failed to create shader objects");
    }
  }
}
void fan::vulkan::pipeline_t::close(fan::vulkan::context_t& context) {
  if (m_shaders[0]) { fan_vkDestroyShaderEXT(context.device, m_shaders[0], nullptr); }
  if (m_shaders[1]) { fan_vkDestroyShaderEXT(context.device, m_shaders[1], nullptr); }
  if (m_layout) { vkDestroyPipelineLayout(context.device, m_layout, nullptr); }
}

namespace fan::vulkan::core {
  std::uint32_t get_draw_mode(std::uint8_t draw_mode) {
    switch (draw_mode) {
    case fan::graphics::primitive_topology_t::points:
      return fan::vulkan::context_t::primitive_topology_t::points;
    case fan::graphics::primitive_topology_t::lines:
      return fan::vulkan::context_t::primitive_topology_t::lines;
    case fan::graphics::primitive_topology_t::line_strip:
      return fan::vulkan::context_t::primitive_topology_t::line_strip;
    case fan::graphics::primitive_topology_t::triangles:
      return fan::vulkan::context_t::primitive_topology_t::triangles;
    case fan::graphics::primitive_topology_t::triangle_strip:
      return fan::vulkan::context_t::primitive_topology_t::triangle_strip;
    case fan::graphics::primitive_topology_t::triangle_fan:
      return fan::vulkan::context_t::primitive_topology_t::triangle_fan;
    case fan::graphics::primitive_topology_t::lines_with_adjacency:
      return fan::vulkan::context_t::primitive_topology_t::lines_with_adjacency;
    case fan::graphics::primitive_topology_t::line_strip_with_adjacency:
      return fan::vulkan::context_t::primitive_topology_t::line_strip_with_adjacency;
    case fan::graphics::primitive_topology_t::triangles_with_adjacency:
      return fan::vulkan::context_t::primitive_topology_t::triangles_with_adjacency;
    case fan::graphics::primitive_topology_t::triangle_strip_with_adjacency:
      return fan::vulkan::context_t::primitive_topology_t::triangle_strip_with_adjacency;
    default:
      fan::throw_error("invalid draw mode");
      return -1;
    }
  }
}


#endif