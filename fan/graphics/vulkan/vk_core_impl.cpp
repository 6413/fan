module;

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

fan::vulkan::context_t::descriptor_t::descriptor_t(descriptor_t&& other) noexcept {
  *this = std::move(other);
}
fan::vulkan::context_t::descriptor_t& fan::vulkan::context_t::descriptor_t::operator=(descriptor_t&& other) noexcept {
  m_properties = std::move(other.m_properties);
  m_layout = other.m_layout;
  std::memcpy(m_descriptor_set, other.m_descriptor_set, sizeof(m_descriptor_set));
  other.m_layout = VK_NULL_HANDLE;
  std::memset(other.m_descriptor_set, 0, sizeof(other.m_descriptor_set));
  return *this;
}
void fan::vulkan::context_t::descriptor_t::open(fan::vulkan::context_t& context, const properties_t& properties) {
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
void fan::vulkan::context_t::descriptor_t::close(fan::vulkan::context_t& context) {
  if (m_layout == VK_NULL_HANDLE) { return; }
  vkDestroyDescriptorSetLayout(context.device, m_layout, 0);
  m_layout = VK_NULL_HANDLE;
}
void fan::vulkan::context_t::descriptor_t::update(
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

void fan::vulkan::context_t::pipeline_t::open(fan::vulkan::context_t& context, const properties_t& p) {
  properties = p;

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
  depthStencil.depthTestEnable = p.enable_depth_test;
  depthStencil.depthWriteEnable = p.enable_depth_test;
  depthStencil.depthCompareOp = p.depth_test_compare_op;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_NO_OP;
  colorBlending.attachmentCount = static_cast<std::uint32_t>(p.color_blend_attachments.size());
  colorBlending.pAttachments = p.color_blend_attachments.data();
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
  dynamicState.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = static_cast<std::uint32_t>(p.descriptor_layouts.size());
  pipelineLayoutInfo.pSetLayouts = p.descriptor_layouts.data();

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
  pipelineInfo.renderPass = p.render_pass == VK_NULL_HANDLE ?
    context.render_pass : p.render_pass;
  pipelineInfo.subpass = p.subpass;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

  shader_nr = p.shader;
  if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
    fan::throw_error("failed to create graphics pipeline");
  }
}
void fan::vulkan::context_t::pipeline_t::close(fan::vulkan::context_t& context) {
  vkDestroyPipeline(context.device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(context.device, m_layout, nullptr);
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
