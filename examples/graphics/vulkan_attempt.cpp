#include <fan/pch.h>

struct push_constants_t {
  uint32_t texture_id;
  uint32_t camera_id;
};

int main() {
  fan::graphics::engine_t engine{{
    .window_flags=fan::window_t::flags::vulkan
  }};
  fan::vulkan::context_t::shader_t render_fullscreen_shader;

  fan::vulkan::core::memory_write_queue_t m_write_queue;

  render_fullscreen_shader.open(engine.vk_context, &m_write_queue);
  render_fullscreen_shader.set_vertex(
    engine.vk_context,
    "shaders/vulkan/2D/objects/loco_fbo.vert",
    #include <shaders/vulkan/2D/objects/loco_fbo.vert>
  );
  render_fullscreen_shader.set_fragment(
    engine.vk_context,
    "shaders/vulkan/2D/objects/loco_fbo.frag",
    #include <shaders/vulkan/2D/objects/loco_fbo.frag>
  );

  fan::vulkan::pipeline_t::properties_t pipeline_p;
  pipeline_p.descriptor_layout_count = 0;
  pipeline_p.descriptor_layout = 0;
  pipeline_p.shader = &render_fullscreen_shader;
  pipeline_p.push_constants_size = sizeof(push_constants_t);
  pipeline_p.subpass = 0;
  VkDescriptorImageInfo imageInfo{};

  VkPipelineColorBlendAttachmentState color_blend_attachment[2]{};
  color_blend_attachment[0].colorWriteMask =
    VK_COLOR_COMPONENT_R_BIT | 
    VK_COLOR_COMPONENT_G_BIT | 
    VK_COLOR_COMPONENT_B_BIT | 
    VK_COLOR_COMPONENT_A_BIT
    ;
  color_blend_attachment[0].blendEnable = VK_TRUE;
  color_blend_attachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  color_blend_attachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  color_blend_attachment[0].colorBlendOp = VK_BLEND_OP_ADD;
  color_blend_attachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  color_blend_attachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  color_blend_attachment[0].alphaBlendOp = VK_BLEND_OP_ADD;

  color_blend_attachment[1] = color_blend_attachment[0];
  pipeline_p.color_blend_attachment_count = std::size(color_blend_attachment);
  pipeline_p.color_blend_attachment = color_blend_attachment;

  engine.vk_context.render_fullscreen_pl.open(engine.vk_context, pipeline_p);
  int f = 0;
  engine.loop([&] {
    fan_ev_timer_loop(1000, fan::print(1.0f / engine.delta_time););
    if (f == 1) {
      engine.set_target_fps(0);
    }
    f++;
    m_write_queue.process(engine.vk_context);
    engine.vk_context.begin_render(&gloco->window, 0);
    engine.vk_context.end_render(&gloco->window);
  });
  return 0;
}