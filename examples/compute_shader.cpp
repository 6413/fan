// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 3
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context

//#define loco_line
#define loco_compute_shader
#include _FAN_PATH(graphics/loco.h)

struct pile_t {
  loco_t loco;
};

int main() {

  pile_t* pile = new pile_t;

  void* data;

  pile->loco.draw_queue = [&] {
    auto context = pile->loco.get_context();
    auto cmd = context->commandBuffers[context->currentFrame];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pile->loco.compute_shader.m_pipeline);

    vkCmdBindDescriptorSets(
			cmd,
			VK_PIPELINE_BIND_POINT_COMPUTE,
			pile->loco.compute_shader.m_pipeline_layout,
			0,
			1,
			pile->loco.compute_shader.m_descriptor.m_descriptor_set,
			0,
			nullptr
		);

    vkCmdDispatch(cmd, 5, 1, 1);

    VkBufferMemoryBarrier barrier{};
    barrier.buffer = pile->loco.compute_shader.buffer;
    barrier.size = VK_WHOLE_SIZE;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);

    fan::vulkan::validate(vkMapMemory(context->device, pile->loco.compute_shader.device_memory, 0, 10000, 0, (void**)&data));
  };

  pile->loco.loop([&] {

  });

  vkDeviceWaitIdle(pile->loco.get_context()->device);

  for (uint32_t i = 1; i; ++i) {
    if (((uint32_t*)data)[i] == 0) {
      break;
    }
    fan::print(i, ((uint32_t*)data)[i]);
  }

  return 0;
}