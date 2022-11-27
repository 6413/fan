// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 3
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

//#define loco_window
#define loco_context

//#define loco_line
#define loco_compute_shader
#include _FAN_PATH(graphics/loco.h)

struct pile_t {
	loco_t loco;
};

int main() {

	pile_t* pile = new pile_t;

	loco_t::compute_shader_t::properties_t p;
	p.shader.path = "compute_shader.spv";

	loco_t::compute_shader_t compute_shader(&pile->loco, p);

	void* data;
	fan::vulkan::validate(vkMapMemory(pile->loco.get_context()->device, compute_shader.device_memory, 0, 10000, 0, (void**)&data));

	auto context = pile->loco.get_context();

	context->begin_compute_shader();
	compute_shader.execute(&pile->loco, fan::vec3(5, 1, 1));


	VkBufferMemoryBarrier barrier{};
	barrier.buffer = compute_shader.buffer;
	barrier.size = VK_WHOLE_SIZE;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	vkCmdPipelineBarrier(
		context->commandBuffers[context->currentFrame], 
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
		VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 
		0, 0, nullptr, 1, &barrier, 0, nullptr
	);

	context->end_compute_shader();

	compute_shader.wait_finish(&pile->loco);

	for (uint32_t i = 0; i < 5 * 1 * 1 * 128 * 1 * 1; ++i) {
		if (i && ((uint32_t*)data)[i] == 0) {
			break;
		}
		fan::print(i, ((uint32_t*)data)[i]);
	}


	context->begin_compute_shader();
	compute_shader.execute(&pile->loco, fan::vec3(5, 1, 1));

	vkCmdPipelineBarrier(
		context->commandBuffers[context->currentFrame], 
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
		VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 
		0, 0, nullptr, 1, &barrier, 0, nullptr
	);

	context->end_compute_shader();

	compute_shader.wait_finish(&pile->loco);

	for (uint32_t i = 0; i < 5 * 1 * 1 * 128 * 1 * 1; ++i) {
		if (i && ((uint32_t*)data)[i] == 0) {
			break;
		}
		fan::print(i, ((uint32_t*)data)[i]);
	}


	return 0;
}