// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
	#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

//#define loco_window
#define loco_context

#define loco_compute_shader
#include _FAN_PATH(graphics/loco.h)

struct pile_t {
	loco_t loco;
};

int main() {

	pile_t* pile = new pile_t;
	auto context = pile->loco.get_context();

	std::array<fan::vulkan::write_descriptor_set_t, 1> ds_properties{ 0 };

	fan::vulkan::descriptor_t<ds_properties.size()> descriptor;

	uint32_t buffer_size = 10000;

	fan::vulkan::core::memory_t memory;

	fan::vulkan::core::create_buffer(
		context,
		buffer_size,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		//VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // ?
		// faster ^? depends about buffer size maxMemoryAllocationCount maybe
		memory.buffer,
		memory.device_memory
	);

	ds_properties[0].binding = 0;
	ds_properties[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	ds_properties[0].flags = VK_SHADER_STAGE_COMPUTE_BIT;
	ds_properties[0].range = VK_WHOLE_SIZE;
	ds_properties[0].buffer = memory.buffer;
	ds_properties[0].dst_binding = 0;

	descriptor.open(context, pile->loco.descriptor_pool.m_descriptor_pool, ds_properties);
	descriptor.update(context);

	loco_t::compute_shader_t::properties_t p;
	p.shader.path = "compute_shader.spv";
	p.descriptor.layouts = &descriptor.m_layout;
	p.descriptor.sets = descriptor.m_descriptor_set;

	loco_t::compute_shader_t compute_shader(&pile->loco, p);

	void* data;
	fan::vulkan::validate(vkMapMemory(pile->loco.get_context()->device, memory.device_memory, 0, 10000, 0, (void**)&data));

	context->begin_compute_shader();

	compute_shader.execute(&pile->loco, fan::vec3(5, 1, 1));

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