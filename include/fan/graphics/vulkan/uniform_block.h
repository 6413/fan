#pragma once

namespace fan {
	namespace vulkan {
		namespace core {

			template <typename type_t, uint32_t element_size>
			struct uniform_block_t {

				static constexpr auto buffer_count = fan::vulkan::MAX_FRAMES_IN_FLIGHT;

				struct open_properties_t {
					open_properties_t() {}
				}op;

				using nr_t = uint8_t;
				using instance_id_t = uint8_t;

				uniform_block_t() = default;

				uniform_block_t(fan::vulkan::context_t* context, open_properties_t op_ = open_properties_t()) {
					open(context, op);
				}

				void open(fan::vulkan::context_t* context, open_properties_t op_ = open_properties_t()) {
					common.open(context, [context, this] () {
						for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; ++frame) {

							uint8_t* data;
							validate(vkMapMemory(context->device, common.memory[frame].device_memory, 0, element_size * sizeof(type_t), 0, (void**)&data));
							
							for (auto j : common.indices) {
								((type_t*)data)[j.i] = ((type_t*)buffer)[j.i];
							}
							// unnecessary? is necessary
							vkUnmapMemory(context->device, common.memory[frame].device_memory);

							common.on_edit(context);
						}
					});

					op = op_;

					m_size = 0;

					VkDeviceSize bufferSize = sizeof(type_t) * element_size;

					for (size_t i = 0; i < buffer_count; i++) {
						createBuffer(
							context,
							bufferSize, 
							VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
							VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
							common.memory[i].buffer,
							common.memory[i].device_memory
						);
					}
				}
				void close(fan::vulkan::context_t* context, memory_write_queue_t* queue) {
					common.close(context, queue);
				}

				uint32_t size() const {
					return m_size / sizeof(type_t);
				}

				void push_ram_instance(fan::vulkan::context_t* context, memory_write_queue_t* wq, const type_t& data) {
					std::memmove(&buffer[m_size], (void*)&data, sizeof(type_t));
					m_size += sizeof(type_t);
					common.edit(context, wq, {0, (unsigned char)(m_size / sizeof(type_t) - 1)});
				}

				//type_t* get_instance(fan::vulkan::context_t* context, uint32_t i) {
				//	return (type_t*)&buffer[i * sizeof(type_t)];
				//}

				void edit_instance(fan::vulkan::context_t* context, memory_write_queue_t* wq, uint32_t i, auto type_t::*member, auto value) {
					((type_t*)buffer)[i].*member = value;
					common.edit(context, wq, {0, (unsigned char)i});
				}

				//void edit_instance(fan::vulkan::context_t* context, memory_write_queue_t* wq, uint32_t i, uint64_t byte_offset, auto value) {
				//	// maybe xd
				//	*(decltype(value)*)(((uint8_t*)((type_t*)buffer)[i]) + byte_offset) = value;
				//	common.edit(context, {wq, (unsigned char)i});
				//}

				// nr_t is useless here
				memory_common_t<nr_t, instance_id_t> common;
				uint8_t buffer[element_size * sizeof(type_t)];
				uint32_t m_size;
			};
		}
	}
}