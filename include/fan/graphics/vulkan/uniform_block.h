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

				void open(fan::vulkan::context_t* context, open_properties_t op_ = open_properties_t()) {
					common.open(context, [context, this] () {
						for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {

							uint64_t src = common.m_min_edit;
							uint64_t dst = common.m_max_edit;

							// UPDATE BUFFER HERE

							void* data;
							validate(vkMapMemory(context->device, common.memory[i].device_memory, 0, dst - src, 0, &data));
							//data += src; ??
							memcpy(data, buffer, dst - src);
							// unnecessary?
							vkUnmapMemory(context->device, common.memory[i].device_memory);
							//fan::vulkan::core::edit_glbuffer(context, write_queue[it]->m_vbo, buffer, src, dst - src, fan::opengl::GL_UNIFORM_BUFFER);

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

					for (size_t i = 0; i < buffer_count; i++) {
						vkDestroyBuffer(context->device, common.memory[i].buffer, nullptr);
						vkFreeMemory(context->device, common.memory[i].device_memory, nullptr);
					}
				}

				uint32_t size() const {
					return common.m_size / sizeof(type_t);
				}

				void push_ram_instance(fan::vulkan::context_t* context, const type_t& data) {
					std::memmove(&buffer[m_size], (void*)&data, sizeof(type_t));
					m_size += sizeof(type_t);
				}

				type_t* get_instance(fan::vulkan::context_t* context, uint32_t i) {
					return (type_t*)&buffer[i * sizeof(type_t)];
				}

				void edit_instance(fan::vulkan::context_t* context, uint32_t i, auto member, auto value) {
					#if fan_debug >= fan_debug_low
					/* if (i * sizeof(type_t) >= common.m_size) {
						 fan::throw_error("uninitialized access");
					 }*/
					#endif
					((type_t*)buffer)[i].*member = value;
				}
				// for copying whole thing
				void copy_instance(fan::vulkan::context_t* context, uint32_t i, type_t* instance) {
					#if fan_debug >= fan_debug_low
					if (i * sizeof(type_t) >= common.m_size) {
						fan::throw_error("uninitialized access");
					}
					#endif
					std::memmove(buffer + i * sizeof(type_t), instance, sizeof(type_t));
					common.edit(
						loco->get_context(),
						&loco->m_write_queue,
						i * sizeof(instance_t),
						i * sizeof(instance_t) + sizeof(instance_t)
					);

				}

				memory_common_t common;
				uint8_t buffer[element_size * sizeof(type_t)];
				uint32_t m_size;
			};
		}
	}
}