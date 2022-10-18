#pragma once

namespace fan {
	namespace vulkan {
		namespace core {

			template <typename type_t>
			struct ssbo_t	 {

				static constexpr auto buffer_count = fan::vulkan::MAX_FRAMES_IN_FLIGHT;

				struct open_properties_t {
					open_properties_t() {}
					// creates buffer and device
					uint64_t preallocate = 1;
				};

				void allocate(fan::vulkan::context_t* context, uint64_t size, uint32_t frame) {

					if (buffer.size() != 0) {
						vkDestroyBuffer(context->device, common.memory[frame].buffer, 0);
					}

					createBuffer(
						context,
						size,
						VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // ?
						common.memory[frame].buffer,
						common.memory[frame].device_memory
					);
				}

				void write(fan::vulkan::context_t* context, uint64_t src, uint64_t dst, uint32_t frame) {

					void* data;
					validate(vkMapMemory(context->device, common.memory[frame].device_memory, 0, dst - src, 0, &data));
					//data += src; ??
					memcpy(data, (uint8_t*)buffer.data(), dst - src);
					// unnecessary?
					vkUnmapMemory(context->device, common.memory[frame].device_memory);

					validate(vkMapMemory(context->device, common.memory[frame].device_memory, 0, dst - src, 0, &data));
					//data += src; ??
					// unnecessary?
					vkUnmapMemory(context->device, common.memory[frame].device_memory);

					common.on_edit(context);
				}

				void open(fan::vulkan::context_t* context, open_properties_t op = open_properties_t()) {
					common.open(context, [context, this] {
						for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
							write(context, common.m_min_edit, common.m_max_edit, i);
						}
					});

					if (op.preallocate == 0) {
						return;
					}

					buffer.reserve(op.preallocate);
					
					vram_capacity = buffer.capacity() * sizeof(type_t);

					for (size_t i = 0; i < buffer_count; i++) {
						allocate(context, vram_capacity, i);
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
					return buffer.size();
				}

				uint32_t add(fan::vulkan::context_t* context, uint32_t how_many) {
					uint32_t old_size = buffer.size();
					buffer.resize(buffer.size() + how_many);
					fan::print(buffer.capacity() * sizeof(type_t));
					if (vram_capacity < buffer.capacity() * sizeof(type_t)) {
						vram_capacity = buffer.capacity() * sizeof(type_t);
						for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
							allocate(context, vram_capacity, i);
							common.m_min_edit = 0;
							common.m_max_edit = old_size * sizeof(type_t);
						}
					}
					return buffer.size() - how_many;
				}

			/*	void push_ram_instance(fan::vulkan::context_t* context, const type_t& data) {
					buffer.push_back(data);
					if (vram_capacity != buffer.capacity() * sizeof(type_t)) {
						vram_capacity = buffer.capacity() * sizeof(type_t);
						for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
							allocate(context, vram_capacity, i);
							write(context, 0, vram_capacity, i);
						}
					}
				}*/

				type_t* get_instance(fan::vulkan::context_t* context, uint32_t i) {
					return &buffer[i];
				}

				/*void edit_instance(fan::vulkan::context_t* context, uint32_t i, auto member, auto value) {
					buffer[i].*member = value;
				}*/
				// for copying whole thing
				void copy_instance(fan::vulkan::context_t* context, memory_write_queue_t* write_queue, uint32_t i, type_t* instance) {
					buffer[i] = *instance;

					common.edit(
						context,
						write_queue,
						i * sizeof(type_t),
						i * sizeof(type_t) + sizeof(type_t)
					);
				}

				memory_common_t common;
				std::vector<type_t> buffer;
				uint64_t vram_capacity;
			};
		}
	}
}