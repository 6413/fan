#pragma once

namespace fan {
	namespace vulkan {
		namespace core {

			struct uniform_block_common_t;

			struct uniform_write_queue_t {
				void open() {
					write_queue.open();
				}
				void close() {
					write_queue.close();
				}

				uint32_t push_back(fan::vulkan::core::uniform_block_common_t* block);

				void process(fan::vulkan::context_t* context);

				void erase(uint32_t node_reference) {
					write_queue.erase(node_reference);
				}

				void clear() {
					write_queue.clear();
				}

			protected:
				bll_t<fan::vulkan::core::uniform_block_common_t*> write_queue;
			};

			struct memory_t {
				VkBuffer uniform_buffer;
				VkDeviceMemory device_memory;
			};

			struct uniform_block_common_t {
				static constexpr auto buffer_count = fan::vulkan::MAX_FRAMES_IN_FLIGHT;

				uint32_t m_size;

				memory_t memory[buffer_count];

				void open(fan::vulkan::context_t* context, uint64_t buffer_size) {

					m_edit_index = fan::uninitialized;

					m_min_edit = 0xffffffff;
					//context <- uniform_block <-> uniform_write_queue <- loco
					m_max_edit = 0x00000000;

					m_size = 0;
					m_buffer_size = buffer_size;
				}
				void close(fan::vulkan::context_t* context, uniform_write_queue_t* queue) {
					if (is_queued()) {
						queue->erase(m_edit_index);
					}
				}

				bool is_queued() const {
					return m_edit_index != fan::uninitialized;
				}

				void edit(fan::vulkan::context_t* context, uniform_write_queue_t* queue, uint32_t begin, uint32_t end) {

					m_min_edit = fan::min(m_min_edit, begin);
					m_max_edit = fan::max(m_max_edit, end);

					if (is_queued()) {
						return;
					}
					m_edit_index = queue->push_back(this);

					// context->process();
				}

				void on_edit(fan::vulkan::context_t* context) {
					reset_edit();
				}

				void reset_edit() {
					m_min_edit = 0xffffffff;
					m_max_edit = 0x00000000;

					m_edit_index = fan::uninitialized;
				}

				uint32_t m_edit_index;

				uint32_t m_min_edit;
				uint32_t m_max_edit;

				uint64_t m_buffer_size;
			};


			template <typename type_t, uint32_t element_size>
			struct uniform_block_t {

				static constexpr auto buffer_count = fan::vulkan::MAX_FRAMES_IN_FLIGHT;

				struct open_properties_t {
					open_properties_t() {}

					//uint32_t target = fan::opengl::GL_UNIFORM_BUFFER;
					//uint32_t usage = fan::opengl::GL_DYNAMIC_DRAW;
				}op;

				void open(fan::vulkan::context_t* context, open_properties_t op_ = open_properties_t()) {
					common.open(context, sizeof(type_t) * element_size);

					op = op_;

					VkDeviceSize bufferSize = sizeof(type_t) * element_size;

					for (size_t i = 0; i < buffer_count; i++) {
						createBuffer(
							context,
							bufferSize, 
							VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
							VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
							common.memory[i].uniform_buffer,
							common.memory[i].device_memory
						);
					}
				}
				void close(fan::vulkan::context_t* context, uniform_write_queue_t* queue) {
					common.close(context, queue);

					for (size_t i = 0; i < buffer_count; i++) {
						vkDestroyBuffer(context->device, common.memory[i].uniform_buffer, nullptr);
						vkFreeMemory(context->device, common.memory[i].device_memory, nullptr);
					}
				}

				uint32_t size() const {
					return common.m_size / sizeof(type_t);
				}

				void createBuffer(fan::vulkan::context_t* context, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
					VkBufferCreateInfo bufferInfo{};
					bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
					bufferInfo.size = size;
					bufferInfo.usage = usage;
					bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

					if (vkCreateBuffer(context->device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
						throw std::runtime_error("failed to create buffer!");
					}

					VkMemoryRequirements memRequirements;
					vkGetBufferMemoryRequirements(context->device, buffer, &memRequirements);

					VkMemoryAllocateInfo allocInfo{};
					allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
					allocInfo.allocationSize = memRequirements.size;
					allocInfo.memoryTypeIndex = findMemoryType(context, memRequirements.memoryTypeBits, properties);

					if (vkAllocateMemory(context->device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
						throw std::runtime_error("failed to allocate buffer memory!");
					}

					validate(vkBindBufferMemory(context->device, buffer, bufferMemory, 0));
				}

				uint32_t findMemoryType(fan::vulkan::context_t* context, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
					VkPhysicalDeviceMemoryProperties memProperties;
					vkGetPhysicalDeviceMemoryProperties(context->physicalDevice, &memProperties);

					for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
						if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
							return i;
						}
					}

					fan::throw_error("failed to find suitable memory type!");
				}

				void push_ram_instance(fan::vulkan::context_t* context, const type_t& data) {
					std::memmove(&buffer[common.m_size], (void*)&data, sizeof(type_t));
					common.m_size += sizeof(type_t);
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
				}

				uniform_block_common_t common;
				uint8_t buffer[element_size * sizeof(type_t)];
			};


			uint32_t uniform_write_queue_t::push_back(uniform_block_common_t* block) {
				return write_queue.push_back(block);
			}
		}
	}
}