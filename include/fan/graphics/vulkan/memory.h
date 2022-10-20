#pragma once

namespace fan {
	namespace vulkan {
		namespace core {

			struct memory_write_queue_t {

				memory_write_queue_t() = default;

				using memory_edit_cb_t = fan::function_t<void()>;


				#include "memory_bll_settings.h"
			protected:
				#include _FAN_PATH(BLL/BLL.h)
				write_queue_t write_queue;
			public:

				using nr_t = write_queue_NodeReference_t;

				nr_t push_back(const memory_edit_cb_t& cb) {
					auto nr = write_queue.NewNodeLast();
					write_queue[nr].cb = cb;
					return nr;
				}

				void process(fan::vulkan::context_t* context) {
					auto it = write_queue.GetNodeFirst();
					while (it != write_queue.dst) {
						write_queue.StartSafeNext(it);
						write_queue[it].cb();

						it = write_queue.EndSafeNext();
					}

					write_queue.Clear();
				}

				void erase(nr_t node_reference) {
					write_queue.unlrec(node_reference);
				}

				void clear() {
					write_queue.Clear();
				}
			};

			struct memory_t {
				VkBuffer buffer;
				VkDeviceMemory device_memory;
			};

			struct memory_common_t {
				static constexpr auto buffer_count = fan::vulkan::MAX_FRAMES_IN_FLIGHT;

				memory_t memory[buffer_count];

				memory_write_queue_t::memory_edit_cb_t write_cb;

				void open(fan::vulkan::context_t* context, const memory_write_queue_t::memory_edit_cb_t& cb) {

					write_cb = cb;

					queued = false;

					m_min_edit = 0xFFFFFFFFFFFFFFFF;
					//context <- uniform_block <-> uniform_write_queue <- loco
					m_max_edit = 0x00000000;
				}
				void close(fan::vulkan::context_t* context, memory_write_queue_t* queue) {
					if (is_queued()) {
						queue->erase(m_edit_index);
					}

					for (uint32_t i = 0; i < fan::vulkan::MAX_FRAMES_IN_FLIGHT; ++i) {
						vkDestroyBuffer(context->device, memory[i].buffer, nullptr);
						vkFreeMemory(context->device, memory[i].device_memory, nullptr);
					}
				}

				bool is_queued() const {
					return queued;
				}

				void edit(fan::vulkan::context_t* context, memory_write_queue_t* queue, uint32_t begin, uint32_t end) {

					m_min_edit = fan::min(m_min_edit, begin);
					m_max_edit = fan::max(m_max_edit, end);

					if (is_queued()) {
						return;
					}
					queued = true;
					m_edit_index = queue->push_back(write_cb);

					// context->process();
				}

				void on_edit(fan::vulkan::context_t* context) {
					reset_edit();
				}

				void reset_edit() {
					m_min_edit = 0xFFFFFFFFFFFFFFFF;
					m_max_edit = 0x00000000;

					queued = false;
				}

				memory_write_queue_t::nr_t m_edit_index;

				uint64_t m_min_edit;
				uint64_t m_max_edit;

				bool queued;
			};

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

			void createBuffer(fan::vulkan::context_t* context, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
				VkBufferCreateInfo bufferInfo{};
				bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
				bufferInfo.size = size;
				bufferInfo.usage = usage;
				bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

				validate(vkCreateBuffer(context->device, &bufferInfo, nullptr, &buffer));

				VkMemoryRequirements memRequirements;
				vkGetBufferMemoryRequirements(context->device, buffer, &memRequirements);

				VkMemoryAllocateInfo allocInfo{};
				allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
				allocInfo.allocationSize = memRequirements.size;
				allocInfo.memoryTypeIndex = findMemoryType(context, memRequirements.memoryTypeBits, properties);

				validate(vkAllocateMemory(context->device, &allocInfo, nullptr, &bufferMemory));

				validate(vkBindBufferMemory(context->device, buffer, bufferMemory, 0));
			}

		}
	}
}