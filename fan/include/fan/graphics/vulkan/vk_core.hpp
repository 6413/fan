#pragma once

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_vulkan

#include <fan/types/types.hpp>

#include <vulkan/vulkan.h>

#include <vulkan/vulkan_core.h>

#include <fan/graphics/shared.hpp>

namespace fan {

	namespace gpu_memory {

		enum class buffer_type {
			buffer,
			index,
			staging,
			texture,
			last
		};

		static uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
			VkPhysicalDeviceMemoryProperties memory_properties;
			vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

			for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
				if ((type_filter & (1 << i)) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
					return i;
				}
			}

			throw std::runtime_error("failed to find suitable memory type.");
		}

		static void create_buffer(VkDevice device, VkPhysicalDevice physical_device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {

			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size = size;
			bufferInfo.usage = usage;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
				throw std::runtime_error("failed to create buffer!");
			}

			VkMemoryRequirements memory_requirements;
			vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

			VkMemoryAllocateInfo alloc_info{};
			alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			alloc_info.allocationSize = memory_requirements.size;
			alloc_info.memoryTypeIndex = find_memory_type(physical_device, memory_requirements.memoryTypeBits, properties);

			if (vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate buffer memory!");
			}

			vkBindBufferMemory(device, buffer, buffer_memory, 0);
		}

		static void copy_buffer(VkDevice device, VkCommandPool command_pool, VkQueue queue, VkBuffer src, VkBuffer dst, VkDeviceSize size, VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0) {
				
			VkCommandBufferAllocateInfo alloc_info{};
			alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			alloc_info.commandPool = command_pool;
			alloc_info.commandBufferCount = 1;

			VkCommandBuffer command_buffer;

			vkAllocateCommandBuffers(device, &alloc_info, &command_buffer);

			VkSubmitInfo submit_info{};

			VkCommandBufferBeginInfo begin_info{};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;


			vkBeginCommandBuffer(command_buffer, &begin_info);

			VkBufferCopy copy_region{};
			copy_region.size = size;
			copy_region.srcOffset = src_offset;
			copy_region.dstOffset = dst_offset;

			vkCmdCopyBuffer(command_buffer, src, dst, 1, &copy_region);

			vkEndCommandBuffer(command_buffer);

			submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submit_info.commandBufferCount = 1;
			submit_info.pCommandBuffers = &command_buffer;
				

			vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
			vkQueueWaitIdle(queue);
			vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
		}

		template <
			buffer_type T_buffer_type
		>
			class glsl_location_handler {
			public:

			int usage =
				fan::conditional_value_t < T_buffer_type == buffer_type::buffer, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
				fan::conditional_value_t < T_buffer_type == buffer_type::index, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
				fan::conditional_value_t < T_buffer_type == buffer_type::staging, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, (uint_t)-1>::value>::value>::value;

			int properties =
				fan::conditional_value_t < T_buffer_type == buffer_type::buffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				fan::conditional_value_t < T_buffer_type == buffer_type::index, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				fan::conditional_value_t < T_buffer_type == buffer_type::staging, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, (uint_t)-1>::value>::value>::value;

			VkDevice* m_device;
			VkPhysicalDevice* m_physical_device;

			VkBuffer m_buffer_object;
			VkDeviceMemory m_device_memory;

			VkDeviceSize buffer_size = 0;

			glsl_location_handler(VkDevice* device, VkPhysicalDevice* physical_device, VkDeviceSize size = 0) :
				m_device(device),
				m_physical_device(physical_device),
				m_buffer_object(nullptr),
				m_device_memory(nullptr)
			{
				this->allocate_buffer(size);
			}

			~glsl_location_handler() {
				this->free_buffer();
			}

			void allocate_buffer(VkDeviceSize size = 0) {
				buffer_size = size;
				fan::gpu_memory::create_buffer(*m_device, *m_physical_device, size, usage, properties, m_buffer_object, m_device_memory);
			}

			void copy_buffer(VkCommandPool command_pool, VkQueue queue, VkBuffer src, VkDeviceSize size) {
				fan::gpu_memory::copy_buffer(*m_device, command_pool, queue, src, m_buffer_object, size);
			}

			void free_buffer() {
				buffer_size = 0;
				if (m_buffer_object) {
					vkDestroyBuffer(*m_device, m_buffer_object, nullptr);
					m_buffer_object = nullptr;
				}

				if (m_device_memory) {
					vkFreeMemory(*m_device, m_device_memory, nullptr);
					m_device_memory = nullptr;
				}

			}

			protected:

		};

		class uniform_handler {
		public:

			VkDevice* m_device;
			VkPhysicalDevice* m_physical_device;

			struct buffer_t {
				VkBuffer buffer;
				VkDeviceMemory memory;
			};

			std::vector<buffer_t> m_buffer_object;

			int usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

			int properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

			uniform_handler(VkDevice* device, VkPhysicalDevice* physical_device, VkDeviceSize swap_chain_size) :
				m_device(device),
				m_physical_device(physical_device)
			{
				m_buffer_object.resize(swap_chain_size);

				for (int i = 0; i < swap_chain_size; i++) {

					fan::gpu_memory::create_buffer(
						*m_device,
						*m_physical_device,
						sizeof(VkDeviceSize),
						usage,
						properties,
						m_buffer_object[i].buffer,
						m_buffer_object[i].memory
					);

				}
			}

			template <typename user_data_t>
			void upload(user_data_t* user_data, uint32_t image) {

				void* data;

				vkMapMemory(*m_device, m_buffer_object[image].memory, 0, sizeof(*user_data), 0, &data);

				memcpy(data, user_data, sizeof(user_data));

				vkUnmapMemory(*m_device, m_buffer_object[image].memory);
			}

		};

		template <typename object_type, buffer_type T_buffer_type>
		struct buffer_object {
				
			using buffer_t = glsl_location_handler<T_buffer_type>;

			buffer_object(VkDevice* device, VkPhysicalDevice* physical_device, VkCommandPool* pool, glsl_location_handler<gpu_memory::buffer_type::staging>* staging, VkQueue* graphics_queue, VkDeviceSize size = 0)
				:
				buffer(new buffer_t(device, physical_device, size)),
				staging(staging),
				graphics_queue(graphics_queue),
				pool(pool)
			{

				VkCommandBufferAllocateInfo alloc_info{};
				alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				alloc_info.commandPool = *pool;
				alloc_info.commandBufferCount = 1;

				vkAllocateCommandBuffers(*device, &alloc_info, &command_buffer);

				recreate_command_buffer(size, 0, 0);
				current_buffer_size = size;
			}

			~buffer_object() {
					

				if (command_buffer) {
					vkFreeCommandBuffers(*buffer->m_device, *pool, 1, &command_buffer);
					command_buffer = nullptr;
				}

				if (buffer) {
					delete buffer;
					buffer = nullptr;
				}

			}

			void push_back(const object_type& value) {
				m_instance.emplace_back(value);
			}

			object_type& get_value(uint32_t i) {
				return m_instance[i];
			}

			object_type get_value(uint32_t i) const {
				return m_instance[i];
			}

			void set_value(uint32_t i, const object_type& value) {
				m_instance[i] = value;
			}

			void map_data(VkDeviceSize size, VkDeviceSize offset = 0) {

				void* data = nullptr;

				if (staging->buffer_size < size) {
					staging->free_buffer();
					staging->allocate_buffer(size);
				}

				vkMapMemory(
					*buffer->m_device,
					staging->m_device_memory,
					offset,
					size,
					0,
					&data
				);

				std::memcpy(data, m_instance.data(), size);

				vkUnmapMemory(*buffer->m_device, staging->m_device_memory);
			}
				
			void write_data() {
				VkDeviceSize buffer_size = sizeof(object_type) * m_instance.size();

				map_data(buffer_size);

				if (buffer_size != current_buffer_size) {
					recreate_command_buffer(buffer_size, 0, 0);
					current_buffer_size = buffer_size;
				}

				VkSubmitInfo submit_info{};

				submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
				submit_info.commandBufferCount = 1;
				submit_info.pCommandBuffers = &command_buffer;

				vkQueueSubmit(*graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
			}

			// slow
			void edit_data(uint32_t i) {

				VkDeviceSize bufferSize = sizeof(object_type);

				void* data;
				vkMapMemory(
					*buffer->m_device,
					staging->m_device_memory,
					0,
					bufferSize,
					0,
					&data
				);

				memcpy(data, &m_instance[i], (std::size_t)bufferSize);
				vkUnmapMemory(*buffer->m_device, staging->m_device_memory);

				fan::gpu_memory::copy_buffer(*buffer->m_device, *pool, *graphics_queue, staging->m_buffer_object, buffer->m_buffer_object, bufferSize, 0, sizeof(object_type) * i);
			}

			void recreate_command_buffer(VkDeviceSize buffer_size, VkDeviceSize src_offset, VkDeviceSize dst_offset) {

				VkCommandBufferBeginInfo begin_info{};

				begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

				vkBeginCommandBuffer(command_buffer, &begin_info);

				VkBufferCopy copy_region{};

				copy_region.size = buffer_size;
				copy_region.srcOffset = src_offset;
				copy_region.dstOffset = dst_offset;

				vkCmdCopyBuffer(command_buffer, staging->m_buffer_object, buffer->m_buffer_object, 1, &copy_region);

				vkEndCommandBuffer(command_buffer);
			}

			std::size_t size() const {
				return m_instance.size();
			}

			std::vector<object_type> m_instance;

			buffer_t* buffer = nullptr;

			VkCommandBuffer command_buffer = nullptr;

			glsl_location_handler<gpu_memory::buffer_type::staging>* staging = nullptr;
			VkQueue* graphics_queue = nullptr;

			VkCommandPool* pool = nullptr;

			VkDeviceSize current_buffer_size = 0;

		};

	}

}

#endif