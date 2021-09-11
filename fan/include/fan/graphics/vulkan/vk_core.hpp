#pragma once

#include <fan/graphics/renderer.hpp>

#if fan_renderer == fan_renderer_vulkan

#include <fan/types/types.hpp>

#include <vulkan/vulkan.h>

#include <vulkan/vulkan_core.h>

#include <fan/time/time.hpp>

namespace fan {

	namespace gpu_memory {

		inline std::vector<VkSubmitInfo> submit_queue;

		enum class buffer_type {
			buffer,
			index,
			staging,
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

		static VkCommandBuffer begin_command_buffer(VkDevice device, VkCommandPool pool) {
			VkCommandBufferAllocateInfo alloc_info{};
			alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			alloc_info.commandPool = pool;
			alloc_info.commandBufferCount = 1;

			VkCommandBuffer command_buffer;
			vkAllocateCommandBuffers(device, &alloc_info, &command_buffer);

			VkCommandBufferBeginInfo begin_info{};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			vkBeginCommandBuffer(command_buffer, &begin_info);

			return command_buffer;
		}

		static void end_command_buffer(VkCommandBuffer command_buffer, VkDevice device, VkCommandPool pool, VkQueue graphics_queue) {
			vkEndCommandBuffer(command_buffer);

			VkSubmitInfo submit_info{};
			submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submit_info.commandBufferCount = 1;
			submit_info.pCommandBuffers = &command_buffer;

			vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
			vkQueueWaitIdle(graphics_queue);

			vkFreeCommandBuffers(device, pool, 1, &command_buffer);
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
			assert(0);
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
				fan::conditional_value_t < T_buffer_type == buffer_type::staging, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, (uintptr_t)-1>::value>::value>::value;

			int properties =
				fan::conditional_value_t < T_buffer_type == buffer_type::buffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				fan::conditional_value_t < T_buffer_type == buffer_type::index, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				fan::conditional_value_t < T_buffer_type == buffer_type::staging, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, (uintptr_t)-1>::value>::value>::value;

			VkDevice* m_device;
			VkPhysicalDevice* m_physical_device;

			VkBuffer m_buffer_object;
			VkDeviceMemory m_device_memory;

			VkDeviceSize buffer_size = 0;

			glsl_location_handler(VkDevice* device, VkPhysicalDevice* physical_device) :
				m_device(device),
				m_physical_device(physical_device),
				m_buffer_object(nullptr),
				m_device_memory(nullptr)
			{

			}

			~glsl_location_handler() {
				this->free();
			}

			// creates new buffer with given size
			void allocate(VkDeviceSize size) {

				buffer_size = size;
				fan::gpu_memory::create_buffer(*m_device, *m_physical_device, size, usage, properties, m_buffer_object, m_device_memory);

			}

			void copy(VkCommandPool command_pool, VkQueue queue, VkBuffer src, VkDeviceSize size) {
				fan::gpu_memory::copy_buffer(*m_device, command_pool, queue, src, m_buffer_object, size);
			}

			void free() {

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

			void* user_data = nullptr;
			VkDeviceSize user_data_size = 0;

			VkDevice* m_device;
			VkPhysicalDevice* m_physical_device;

			struct buffer_t {
				VkBuffer buffer;
				VkDeviceMemory memory;
			};

			std::vector<buffer_t> m_buffer_object;

			static constexpr int usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

			static constexpr int properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

			uniform_handler(VkDevice* device, VkPhysicalDevice* physical_device, VkDeviceSize swap_chain_size, void* user_data, VkDeviceSize user_data_size) :
				m_device(device),
				m_physical_device(physical_device),
				user_data(user_data),
				user_data_size(user_data_size)
			{
				m_buffer_object.resize(swap_chain_size);

				for (int i = 0; i < swap_chain_size; i++) {

					fan::gpu_memory::create_buffer(
						*m_device,
						*m_physical_device,
						user_data_size,
						usage,
						properties,
						m_buffer_object[i].buffer,
						m_buffer_object[i].memory
					);

				}
			}
			
			~uniform_handler() {
				this->free();
			}

			void recreate(VkDeviceSize swap_chain_size) {

				this->free();

				m_buffer_object.resize(swap_chain_size);

				for (int i = 0; i < swap_chain_size; i++) {

					fan::gpu_memory::create_buffer(
						*m_device,
						*m_physical_device,
						user_data_size,
						usage,
						properties,
						m_buffer_object[i].buffer,
						m_buffer_object[i].memory
					);
					
				}
			}

			void free() {
				for (int i = 0; i < m_buffer_object.size(); i++) {

					if (m_buffer_object[i].buffer) {
						vkDestroyBuffer(*m_device, m_buffer_object[i].buffer, nullptr);
					}
					if (m_buffer_object[i].memory) {
						vkFreeMemory(*m_device, m_buffer_object[i].memory, nullptr);
					}
				}

				m_buffer_object.clear();
			}

			void upload(uint32_t image) {

				void* data;

				vkMapMemory(*m_device, m_buffer_object[image].memory, 0, user_data_size, 0, &data);

				memcpy(data, user_data, user_data_size);

				vkUnmapMemory(*m_device, m_buffer_object[image].memory);
			}

		};

		struct texture_handler {

			texture_handler(VkDevice* device, VkPhysicalDevice* physical_device, VkCommandPool* pool, VkQueue* graphics_queue)
				:
				m_device(device),
				m_physical_device(physical_device),
				m_pool(pool),
				m_graphics_queue(graphics_queue),
				descriptor_handler(new fan::vk::graphics::descriptor_set(device))
			{
				VkPhysicalDeviceProperties properties{};
				vkGetPhysicalDeviceProperties(*physical_device, &properties);

				VkSamplerCreateInfo samplerInfo{};
				samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
				samplerInfo.magFilter = VK_FILTER_LINEAR;
				samplerInfo.minFilter = VK_FILTER_LINEAR;
				samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
				samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
				samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
				samplerInfo.anisotropyEnable = VK_TRUE;
				samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
				samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
				samplerInfo.unnormalizedCoordinates = VK_FALSE;
				samplerInfo.compareEnable = VK_FALSE;
				samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
				samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
				//samplerInfo.minLod = 10;
				//samplerInfo.maxLod = 0;

				if (vkCreateSampler(*device, &samplerInfo, nullptr, &texture_sampler) != VK_SUCCESS) {
					throw std::runtime_error("failed to create texture sampler!");
				}
			}

			~texture_handler() {
				this->free();
			}

			void allocate(VkImage image) {
				VkMemoryRequirements memory_requirements;
				vkGetImageMemoryRequirements(*m_device, image, &memory_requirements);

				VkMemoryAllocateInfo alloc_info{};
				alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
				alloc_info.allocationSize = memory_requirements.size;
				alloc_info.memoryTypeIndex = fan::gpu_memory::find_memory_type(*m_physical_device, memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

				m_image_memory.resize(m_image_memory.size() + 1);

				if (vkAllocateMemory(*m_device, &alloc_info, nullptr, &m_image_memory[m_image_memory.size() - 1]) != VK_SUCCESS) {
					throw std::runtime_error("failed to allocate image memory!");
				}

				vkBindImageMemory(*m_device, image, m_image_memory[m_image_memory.size() - 1], 0);
			}

			void free() {
				for (int i = 0; i < m_image_memory.size(); i++) {

					vkFreeMemory(*m_device, m_image_memory[i], nullptr);
					m_image_memory[i] = nullptr;
				}


				vkDestroySampler(*m_device, texture_sampler, nullptr);
				texture_sampler = nullptr;

				delete descriptor_handler;
				descriptor_handler = nullptr;

				for (int i = 0; i < image_views.size(); i++) {
					vkDestroyImageView(*m_device, image_views[i], nullptr);
					image_views[i] = nullptr;
				}

			}

			void copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {

				VkCommandBuffer command_buffer = begin_command_buffer(*m_device, *m_pool);

				VkBufferImageCopy region{};
				region.bufferOffset = 0;
				region.bufferRowLength = 0;
				region.bufferImageHeight = 0;

				region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				region.imageSubresource.mipLevel = 0;
				region.imageSubresource.baseArrayLayer = 0;
				region.imageSubresource.layerCount = 1;

				region.imageOffset = { 0, 0, 0 };
				region.imageExtent = {
					width,
					height,
					1
				};

				vkCmdCopyBufferToImage(
					command_buffer,
					buffer,
					image,
					VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					1,
					&region
				);

				end_command_buffer(command_buffer, *m_device, *m_pool, *m_graphics_queue);
			}

			void transition_image_layout(VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout, uint32_t mip_levels) {

				VkCommandBuffer command_buffer = begin_command_buffer(*m_device, *m_pool);

				VkImageMemoryBarrier barrier{};
				barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
				barrier.oldLayout = old_layout;
				barrier.newLayout = new_layout;
				barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.image = image;
				barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				barrier.subresourceRange.baseMipLevel = 0;
				barrier.subresourceRange.levelCount = mip_levels;
				barrier.subresourceRange.baseArrayLayer = 0;
				barrier.subresourceRange.layerCount = 1;

				VkPipelineStageFlags source_stage;
				VkPipelineStageFlags destination_stage;

				if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
					barrier.srcAccessMask = 0;
					barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

					source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
					destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
				}
				else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
					barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
					barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

					source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
					destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
				}
				else {
					throw std::invalid_argument("unsupported layout transition.");
				}

				vkCmdPipelineBarrier(
					command_buffer,
					source_stage, destination_stage,
					0,
					0, nullptr,
					0, nullptr,
					1, &barrier
				);

				end_command_buffer(command_buffer, *m_device, *m_pool, *m_graphics_queue);
			}

			VkImageView create_image_view(VkImage image, VkFormat format, uint32_t mip_levels) {

				VkImageViewCreateInfo view_info{};
				view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
				view_info.image = image;
				view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
				view_info.format = format;
				view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				view_info.subresourceRange.baseMipLevel = 0;
				view_info.subresourceRange.levelCount = mip_levels;
				view_info.subresourceRange.baseArrayLayer = 0;
				view_info.subresourceRange.layerCount = 1;

				VkImageView image_view;

				if (vkCreateImageView(*m_device, &view_info, nullptr, &image_view) != VK_SUCCESS) {
					throw std::runtime_error("failed to create texture image view!");
				}

				return image_view;
			}

			template <typename T>
			uint64_t push_back(
				VkImage texture_id, 
				T* uniform_buffers,
				VkDeviceSize swap_chain_image_size,
				uint32_t mipmap_level
				) {

				image_views.emplace_back(create_image_view(texture_id, VK_FORMAT_R8G8B8A8_UNORM, mipmap_level));

				descriptor_handler->push_back(
					*m_device, 
					uniform_buffers,
					descriptor_handler->descriptor_set_layout,
					descriptor_handler->descriptor_pool,
					image_views[image_views.size() - 1],
					texture_sampler,
					swap_chain_image_size
				);
				
				return descriptor_handler->descriptor_sets.size() / swap_chain_image_size - 1;
			}

			std::vector<VkDeviceMemory> m_image_memory;

			VkDevice* m_device = nullptr;
			VkPhysicalDevice* m_physical_device = nullptr;

			VkCommandPool* m_pool = nullptr;
			VkQueue* m_graphics_queue = nullptr;

			std::vector<VkImageView> image_views;

			fan::vk::graphics::descriptor_set* descriptor_handler = nullptr;

			VkSampler texture_sampler = nullptr;

		};

		struct memory_update_queue_t {
			VkBuffer* staging_buffer = nullptr; // staging->m_buffer_object
			VkBuffer* buffer_buffer = nullptr; // buffer->m_buffer_object
			VkBufferCopy copy_region{};
		};

		struct memory_update_queue_vector_t {
			uint64_t key;
			memory_update_queue_t queue;
		};

		inline std::vector<memory_update_queue_vector_t> memory_update_map;

		static void register_memory_update(uint64_t key, const memory_update_queue_t& memory_update_queue) {
			memory_update_map.emplace_back(memory_update_queue_vector_t{ key, memory_update_queue });
		}

		inline VkCommandBuffer memory_command_buffer = nullptr;

		static void update_memory_buffer() {

			VkCommandBufferBeginInfo begin_info{};

			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

			vkBeginCommandBuffer(memory_command_buffer, &begin_info);
			
			for (int i = 0; i < memory_update_map.size(); i++) {
				if (memory_update_map[i].queue.copy_region.size) {
					vkCmdCopyBuffer(memory_command_buffer, *memory_update_map[i].queue.staging_buffer, *memory_update_map[i].queue.buffer_buffer, 1, &memory_update_map[i].queue.copy_region);
				}
			}

			vkEndCommandBuffer(memory_command_buffer);

			memory_update_map.clear();
		}

		template <typename object_type, buffer_type T_buffer_type>
		struct buffer_object {
				
			using buffer_t = glsl_location_handler<T_buffer_type>;
			
			using staging_buffer_t = fan::gpu_memory::glsl_location_handler<fan::gpu_memory::buffer_type::staging>;

			staging_buffer_t* staging_buffer = nullptr;

			static constexpr auto mb = 1000000;

			static constexpr auto gpu_stack = 10 * mb; // mb

			buffer_object(VkDevice* device, VkPhysicalDevice* physical_device, VkCommandPool* pool, VkQueue* graphics_queue, std::function<void()> vulkan_command_buffer_recreation_)
				:
				buffer(new buffer_t(device, physical_device)),
				graphics_queue(graphics_queue),
				pool(pool),
				vulkan_command_buffer_recreation(vulkan_command_buffer_recreation_)
			{

				staging_buffer = new staging_buffer_t(device, physical_device);
				staging_buffer->allocate(gpu_stack);

			}

			~buffer_object() {

				if (staging_buffer) {
					delete staging_buffer;
					staging_buffer = nullptr;
				}

				if (buffer) {
					delete buffer;
					buffer = nullptr;
				}
			}

			constexpr void push_back(const object_type& value) {
				m_instance.emplace_back(value);
			}

			constexpr object_type& get_value(uint32_t i) {
				return m_instance[i];
			}

			constexpr object_type get_value(uint32_t i) const {
				return m_instance[i];
			}

			constexpr void set_value(uint32_t i, const object_type& value) {
				m_instance[i] = value;
			}

			void map_data(VkDeviceSize size, VkDeviceSize offset = 0) {
				
				
				void* data = nullptr;

				if (staging_buffer->buffer_size < size) {
					staging_buffer->free();
					staging_buffer->allocate(size);
				}

				vkMapMemory(
					*buffer->m_device,
					staging_buffer->m_device_memory,
					sizeof(object_type) * offset,
					size,
					0,
					&data
				);

				std::memcpy(data, m_instance.data() + offset, size);

				vkUnmapMemory(*buffer->m_device, staging_buffer->m_device_memory);
			}
				
			void write_data() {

				VkDeviceSize buffer_size = sizeof(object_type) * m_instance.size();

				if (!buffer_size) {
					return;
				}

				auto found = std::find_if(memory_update_map.begin(), memory_update_map.end(), [&](const memory_update_queue_vector_t& a) { return a.key == (uint64_t)(this + buffer_size); }) != memory_update_map.end();

				if (found) {
					return;
				}
				
				auto previous_size = buffer->buffer_size;

				if (previous_size < buffer_size) {

					buffer->free();

					buffer->allocate(buffer_size);
				}
				
				map_data(buffer_size);

				memory_update_queue_t queue;
				queue.staging_buffer = &staging_buffer->m_buffer_object;
				queue.buffer_buffer = &buffer->m_buffer_object;

				VkBufferCopy copy{ 0 };

				copy.size = buffer_size;
				queue.copy_region = copy;

				register_memory_update((uint64_t)(this + buffer_size), queue);

			}

			void edit_data(uint32_t i) {

				VkDeviceSize buffer_size = sizeof(object_type);

				if (!buffer_size) {
					return;
				}

				auto found = std::find_if(memory_update_map.begin(), memory_update_map.end(), [&](const memory_update_queue_vector_t& a) { return a.key == (uint64_t)(this + buffer_size); }) != memory_update_map.end();

				if (found) {
					return;
				}

				auto previous_size = buffer->buffer_size;

				if (previous_size < buffer_size) {

					buffer->free();

					buffer->allocate(buffer_size);
				}

				map_data(buffer_size, i);

				memory_update_queue_t queue;
				queue.staging_buffer = &staging_buffer->m_buffer_object;
				queue.buffer_buffer = &buffer->m_buffer_object;

				VkBufferCopy copy{ 0 };

				copy.size = buffer_size;
				copy.srcOffset = sizeof(object_type) * i;
				copy.dstOffset = sizeof(object_type) * i;

				queue.copy_region = copy;

				register_memory_update((uint64_t)(this + buffer_size), queue);
			
			}

			void edit_data(uint32_t begin, uint32_t end) {

				VkDeviceSize buffer_size = sizeof(object_type) * (end - begin); // + 1 ?

				if (!buffer_size) {
					return;
				}

				auto found = std::find_if(memory_update_map.begin(), memory_update_map.end(), [&](const memory_update_queue_vector_t& a) { return a.key == (uint64_t)(this + buffer_size); }) != memory_update_map.end();

				if (found) {
					return;
				}

				auto previous_size = buffer->buffer_size;

				if (previous_size < buffer_size) {

					buffer->free();

					buffer->allocate(buffer_size);
				}

				map_data(buffer_size, begin);

				memory_update_queue_t queue;
				queue.staging_buffer = &staging_buffer->m_buffer_object;
				queue.buffer_buffer = &buffer->m_buffer_object;

				VkBufferCopy copy{ 0 };

				copy.size = buffer_size;
				copy.srcOffset = sizeof(object_type) * begin;
				copy.dstOffset = sizeof(object_type) * begin;

				queue.copy_region = copy;

				register_memory_update((uint64_t)(this + buffer_size), queue);
			}

			std::size_t size() const {
				return m_instance.size();
			}

			std::vector<object_type> m_instance;

			buffer_t* buffer = nullptr;

			VkQueue* graphics_queue = nullptr;

			VkCommandPool* pool = nullptr;

			VkDeviceSize current_buffer_size = 0;

			std::function<void()> vulkan_command_buffer_recreation;

		};

	}

}

#endif