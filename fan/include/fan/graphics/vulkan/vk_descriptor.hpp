#pragma once

#include <array>
#include <vector>
#include <stdexcept>

#include <vulkan/vulkan.h>

#include <fan/types/types.hpp>

namespace fan {

	namespace vk {

		namespace graphics {

			constexpr auto maximum_textures_per_instance = 128;

			struct descriptor_set {

				VkDevice* m_device = nullptr;

				descriptor_set(VkDevice* device) : m_device(device) {
					VkDescriptorSetLayoutBinding uboLayoutBinding{};
					uboLayoutBinding.binding = 0;
					uboLayoutBinding.descriptorCount = 1;
					uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
					uboLayoutBinding.pImmutableSamplers = nullptr;
					uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

					VkDescriptorSetLayoutBinding samplerLayoutBinding{};
					samplerLayoutBinding.binding = 1;
					samplerLayoutBinding.descriptorCount = 1;
					samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					samplerLayoutBinding.pImmutableSamplers = nullptr;
					samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

					std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
					VkDescriptorSetLayoutCreateInfo layoutInfo{};
					layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
					layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
					layoutInfo.pBindings = bindings.data();

					if (vkCreateDescriptorSetLayout(*device, &layoutInfo, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
						throw std::runtime_error("failed to create descriptor set layout!");
					}
				}

				// creates new if not already made
				void recreate_descriptor_pool(VkDeviceSize swap_chain_images_size) {
					if (descriptor_pool) {

						if (descriptor_sets.size()) {
							vkFreeDescriptorSets(*m_device, descriptor_pool, descriptor_sets.size(), descriptor_sets.data());

							std::fill(descriptor_sets.begin(), descriptor_sets.end(), nullptr);
						}

						vkDestroyDescriptorPool(*m_device, descriptor_pool, nullptr);
						descriptor_pool = nullptr;
					}

					// todo fix
					fan::print("i disturb you as long as you dont fix me");
					std::array<VkDescriptorPoolSize, 128> poolSizes{};
					poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
					poolSizes[0].descriptorCount = swap_chain_images_size;

					poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					poolSizes[1].descriptorCount = swap_chain_images_size;

					for (int i = 2; i < poolSizes.size(); i++) {
						poolSizes[i] = poolSizes[i & 1];
					}

					VkDescriptorPoolCreateInfo poolInfo{};
					poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
					poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
					poolInfo.pPoolSizes = poolSizes.data();
					poolInfo.maxSets = maximum_textures_per_instance * swap_chain_images_size;
					poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

					if (vkCreateDescriptorPool(*m_device, &poolInfo, nullptr, &descriptor_pool) != VK_SUCCESS) {
						throw std::runtime_error("failed to create descriptor pool.");
					}

				}

				void free_set_layout() {

					if (descriptor_set_layout) {
						vkDestroyDescriptorSetLayout(*m_device, descriptor_set_layout, nullptr);
						descriptor_set_layout = nullptr;
					}
				}

				~descriptor_set() {

					vkDestroyDescriptorSetLayout(*m_device, descriptor_set_layout, nullptr);
					descriptor_set_layout = nullptr;

					vkDestroyDescriptorPool(*m_device, descriptor_pool, nullptr);
					descriptor_pool = nullptr;

				}

				std::size_t size() const {
					return descriptor_sets.size();
				}


				VkDescriptorSet& get(std::size_t i) {
					return descriptor_sets[i];
				}

				VkDescriptorSet get(std::size_t i) const {
					return descriptor_sets[i];
				}

				// if image_view is nullptr, image view is not made
				template <typename T>
				void push_back(
					VkDevice device,
					T* uniform_buffer,
					VkDescriptorSetLayout descriptor_layout,
					VkDescriptorPool descriptor_pool,
					VkImageView image_view,
					VkSampler texture_sampler,
					VkDeviceSize swap_chain_images_size
				) {

					std::vector<VkDescriptorSetLayout> layouts(swap_chain_images_size, descriptor_layout);
					VkDescriptorSetAllocateInfo allocInfo{};
					allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
					allocInfo.descriptorPool = descriptor_pool;
					allocInfo.descriptorSetCount = swap_chain_images_size;
					allocInfo.pSetLayouts = layouts.data();

					auto offset = descriptor_sets.size();

					descriptor_sets.resize(descriptor_sets.size() + swap_chain_images_size);

					VkResult status;

					if ((status = vkAllocateDescriptorSets(device, &allocInfo, &descriptor_sets[offset])) != VK_SUCCESS) {
						throw std::runtime_error("failed to allocate descriptor sets.");
					}

					for (size_t i = 0; i < swap_chain_images_size; i++) {
						VkDescriptorBufferInfo bufferInfo{};
						bufferInfo.buffer = uniform_buffer->m_buffer_object[i].buffer;
						bufferInfo.offset = 0;
						bufferInfo.range = uniform_buffer->user_data_size;

						VkDescriptorImageInfo image_info{};
						image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
						image_info.imageView = image_view;
						image_info.sampler = texture_sampler;

						std::array<VkWriteDescriptorSet, 2> descriptorWrites{};


						descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						descriptorWrites[0].dstSet = descriptor_sets[i + offset];
						descriptorWrites[0].dstBinding = 0;
						descriptorWrites[0].dstArrayElement = 0;
						descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
						descriptorWrites[0].descriptorCount = 1;
						descriptorWrites[0].pBufferInfo = &bufferInfo;

						if (image_view != nullptr) {

							descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
							descriptorWrites[1].dstSet = descriptor_sets[i + offset];
							descriptorWrites[1].dstBinding = 1;
							descriptorWrites[1].dstArrayElement = 0;
							descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
							descriptorWrites[1].descriptorCount = 1;
							descriptorWrites[1].pImageInfo = &image_info;
						}

						vkUpdateDescriptorSets(device, image_view == nullptr ? 1 : descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
					}

				}

				void erase(uint32_t i, uint32_t swap_chain_images_size) {

					vkFreeDescriptorSets(*m_device, descriptor_pool, swap_chain_images_size, descriptor_sets.data() + i * swap_chain_images_size);

					descriptor_sets.erase(descriptor_sets.begin() + i * swap_chain_images_size, descriptor_sets.begin() + i * swap_chain_images_size + swap_chain_images_size);
				}

				template <typename T>
				void update_descriptor_sets(
					uint32_t index,
					VkDevice device,
					T* uniform_buffer,
					VkDescriptorSetLayout descriptor_layout,
					VkDescriptorPool descriptor_pool,
					VkImageView image_view,
					VkSampler texture_sampler,
					VkDeviceSize swap_chain_images_size
				) {
					std::vector<VkDescriptorSetLayout> layouts(swap_chain_images_size, descriptor_layout);
					VkDescriptorSetAllocateInfo allocInfo{};
					allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
					allocInfo.descriptorPool = descriptor_pool;
					allocInfo.descriptorSetCount = swap_chain_images_size;
					allocInfo.pSetLayouts = layouts.data();

					VkResult status;

					auto offset = index * swap_chain_images_size;

					if ((status = vkAllocateDescriptorSets(device, &allocInfo, &descriptor_sets[offset])) != VK_SUCCESS) {
						throw std::runtime_error("failed to allocate descriptor sets.");
					}

					for (size_t i = 0; i < swap_chain_images_size; i++) {

						VkDescriptorBufferInfo buffer_info{};
						buffer_info.buffer = uniform_buffer->m_buffer_object[i].buffer;
						buffer_info.offset = 0;
						buffer_info.range = uniform_buffer->user_data_size;

						VkDescriptorImageInfo image_info{};
						image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
						image_info.imageView = image_view;
						image_info.sampler = texture_sampler;

						std::array<VkWriteDescriptorSet, 2> descriptor_writes{};

						descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						descriptor_writes[0].dstSet = descriptor_sets[i + offset];
						descriptor_writes[0].dstBinding = 0;
						descriptor_writes[0].dstArrayElement = 0;
						descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
						descriptor_writes[0].descriptorCount = 1;
						descriptor_writes[0].pBufferInfo = &buffer_info;

						if (image_view != nullptr) {
							descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
							descriptor_writes[1].dstSet = descriptor_sets[i + offset];
							descriptor_writes[1].dstBinding = 1;
							descriptor_writes[1].dstArrayElement = 0;
							descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
							descriptor_writes[1].descriptorCount = 1;

							descriptor_writes[1].pImageInfo = &image_info;
						}


						vkUpdateDescriptorSets(device, image_view == nullptr ? 1 : descriptor_writes.size(), descriptor_writes.data(), 0, nullptr);
					}
				}

				std::vector<VkDescriptorSet> descriptor_sets;

				VkDescriptorSetLayout descriptor_set_layout = nullptr;

				VkDescriptorPool descriptor_pool = nullptr;

			};

		}

	}

}