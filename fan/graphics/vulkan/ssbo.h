#pragma once

namespace fan {
	namespace vulkan {
		namespace core {

			// vram instance ram instance
			template <typename vi_t, typename ri_t, uint32_t max_instance_size, uint32_t descriptor_count>
			struct ssbo_t	 {

				static constexpr auto buffer_count = fan::vulkan::MAX_FRAMES_IN_FLIGHT;

				#define BLL_set_MultipleType_Sizes sizeof(vi_t) * max_instance_size, sizeof(ri_t) * max_instance_size
				#include _FAN_PATH(fan_bll_preset.h)
				#define BLL_set_CPP_ConstructDestruct
				#define BLL_set_prefix instance_list
				#define BLL_set_type_node uint16_t
				#define BLL_set_Link 1
				#define BLL_set_MultipleType_LinkIndex 1
				//#define BVEC_set_BufferingFormat 0
				//#define BVEC_set_BufferingFormat0_WantedBufferByteAmount 0xfffff
				static constexpr auto multiple_type_link_index = BLL_set_MultipleType_LinkIndex;
				#define BLL_set_AreWeInsideStruct 1
				#define BLL_set_Overload_Declare \
					vi_t &get_vi(instance_list_NodeReference_t nr, uint32_t i) { \
						return ((vi_t*)this->GetNodeReferenceData(nr, 0))[i]; \
					} \
					ri_t &get_ri(instance_list_NodeReference_t nr, uint32_t i) { \
						return ((ri_t*)this->GetNodeReferenceData(nr, 1))[i]; \
					}
			protected:
				#include _FAN_PATH(BLL/BLL.h)
			public:
				
				using nr_t = instance_list_NodeReference_t;
				using instance_id_t = uint8_t;

				void allocate(fan::vulkan::context_t* context, uint64_t size, uint32_t frame) {

					if (instance_list.Usage() != 0) {
						if (common.memory[frame].buffer != nullptr) {
							vkDeviceWaitIdle(context->device);
							vkDestroyBuffer(context->device, common.memory[frame].buffer, 0);
							vkUnmapMemory(context->device, common.memory[frame].device_memory);
						}
					}

					create_buffer(
						context,
						size,
						VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
						VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
						//VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // ?
						// faster ^? depends about buffer size maxMemoryAllocationCount maybe
						common.memory[frame].buffer,
						common.memory[frame].device_memory
					);
					validate(vkMapMemory(context->device, common.memory[frame].device_memory, 0, vram_capacity, 0, (void**)&data));
				}

				void write(fan::vulkan::context_t* context, uint32_t frame) {
					
					if (common.m_min_edit != (uint64_t)-1) {
						memcpy(data, &instance_list.get_vi(nr_t{}, 0), instance_list.NodeList.Current * max_instance_size * sizeof(vi_t));
					}
					else {
						for (auto i : common.indices) {
							((vi_t*)data)[(uint32_t)i.nr.NRI * max_instance_size + i.i] = instance_list.get_vi(i.nr, i.i);
						}
					}

					common.on_edit(context);
				}

				//void write(fan::vulkan::context_t* context, uint32_t frame) {

				//	uint8_t* data;
				//	validate(vkMapMemory(context->device, common.memory[frame].device_memory, 0, vram_capacity, 0, (void**)&data));
				//	
				//	for (auto i : common.indices) {
				//		((vi_t*)data)[i.i] = instance_list.get_vi(i.nr, i.i);
				//	}
				//	// unnecessary? is necessary
				//	vkUnmapMemory(context->device, common.memory[frame].device_memory);

				//	common.on_edit(context);
				//}

				void open(fan::vulkan::context_t* context) {
					common.open(context, [context, this] {
						for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
							write(context, i);
						}
					});
				}
				void close(fan::vulkan::context_t* context, memory_write_queue_t* queue) {
					vkUnmapMemory(context->device, common.memory[context->currentFrame].device_memory);
					common.close(context, queue);
				}

				void open_descriptors(
					fan::vulkan::context_t* context,
					VkDescriptorPool descriptor_pool, 
					std::array<fan::vulkan::write_descriptor_set_t, descriptor_count> properties
				) {
					m_descriptor.open(context, descriptor_pool, properties);
				}

		/*		uint32_t size() const {
					return buffer.size();
				}*/

				nr_t add(fan::vulkan::context_t* context, memory_write_queue_t* wq) {
					uint32_t old_size = instance_list.Usage();

					nr_t nr = instance_list.NewNode();
					
					if (vram_capacity < instance_list.GetAmountOfAllocated() * sizeof(vi_t) * max_instance_size) {
						vram_capacity = instance_list.GetAmountOfAllocated() * sizeof(vi_t) * max_instance_size;
						for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
							allocate(context, vram_capacity, i);
							if (old_size) {
								common.edit(context, wq, 0, old_size);
							}
							m_descriptor.m_properties[0].buffer = common.memory[i].buffer;
							m_descriptor.update(context, 1);
						}
					}
					return nr;
				}

				void copy_instance(
					fan::vulkan::context_t* context, 
					memory_write_queue_t* write_queue, 
					nr_t src_block_nr, 
					instance_id_t src_instance_id, 
					nr_t dst_block_nr, 
					instance_id_t dst_instance_id
				) {
					instance_list.get_vi(dst_block_nr, dst_instance_id) = 
						instance_list.get_vi(src_block_nr, src_instance_id);

					instance_list.get_ri(dst_block_nr, dst_instance_id) = 
						instance_list.get_ri(src_block_nr, src_instance_id);
					common.edit(
						context,
						write_queue,
						{ dst_block_nr, dst_instance_id }
					);
				}
				
				void copy_instance(fan::vulkan::context_t* context, memory_write_queue_t* write_queue, nr_t nr, instance_id_t i, auto vi_t::*member, auto value) {
					instance_list.get_vi(nr, i).*member = value;
					common.edit(
						context,
						write_queue,
						{ nr, i }
					);
				}
				
				void copy_instance(fan::vulkan::context_t* context, memory_write_queue_t* write_queue, nr_t nr, instance_id_t i, vi_t* instance) {
					instance_list.get_vi(nr, i) = *instance;
					common.edit(
						context,
						write_queue,
						{ nr, i }
					);
				}

				memory_common_t<nr_t, instance_id_t> common;
				instance_list_t instance_list;
				uint64_t vram_capacity = 0;
				fan::vulkan::descriptor_t<descriptor_count> m_descriptor;
				uint8_t* data;
			};
		}
	}
}