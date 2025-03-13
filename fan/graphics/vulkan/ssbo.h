// vram instance, ram instance
struct ssbo_t	{

  uint32_t max_instance_size = 1024;
  uint32_t descriptor_count = 0;

	static constexpr auto buffer_count = fan::vulkan::max_frames_in_flight;

//	#define BLL_set_MultipleType_Sizes sizeof(vi_t) * max_instance_size, sizeof(ri_t) * max_instance_size
//	#include <fan/fan_bll_preset.h>
//	#define BLL_set_prefix instance_list
//	#define BLL_set_type_node uint16_t
//	#define BLL_set_Link 1
//	#define BLL_set_MultipleType_LinkIndex 1
//	//#define BVEC_set_BufferingFormat 0
//	//#define BVEC_set_BufferingFormat0_WantedBufferByteAmount 0xfffff
//	static constexpr auto multiple_type_link_index = BLL_set_MultipleType_LinkIndex;
//	#define BLL_set_AreWeInsideStruct 1
//	#define BLL_set_Overload_Declare \
//		vi_t &get_vi(instance_list_NodeReference_t nr, uint32_t i) { \
//			return ((vi_t*)this->GetNodeReferenceData(nr, 0))[i]; \
//		} \
//		ri_t &get_ri(instance_list_NodeReference_t nr, uint32_t i) { \
//			return ((ri_t*)this->GetNodeReferenceData(nr, 1))[i]; \
//		}
//protected:
//	#include <BLL/BLL.h>
//public:
//	
//	using nr_t = instance_list_NodeReference_t;
//	using instance_id_t = uint8_t;

  using instance_id_t = uint32_t;

  void allocate(fan::vulkan::context_t& context, uint64_t size) {
    for (uint32_t frame = 0; frame < fan::vulkan::max_frames_in_flight; frame++) {
      if (instance_list.size() != 0) {
        if (common.memory[frame].buffer != nullptr) {
          // Only need to wait idle once before destroying all buffers
          if (frame == 0) {
            vkDeviceWaitIdle(context.device);
          }
          vkDestroyBuffer(context.device, common.memory[frame].buffer, 0);
          vkUnmapMemory(context.device, common.memory[frame].device_memory);
        }
      }

      context.create_buffer(
        size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        common.memory[frame].buffer,
        common.memory[frame].device_memory
      );
      fan::vulkan::validate(vkMapMemory(context.device, common.memory[frame].device_memory,
        0, size, 0, (void**)&data[frame]));
    }
  }

	void write(fan::vulkan::context_t& context) {
					
    // write all for now
    auto& ptr = instance_list[0];
		for (uint32_t frame = 0; frame < fan::vulkan::max_frames_in_flight; frame++) {
      memcpy(data[frame], &ptr, instance_list.size() * sizeof(instance_id_t));
    }
    // for loop for each frame
		//if (common.m_min_edit != (uint64_t)-1) {
//       // TODO not probably best way
//       
		//}
		//else {
		//	for (auto i : common.indices) {
		//		((vi_t*)data)[(uint32_t)i.nr.NRI * max_instance_size + i.i] = instance_list.get_vi(i.nr, i.i);
		//	}
		//}

		common.on_edit(context);
	}

	//void write(fan::vulkan::context_t* context, uint32_t frame) {

	//	uint8_t* data;
	//	validate(vkMapMemory(context.device, common.memory[frame].device_memory, 0, vram_capacity, 0, (void**)&data));
	//	
	//	for (auto i : common.indices) {
	//		((vi_t*)data)[i.i] = instance_list.get_vi(i.nr, i.i);
	//	}
	//	// unnecessary? is necessary
	//	vkUnmapMemory(context.device, common.memory[frame].device_memory);

	//	common.on_edit(context);
	//}

	void open(fan::vulkan::context_t& context, uint32_t descriptor_count, uint32_t max_instance_size = 256) {
    this->descriptor_count = descriptor_count;
    this->max_instance_size = max_instance_size;
		common.open(context, [&context, this] {
			write(context);
		});
	}
	void close(fan::vulkan::context_t& context) {
		for (uint32_t frame = 0; frame < fan::vulkan::max_frames_in_flight; frame++) {
      vkUnmapMemory(context.device, common.memory[frame].device_memory);
    }
    m_descriptor.close(context);
		common.close(context);
	}

	void open_descriptors(
		fan::vulkan::context_t& context,
		const std::vector<fan::vulkan::write_descriptor_set_t>& properties
	) {
		m_descriptor.open(context, properties);
	}

/*		uint32_t size() const {
		return buffer.size();
	}*/

	/*nr_t add(fan::vulkan::context_t& context, memory_write_queue_t* wq) {
		uint32_t old_size = instance_list.Usage();

		nr_t nr = instance_list.NewNode();
					
		if (vram_capacity < instance_list.GetAmountOfAllocated() * sizeof(vi_t) * max_instance_size) {
			vram_capacity = instance_list.GetAmountOfAllocated() * sizeof(vi_t) * max_instance_size;
			for (uint32_t i = 0; i < max_frames_in_flight; ++i) {
				allocate(context, vram_capacity, i);
				if (old_size) {
					common.edit(context, wq, 0, old_size);
				}
				m_descriptor.m_properties[0].buffer = common.memory[i].buffer;
				m_descriptor.update(context, 1);
			}
		}
		return nr;
	}*/

/*	void copy_instance(
		fan::vulkan::context_t& context, 
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
	}*/
				
	/*void copy_instance(fan::vulkan::context_t& context, memory_write_queue_t* write_queue, nr_t nr, instance_id_t i, auto vi_t::*member, auto value) {
		instance_list.get_vi(nr, i).*member = value;
		common.edit(
			context,
			write_queue,
			{ nr, i }
		);
	}*/
				
	/*void copy_instance(fan::vulkan::context_t& context, memory_write_queue_t* write_queue, instance_id_t i, vi_t* instance) {
		instance_list[i].vi = *instance;
		common.edit(
			context,
			write_queue,
			{ nr, i }
		);
	}*/

  struct instance_list_t {
    void* vi;
    void* ri;
  };

	memory_common_t<int, instance_id_t> common;
	std::vector<instance_list_t> instance_list;
	uint64_t vram_capacity = 0;
	fan::vulkan::context_t::descriptor_t m_descriptor;
	uint8_t* data[fan::vulkan::max_frames_in_flight];
};