template <typename type_t, uint32_t element_size>
struct uniform_block_t {

	static constexpr auto buffer_count = fan::vulkan::max_frames_in_flight;

	struct open_properties_t {
		open_properties_t() {}
	}op;

	using nr_t = uint8_t;
	using instance_id_t = uint8_t;

	uniform_block_t() = default;

	uniform_block_t(fan::vulkan::context_t& context, open_properties_t op_ = open_properties_t()) {
		open(context, op);
	}

	void open(fan::vulkan::context_t& context, open_properties_t op_ = open_properties_t()) {
		common.open(context, [&context, this] () {
			for (uint32_t frame = 0; frame < fan::vulkan::max_frames_in_flight; ++frame) {

				uint8_t* data;
				validate(vkMapMemory(context.device, common.memory[frame].device_memory, 0, element_size * sizeof(type_t), 0, (void**)&data));
							
				for (auto j : common.indices) {
					((type_t*)data)[j.i] = ((type_t*)buffer)[j.i];
				}
				// unnecessary? is necessary
				vkUnmapMemory(context.device, common.memory[frame].device_memory);

				common.on_edit(context);
			}
		});

		op = op_;

		m_size = 0;

		VkDeviceSize bufferSize = sizeof(type_t) * element_size;

		for (size_t i = 0; i < buffer_count; i++) {
			create_buffer(
				context,
				bufferSize, 
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
				common.memory[i].buffer,
				common.memory[i].device_memory
			);
		}
	}
	void close(fan::vulkan::context_t& context) {
		common.close(context);
	}

	uint32_t size() const {
		return m_size / sizeof(type_t);
	}

	void push_ram_instance(fan::vulkan::context_t& context, const type_t& data) {
		std::memmove(&buffer[m_size], (void*)&data, sizeof(type_t));
		m_size += sizeof(type_t);
		common.edit(context, {0, (unsigned char)(m_size / sizeof(type_t) - 1)});
	}

	void edit_instance(fan::vulkan::context_t& context, uint32_t i, auto type_t::*member, auto value) {
		((type_t*)buffer)[i].*member = value;
		common.edit(context, {0, (unsigned char)i});
	}

	// nr_t is useless here
	fan::vulkan::context_t::memory_common_t<nr_t, instance_id_t> common;
	uint8_t buffer[element_size * sizeof(type_t)];
	uint32_t m_size;
};