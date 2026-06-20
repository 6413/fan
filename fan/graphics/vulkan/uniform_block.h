template <typename type_t, std::uint32_t element_size>
struct uniform_block_t {

	static constexpr auto buffer_count = fan::vulkan::max_frames_in_flight;

	struct open_properties_t {
		open_properties_t() {}
	}op;

	using nr_t = std::uint8_t;
	using instance_id_t = std::uint8_t;

	uniform_block_t() = default;

	uniform_block_t(fan::vulkan::context_t& context, open_properties_t op_ = open_properties_t()) {
		open(context, op_);
	}

	void open(fan::vulkan::context_t& context, open_properties_t op_ = open_properties_t()) {
		common.open(context, [&context, this] () {
			const auto begin = common.m_min_edit;
			const auto end = common.m_max_edit;

			if (!common.is_current_frame_dirty(context) || begin == 0xFFFFFFFFFFFFFFFF || end <= begin) {
				common.on_edit(context);
				return;
			}

			auto frame = context.current_frame;
			std::uint8_t* data;
			fan::vulkan::validate(vmaMapMemory(context.allocator, common.memory[frame].device_memory, (void**)&data));

			std::memcpy(data + begin, buffer + begin, end - begin);

			vmaUnmapMemory(context.allocator, common.memory[frame].device_memory);

			common.on_edit(context);
		});

		op = op_;

		m_size = 0;

		VkDeviceSize bufferSize = sizeof(type_t) * element_size;

		for (std::size_t i = 0; i < buffer_count; i++) {
			context.create_buffer(
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

	std::uint32_t size() const {
		return m_size / sizeof(type_t);
	}

	void push_ram_instance(fan::vulkan::context_t& context, const type_t& data) {
		const auto begin = static_cast<std::uint32_t>(m_size);
		std::memmove(&buffer[m_size], &data, sizeof(type_t));
		m_size += sizeof(type_t);
		common.edit(context, begin, static_cast<std::uint32_t>(m_size));
	}

  template <typename member_t>
  void edit_instance(fan::vulkan::context_t& context, std::uint32_t i, member_t type_t::* member, const member_t& value) {
    ((type_t*)buffer)[i].*member = value;

    const auto begin = static_cast<std::uint32_t>(
      sizeof(type_t) * i + fan::member_offset(member)
      );

    common.edit(context, begin, begin + sizeof(member_t));
  }

	fan::vulkan::context_t::memory_common_t<fan::graphics::shader_nr_t, instance_id_t> common;
	std::uint8_t buffer[element_size * sizeof(type_t)];
	std::uint32_t m_size;
};