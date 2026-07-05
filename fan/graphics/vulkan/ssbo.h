struct ssbo_t	{

  ssbo_t() = default;
  ssbo_t(const ssbo_t& other) {
    *this = other;
  }
  ssbo_t& operator=(const ssbo_t& other) {
    auto& src = const_cast<ssbo_t&>(other);
    max_instance_size = src.max_instance_size;
    descriptor_count = src.descriptor_count;
    common = src.common;
    common.user_data = this;
    instance_list = std::move(src.instance_list);
    vram_capacity = src.vram_capacity;
    m_descriptor = std::move(src.m_descriptor);
    for (std::uint32_t i = 0; i < fan::vulkan::max_frames_in_flight; ++i) {
      data[i] = src.data[i];
      src.data[i] = nullptr;
      src.common.memory[i] = {};
    }
    src.vram_capacity = 0;
    return *this;
  }
  ssbo_t(ssbo_t&& other) noexcept { *this = std::move(other); }
  ssbo_t& operator=(ssbo_t&& other) noexcept {
    max_instance_size = other.max_instance_size;
    descriptor_count = other.descriptor_count;
    common = other.common;
    common.user_data = this;
    instance_list = std::move(other.instance_list);
    vram_capacity = other.vram_capacity;
    m_descriptor = std::move(other.m_descriptor);
    for (std::uint32_t i = 0; i < fan::vulkan::max_frames_in_flight; ++i) {
      data[i] = other.data[i];
    }
    other.vram_capacity = 0;
    for (std::uint32_t i = 0; i < fan::vulkan::max_frames_in_flight; ++i) {
      other.common.memory[i] = {};
      other.data[i] = nullptr;
    }
    return *this;
  }

  std::uint32_t max_instance_size = 1024;
  std::uint32_t descriptor_count = 0;

	static constexpr auto buffer_count = fan::vulkan::max_frames_in_flight;

  using instance_id_t = std::uint32_t;

  void allocate(fan::vulkan::context_t& context, std::uint64_t size) {
    vram_capacity = size;
    for (std::uint32_t frame = 0; frame < fan::vulkan::max_frames_in_flight; frame++) {
      if (common.memory[frame].buffer != VK_NULL_HANDLE) {
        if (frame == 0) {
          vkDeviceWaitIdle(context.device);
        }
        if (data[frame]) {
          vmaUnmapMemory(context.allocator, common.memory[frame].device_memory);
          data[frame] = nullptr;
        }
        context.destroy_buffer(common.memory[frame].buffer, common.memory[frame].device_memory);
      }

      context.create_buffer(
        size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        common.memory[frame].buffer,
        common.memory[frame].device_memory
      );
      fan::vulkan::validate(vmaMapMemory(context.allocator, common.memory[frame].device_memory, (void**)&data[frame]));
    }
  }

	void write(fan::vulkan::context_t& context) {
    if (!common.is_current_frame_dirty(context)) {
      common.on_edit(context);
      return;
    }

    auto frame = context.current_frame;
    if (!instance_list.empty()) {
      auto& ptr = instance_list[0];
      memcpy(data[frame], &ptr, instance_list.size() * sizeof(instance_id_t));
    }

		common.on_edit(context);
	}

  static void write_cb_impl(fan::vulkan::context_t& context, void* ptr) {
    static_cast<ssbo_t*>(ptr)->write(context);
  }

	void open(fan::vulkan::context_t& context, std::uint32_t descriptor_count, std::uint32_t max_instance_size = 256) {
    this->descriptor_count = descriptor_count;
    this->max_instance_size = max_instance_size;
		common.open(context, write_cb_impl, this);
	}
	void close(fan::vulkan::context_t& context) {
    for (std::uint32_t frame = 0; frame < fan::vulkan::max_frames_in_flight; frame++) {
      if (data[frame]) {
        vmaUnmapMemory(context.allocator, common.memory[frame].device_memory);
        data[frame] = nullptr;
      }
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

  struct instance_list_t {
    void* vi;
    void* ri;
  };

	memory_common_t<int, instance_id_t> common;
	std::vector<instance_list_t> instance_list;
	std::uint64_t vram_capacity = 0;
	fan::vulkan::context_t::descriptor_t m_descriptor;
	std::uint8_t* data[fan::vulkan::max_frames_in_flight] {};
};