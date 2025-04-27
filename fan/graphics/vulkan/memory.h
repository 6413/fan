struct memory_write_queue_t {

	memory_write_queue_t() : write_queue() {
					
	}

	using memory_edit_cb_t = std::function<void()>;


	#include "memory_bll_settings.h"
protected:
	#include <BLL/BLL.h>
	write_queue_t write_queue;
public:

	using nr_t = write_queue_NodeReference_t;

	nr_t push_back(const memory_edit_cb_t& cb) {
		auto nr = write_queue.NewNodeLast();
		write_queue[nr].cb = cb;
		return nr;
	}

	void process(fan::vulkan::context_t& context) {
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
	VkBuffer buffer = nullptr;
	VkDeviceMemory device_memory = nullptr;
};

template <typename nr_t, typename instance_id_t>
struct memory_common_t {
	static constexpr auto buffer_count = fan::vulkan::max_frames_in_flight;

	struct index_t {
		nr_t nr;
		instance_id_t i;
	};

	memory_t memory[buffer_count];

	memory_write_queue_t::memory_edit_cb_t write_cb;

	void open(fan::vulkan::context_t& context, const memory_write_queue_t::memory_edit_cb_t& cb) {
		write_cb = cb;
		queued = false;

		m_min_edit = 0xFFFFFFFFFFFFFFFF;
		//context <- uniform_block <-> uniform_write_queue <- loco
		m_max_edit = 0x00000000;
	}
	void close(fan::vulkan::context_t& context) {
		if (is_queued()) {
			context.memory_queue.erase(m_edit_index);
		}

		for (uint32_t i = 0; i < fan::vulkan::max_frames_in_flight; ++i) {
			vkDestroyBuffer(context.device, memory[i].buffer, nullptr);
			vkFreeMemory(context.device, memory[i].device_memory, nullptr);
		}
	}

	bool is_queued() const {
		return queued;
	}

	void edit(fan::vulkan::context_t& context, const index_t& idx) {
		indices.push_back(idx);

		if (is_queued()) {
			return;
		}
		queued = true;
		m_edit_index = context.memory_queue.push_back(write_cb);
	}

	void edit(fan::vulkan::context_t& context, uint32_t begin, uint32_t end) {
		m_min_edit = fan::min(m_min_edit, begin);
		m_max_edit = fan::max(m_max_edit, end);

		if (is_queued()) {
			return;
		}
		queued = true;
		m_edit_index = context.memory_queue.push_back(write_cb);
	}

	void on_edit(fan::vulkan::context_t& context) {
		reset_edit();
	}

	void reset_edit() {
		queued = false;
		indices.clear();

		m_min_edit = 0xFFFFFFFFFFFFFFFF;
		m_max_edit = 0x00000000;
	}

	std::vector<index_t> indices;
	fan::vulkan::context_t::memory_write_queue_t::nr_t m_edit_index;

	uint64_t m_min_edit;
	uint64_t m_max_edit;

	bool queued = 0;
};