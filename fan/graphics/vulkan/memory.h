struct memory_write_queue_t {
  memory_write_queue_t() : write_queue() {}

  using memory_edit_cb_t = void (*)(fan::vulkan::context_t&, void*);

  struct node_data_t {
    memory_edit_cb_t cb = nullptr;
    void* user_data = nullptr;
  };

  #include "memory_bll_settings.h"
protected:
  #include <BLL/BLL.h>
  write_queue_t write_queue;
public:
  using nr_t = write_queue_NodeReference_t;
  std::vector<node_data_t> m_scratch_nodes;

  nr_t push_back(memory_edit_cb_t cb, void* user_data) {
    auto nr = write_queue.NewNodeLast();
    write_queue[nr].cb = cb;
    write_queue[nr].user_data = user_data;
    return nr;
  }

  void process(fan::vulkan::context_t& context) {
    m_scratch_nodes.clear();
    auto it = write_queue.GetNodeFirst();
    while (it != write_queue.dst) {
      write_queue.StartSafeNext(it);
      m_scratch_nodes.push_back({write_queue[it].cb, write_queue[it].user_data});
      it = write_queue.EndSafeNext();
    }

    write_queue.Clear();

    for (auto& item : m_scratch_nodes) {
      if (item.cb) { item.cb(context, item.user_data); }
    }
  }

  void erase(nr_t node_reference) {
    write_queue.unlrec(node_reference);
  }

  void clear() {
    write_queue.Clear();
  }
};

struct memory_t {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation device_memory = VK_NULL_HANDLE;
};

template <typename nr_t, typename instance_id_t>
struct memory_common_t {
  using write_cb_t = void (*)(fan::vulkan::context_t&, void*);

  write_cb_t write_cb = nullptr;
  void* user_data = nullptr;
  nr_t shape_nr{};
  memory_write_queue_t::nr_t m_edit_index{};
  uint64_t dirty_frames = 0;
  bool queued = false;
  std::vector<instance_id_t> indices;
  uint64_t m_min_edit = -1;
  uint64_t m_max_edit = 0;
  memory_t memory[fan::vulkan::max_frames_in_flight]{};

  void open(fan::vulkan::context_t& context, write_cb_t cb, void* ptr = nullptr) {
    write_cb = cb;
    user_data = ptr;
  }

  void close(fan::vulkan::context_t& context) {
    if (queued) {
      context.memory_queue.erase(m_edit_index);
      queued = false;
    }
    for (std::uint32_t i = 0; i < fan::vulkan::max_frames_in_flight; ++i) {
      if (memory[i].buffer != VK_NULL_HANDLE) {
        context.destroy_buffer(memory[i].buffer, memory[i].device_memory);
        memory[i].buffer = VK_NULL_HANDLE;
      }
    }
  }

  bool is_queued() const {
    return queued;
  }

  void queue(fan::vulkan::context_t& context) {
    dirty_frames |= (std::uint64_t(1) << fan::vulkan::max_frames_in_flight) - 1;

    if (is_queued()) {
      return;
    }
    queued = true;
    m_edit_index = context.memory_queue.push_back(write_cb, user_data);
  }

  void edit(fan::vulkan::context_t& context, const instance_id_t& idx) {
    indices.push_back(idx);
    queue(context);
  }

  void edit(fan::vulkan::context_t& context, std::uint64_t begin, std::uint64_t end) {
    m_min_edit = std::min(m_min_edit, begin);
    m_max_edit = std::max(m_max_edit, end);
    queue(context);
  }

  bool is_current_frame_dirty(fan::vulkan::context_t& context) const {
    return dirty_frames & (std::uint64_t(1) << context.current_frame);
  }

  void on_edit(fan::vulkan::context_t& context) {
    dirty_frames &= ~(std::uint64_t(1) << context.current_frame);

    if (dirty_frames) {
      queued = false;
      queue(context);
      return;
    }

    queued = false;
    indices.clear();
    m_min_edit = -1;
    m_max_edit = 0;
  }
};