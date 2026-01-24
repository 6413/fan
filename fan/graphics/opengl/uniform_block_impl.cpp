module;

#if defined(FAN_OPENGL)

#include <cstdint>
#include <algorithm>

#endif

module fan.graphics.opengl.uniform_block;

#if defined(FAN_OPENGL)

import fan.graphics.opengl.core;

namespace fan::opengl::core {
  memory_write_queue_t::memory_write_queue_t() = default;

  memory_write_queue_t::nr_t memory_write_queue_t::push_back(const memory_edit_cb_t& cb) {
    auto nr = write_queue.NewNodeLast();
    write_queue[nr].cb = cb;
    return nr;
  }

  void memory_write_queue_t::process(fan::opengl::context_t& context) {
    auto it = write_queue.GetNodeFirst();
    while (it != write_queue.dst) {
      write_queue.StartSafeNext(it);
      write_queue[it].cb();
      it = write_queue.EndSafeNext();
    }
    write_queue.Clear();
  }

  void memory_write_queue_t::erase(nr_t node_reference) {
    write_queue.unlrec(node_reference);
  }

  void memory_write_queue_t::clear() {
    write_queue.Clear();
  }

  void memory_common_t::open(fan::opengl::context_t& context, uint32_t target, const memory_write_queue_t::memory_edit_cb_t& cb) {
    m_vao.open(context);
    m_vbo.open(context, target);
    write_cb = cb;
    queued = false;
    m_min_edit = 0xFFFFFFFFFFFFFFFF;
    m_max_edit = 0x00000000;
  }

  void memory_common_t::close(fan::opengl::context_t& context, memory_write_queue_t* queue) {
    if (is_queued()) {
      queue->erase(m_edit_index);
    }
    m_vao.close(context);
  }

  bool memory_common_t::is_queued() const {
    return queued;
  }

  void memory_common_t::edit(fan::opengl::context_t& context, memory_write_queue_t* queue, uint32_t begin, uint32_t end) {
    m_min_edit = std::min(m_min_edit, (uint64_t)begin);
    m_max_edit = std::max(m_max_edit, (uint64_t)end);
    if (is_queued()) {
      return;
    }
    queued = true;
    m_edit_index = queue->push_back(write_cb);
  }

  void memory_common_t::on_edit(fan::opengl::context_t& context) {
    reset_edit();
  }

  void memory_common_t::reset_edit() {
    m_min_edit = 0xFFFFFFFFFFFFFFFF;
    m_max_edit = 0x00000000;
    queued = false;
  }
}

#endif