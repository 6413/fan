module;

#if defined(FAN_OPENGL)

#include <fan/utility.h>

#include <cstdlib>
#include <functional>

#endif

export module fan.graphics.opengl.uniform_block;

#if defined(FAN_OPENGL)

import fan.utility;
import fan.graphics.opengl.core;

namespace detail{
  using memory_edit_cb_t = std::function<void()>;
  #define BLL_API inline
  #include "memory_bll_settings.h"
  #include <BLL/BLL.h>
}

export namespace fan::opengl::core {
  struct memory_write_queue_t {
    using memory_edit_cb_t = detail::memory_edit_cb_t;
    using nr_t = detail::write_queue_NodeReference_t;

    memory_write_queue_t();
    nr_t push_back(const memory_edit_cb_t& cb);
    void process(fan::opengl::context_t& context);
    void erase(nr_t node_reference);
    void clear();

    detail::write_queue_t write_queue;
  };

  struct memory_common_t {
    void open(fan::opengl::context_t& context, uint32_t target, const memory_write_queue_t::memory_edit_cb_t& cb);
    void close(fan::opengl::context_t& context, memory_write_queue_t* queue);
    bool is_queued() const;
    void edit(fan::opengl::context_t& context, memory_write_queue_t* queue, uint32_t begin, uint32_t end);
    void on_edit(fan::opengl::context_t& context);
    void reset_edit();

    bool queued;
    memory_write_queue_t::nr_t m_edit_index;

    uint64_t m_min_edit;
    uint64_t m_max_edit;

    fan::opengl::core::vao_t m_vao;
    fan::opengl::core::vbo_t m_vbo;

    memory_write_queue_t::memory_edit_cb_t write_cb;
  };
}

#endif