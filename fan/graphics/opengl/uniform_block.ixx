module;

#include <fan/types/types.h>
#include <fan/math/math.h>

#include <functional>

namespace detail{
  using memory_edit_cb_t = std::function<void()>;
  #define BLL_API inline
  #include "memory_bll_settings.h"
  #include <BLL/BLL.h>
}

export module fan:graphics.opengl.uniform_block;

import :graphics.opengl.core;

export namespace fan {
  namespace opengl {
    namespace core {


      struct memory_write_queue_t {

        using memory_edit_cb_t = detail::memory_edit_cb_t;

        memory_write_queue_t() = default;

        detail::write_queue_t write_queue;

        using nr_t = detail::write_queue_NodeReference_t;

        nr_t push_back(const memory_edit_cb_t& cb) {
          auto nr = write_queue.NewNodeLast();
          write_queue[nr].cb = cb;
          return nr;
        }

        void process(fan::opengl::context_t& context) {
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

      struct memory_common_t {

        fan::opengl::core::vao_t m_vao;
        fan::opengl::core::vbo_t m_vbo;

        memory_write_queue_t::memory_edit_cb_t write_cb;

        void open(fan::opengl::context_t& context, uint32_t target, const memory_write_queue_t::memory_edit_cb_t& cb) {

          m_vao.open(context);
          m_vbo.open(context, target);

          write_cb = cb;

          queued = false;

          m_min_edit = 0xFFFFFFFFFFFFFFFF;

          m_max_edit = 0x00000000;
        }
        void close(fan::opengl::context_t& context, memory_write_queue_t* queue) {
          if (is_queued()) {
            queue->erase(m_edit_index);
          }

          m_vao.close(context);
        }

        bool is_queued() const {
          return queued;
        }

        void edit(fan::opengl::context_t& context, memory_write_queue_t* queue, uint32_t begin, uint32_t end) {

          m_min_edit = fan::min(m_min_edit, begin);
          m_max_edit = fan::max(m_max_edit, end);

          if (is_queued()) {
            return;
          }
          queued = true;
          m_edit_index = queue->push_back(write_cb);

          // context.process();
        }

        void on_edit(fan::opengl::context_t& context) {
          reset_edit();
        }

        void reset_edit() {
          m_min_edit = 0xFFFFFFFFFFFFFFFF;
          m_max_edit = 0x00000000;

          queued = false;
        }

        memory_write_queue_t::nr_t m_edit_index;

        uint64_t m_min_edit;
        uint64_t m_max_edit;

        bool queued;
      };
    }
  }
}