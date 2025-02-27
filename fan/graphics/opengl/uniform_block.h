#pragma once

#include <fan/graphics/opengl/core.h>

namespace fan {
  namespace opengl {
    namespace core {

      struct memory_write_queue_t {

        memory_write_queue_t() = default;

        using memory_edit_cb_t = std::function<void()>;


        #include "memory_bll_settings.h"
      protected:
        #include <BLL/BLL.h>
      public:
        write_queue_t write_queue;

        using nr_t = write_queue_NodeReference_t;

        nr_t push_back(const memory_edit_cb_t& cb);

        void process(fan::opengl::context_t& context);

        void erase(nr_t node_reference);

        void clear();
      };

      struct memory_common_t {

        fan::opengl::core::vao_t m_vao;
        fan::opengl::core::vbo_t m_vbo;

        memory_write_queue_t::memory_edit_cb_t write_cb;

        void open(fan::opengl::context_t& context, uint32_t target, const memory_write_queue_t::memory_edit_cb_t& cb);
        void close(fan::opengl::context_t& context, memory_write_queue_t* queue);

        bool is_queued() const;

        void edit(fan::opengl::context_t& context, memory_write_queue_t* queue, uint32_t begin, uint32_t end);

        void on_edit(fan::opengl::context_t& context);

        void reset_edit();

        memory_write_queue_t::nr_t m_edit_index;

        uint64_t m_min_edit;
        uint64_t m_max_edit;

        bool queued;
      };
    }
  }
}