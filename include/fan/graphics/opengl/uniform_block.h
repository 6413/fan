#pragma once

namespace fan {
  namespace opengl {
    namespace core {

      struct memory_write_queue_t {

        memory_write_queue_t() = default;

        using memory_edit_cb_t = fan::function_t<void()>;


        #include "memory_bll_settings.h"
      protected:
        #include _FAN_PATH(BLL/BLL.h)
        write_queue_t write_queue;
      public:

        using nr_t = write_queue_NodeReference_t;

        nr_t push_back(const memory_edit_cb_t& cb) {
          auto nr = write_queue.NewNodeLast();
          write_queue[nr].cb = cb;
          return nr;
        }

        void process(fan::opengl::context_t* context) {
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

        uint32_t m_vbo;
        fan::opengl::core::vao_t m_vao;

        memory_write_queue_t::memory_edit_cb_t write_cb;

        void open(fan::opengl::context_t* context, const memory_write_queue_t::memory_edit_cb_t& cb) {

          m_vao.open(context);

          write_cb = cb;

          queued = false;

          m_min_edit = 0xFFFFFFFFFFFFFFFF;
          
          m_max_edit = 0x00000000;
        }
        void close(fan::opengl::context_t* context, memory_write_queue_t* queue) {
          if (is_queued()) {
            queue->erase(m_edit_index);
          }

          m_vao.close(context);
        }

        bool is_queued() const {
          return queued;
        }

        void edit(fan::opengl::context_t* context, memory_write_queue_t* queue, uint32_t begin, uint32_t end) {

          m_min_edit = fan::min(m_min_edit, begin);
          m_max_edit = fan::max(m_max_edit, end);

          if (is_queued()) {
            return;
          }
          queued = true;
          m_edit_index = queue->push_back(write_cb);

          // context->process();
        }

        void on_edit(fan::opengl::context_t* context) {
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

      template <typename type_t, uint32_t element_size>
      struct uniform_block_t {

        static constexpr uint32_t element_byte_size = element_size;

        uniform_block_t() = default;

        struct open_properties_t {
          open_properties_t() {}

          uint32_t target = fan::opengl::GL_UNIFORM_BUFFER;
          uint32_t usage = fan::opengl::GL_DYNAMIC_DRAW;
        }op;

        void open(fan::opengl::context_t* context, open_properties_t op_ = open_properties_t()) {
          context->opengl.call(context->opengl.glGenBuffers, 1, &common.m_vbo);
          op = op_;
          m_size = 0;
          common.open(context, [context, this] {
            uint64_t src = common.m_min_edit;
            uint64_t dst = common.m_max_edit;

            auto cp_buffer = buffer;

            cp_buffer += src;

            fan::opengl::core::edit_glbuffer(context, common.m_vbo, cp_buffer, src, dst - src, fan::opengl::GL_UNIFORM_BUFFER);

            common.on_edit(context);
          });
          fan::opengl::core::write_glbuffer(context, common.m_vbo, 0, sizeof(type_t) * element_size, op.usage, op.target);
        }

        void close(fan::opengl::context_t* context, memory_write_queue_t* write_queue) {
          #if fan_debug >= fan_debug_low
          if (common.m_vbo == -1) {
            fan::throw_error("tried to remove non existent vbo");
          }
          #endif
          context->opengl.call(context->opengl.glDeleteBuffers, 1, &common.m_vbo);

          common.close(context, write_queue);
        }

        void bind_buffer_range(fan::opengl::context_t* context, uint32_t bytes_size) {
          context->opengl.call(context->opengl.glBindBufferRange, fan::opengl::GL_UNIFORM_BUFFER, 0, common.m_vbo, 0, bytes_size * sizeof(type_t));
        }

        void bind(fan::opengl::context_t* context) const {
          context->opengl.call(context->opengl.glBindBuffer, op.target, common.m_vbo);
        }
        void unbind() const {
          //glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        void push_ram_instance(fan::opengl::context_t* context, const type_t& data) {
          std::memmove(&buffer[m_size], (void*)&data, sizeof(type_t));
          m_size += sizeof(type_t);
        }

        type_t* get_instance(fan::opengl::context_t* context, uint32_t i) {
          return (type_t*)&buffer[i * sizeof(type_t)];
        }
        void get_vram_instance(fan::opengl::context_t* context, type_t* data, uint32_t i) {
          fan::opengl::core::get_glbuffer(context, data, common.m_vbo, sizeof(type_t), i * sizeof(type_t), op.target);
        }
        template <typename T>
        void edit_instance(fan::opengl::context_t* context, memory_write_queue_t* wq, uint32_t i, T member, auto value) {
          #if fan_debug >= fan_debug_low
         /* if (i * sizeof(type_t) >= common.m_size) {
            fan::throw_error("uninitialized access");
          }*/
          #endif
          #if fan_debug >= 2
            std::remove_reference_t<decltype(((type_t*)buffer)[i].*member)> _d = ((type_t*)buffer)[i].*member;
            _d = value;
            int64_t im = sizeof(_d);
            while(im--){
              if (((uint8_t*)&_d)[im] != ((uint8_t*)&(((type_t*)buffer)[i].*member))[im]) {
                break;
              }
            }
            if (im >= (int64_t)sizeof(value)) {
              fan::throw_error("invalid edit_instance");
            }
          #endif
          
          ((type_t*)buffer)[i].*member = value;

          common.edit(context, wq, i * sizeof(type_t), i * sizeof(type_t) + sizeof(type_t));
        }
        // for copying whole thing
        void copy_instance(fan::opengl::context_t* context, memory_write_queue_t* wq, uint32_t i, type_t* instance) {
          #if fan_debug >= fan_debug_low
          if (i * sizeof(type_t) >= m_size) {
            fan::throw_error("uninitialized access");
          }
          #endif
          std::memmove(buffer + i * sizeof(type_t), instance, sizeof(type_t));

          common.edit(context, wq, i * sizeof(type_t), i * sizeof(type_t) + sizeof(type_t));
        }

        void init_uniform_block(fan::opengl::context_t* context, uint32_t program, const char* name, uint32_t buffer_index = 0) {
          uint32_t index = context->opengl.call(context->opengl.glGetUniformBlockIndex, program, name);
          #if fan_debug >= fan_debug_low
          if (index == fan::uninitialized) {
            fan::throw_error(fan::string("failed to initialize uniform block:") + name);
          }
          #endif

          context->opengl.call(context->opengl.glUniformBlockBinding, program, index, buffer_index);
        }

        void draw(fan::opengl::context_t* context, uint32_t begin, uint32_t count, uint32_t draw_mode = fan::opengl::GL_TRIANGLES) {

          common.m_vao.bind(context);

          //context->opengl.call(context->opengl.glDrawArraysInstanced, fan::opengl::GL_TRIANGLES, begin, count / 6, 6);

          context->opengl.call(context->opengl.glDrawArrays, draw_mode, begin, count);
        }

        uint32_t size() const {
          return m_size / sizeof(type_t);
        }

        memory_common_t common;
        uint8_t buffer[element_size * sizeof(type_t)];
        uint32_t m_size;
      };
    }
  }
}