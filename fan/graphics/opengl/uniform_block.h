#pragma once

#include <fan/graphics/opengl/gl_core.h>

namespace fan {
  namespace opengl {
    namespace core {

      struct memory_write_queue_t {

        memory_write_queue_t() = default;

        using memory_edit_cb_t = fan::function_t<void()>;


        #include "memory_bll_settings.h"
      protected:
        #include _FAN_PATH(BLL/BLL.h)
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

      template <typename type_t, uint32_t element_size>
      struct uniform_block_t {

        static constexpr uint32_t element_byte_size = element_size;

        uniform_block_t() = default;

        struct open_properties_t {
          open_properties_t() {}

          uint32_t target = fan::opengl::GL_UNIFORM_BUFFER;
          uint32_t usage = fan::opengl::GL_DYNAMIC_DRAW;
        }op;

        void open(fan::opengl::context_t& context, open_properties_t op_ = open_properties_t()) {
          op = op_;
          m_size = 0;
          common.open(context, op.target, [&context, this] {
            uint64_t src = common.m_min_edit;
            uint64_t dst = common.m_max_edit;

            auto cp_buffer = buffer;

            cp_buffer += src;

            common.m_vbo.edit_buffer(context, cp_buffer, src, dst - src);

            common.on_edit(context);
          });
          common.m_vbo.write_buffer(context, 0, sizeof(type_t) * element_size);
        }

        void close(fan::opengl::context_t& context, memory_write_queue_t* write_queue) {
          common.m_vbo.close(context);

          common.close(context, write_queue);
        }

        void bind_buffer_range(fan::opengl::context_t& context, uint32_t bytes_size) {
          common.m_vbo.bind_buffer_range(context, bytes_size * sizeof(type_t));
        }

        void bind(fan::opengl::context_t& context) const {
          common.m_vbo.bind(context);
        }
        void unbind() const {
          //glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        void push_ram_instance(fan::opengl::context_t& context, const type_t& data) {
          std::memmove(&buffer[m_size], (void*)&data, sizeof(type_t));
          m_size += sizeof(type_t);
        }

        type_t* get_instance(fan::opengl::context_t& context, uintptr_t i) {
          return (type_t*)&buffer[i * sizeof(type_t)];
        }
        void get_vram_instance(fan::opengl::context_t& context, type_t* data, uintptr_t i) {
          common.m_vbo.get_vram_instance(context, data, sizeof(type_t), i * sizeof(type_t));
        }
        template <typename T>
        void edit_instance(fan::opengl::context_t& context, memory_write_queue_t* wq, uintptr_t i, T member, auto value) {
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
        void copy_instance(fan::opengl::context_t& context, memory_write_queue_t* wq, uint32_t i, type_t* instance) {
          #if fan_debug >= fan_debug_low
          if (i * sizeof(type_t) >= m_size) {
            fan::throw_error("uninitialized access");
          }
          #endif
          std::memmove(buffer + i * sizeof(type_t), instance, sizeof(type_t));

          common.edit(context, wq, i * sizeof(type_t), i * sizeof(type_t) + sizeof(type_t));
        }

        void init_uniform_block(fan::opengl::context_t& context, uint32_t program, const char* name, uint32_t buffer_index = 0) {
          uint32_t index = context.opengl.call(context.opengl.glGetUniformBlockIndex, program, name);
          #if fan_debug >= fan_debug_low
          if (index == fan::uninitialized) {
            fan::throw_error(fan::string("failed to initialize uniform block:") + name);
          }
          #endif

          context.opengl.call(context.opengl.glUniformBlockBinding, program, index, buffer_index);
        }

        void draw(fan::opengl::context_t& context, uint32_t begin, uint32_t count, uint32_t draw_mode = fan::opengl::GL_TRIANGLES) {

          common.m_vao.bind(context);

          //context.opengl.call(context.opengl.glDrawArraysInstanced, fan::opengl::GL_TRIANGLES, begin, count / 6, 6);

          context.opengl.call(context.opengl.glDrawArrays, draw_mode, begin, count);
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