#pragma once

#include _FAN_PATH(graphics/opengl/gl_core.h)

namespace fan {
  namespace opengl {
    namespace core {

      struct uniform_block_common_t;

      struct uniform_write_queue_t{
        void open() {
          write_queue.open();
        }
        void close() {
          write_queue.close();
        }

        uint32_t push_back(fan::opengl::core::uniform_block_common_t* block);

        void process(fan::opengl::context_t* context);

        void erase(uint32_t node_reference) {
          write_queue.erase(node_reference);
        }

        void clear() {
          write_queue.clear();
        }

      protected:
        bll_t<fan::opengl::core::uniform_block_common_t*> write_queue;
      };

      struct uniform_block_common_t {
        uint32_t m_vbo;
        fan::opengl::core::vao_t m_vao;
        uint32_t buffer_bytes_size;
        uint32_t m_size;

        void open(fan::opengl::context_t* context) {

          m_edit_index = fan::uninitialized;

          m_min_edit = 0xffffffff;
          //context <- uniform_block <-> uniform_write_queue <- loco
          m_max_edit = 0x00000000;

          m_size = 0;
          m_vao.open(context);
        }
        void close(fan::opengl::context_t* context, uniform_write_queue_t* queue) {
          if (is_queued()) {
            queue->erase(m_edit_index);
          }
          m_vao.close(context);
        }

        bool is_queued() const {
          return m_edit_index != fan::uninitialized;
        }

        void edit(fan::opengl::context_t* context, uniform_write_queue_t* queue, uint32_t begin, uint32_t end) {

          m_min_edit = std::min(m_min_edit, begin);
          m_max_edit = std::max(m_max_edit, end);

          if (is_queued()) {
            return;
          }
          m_edit_index = queue->push_back(this);

          // context->process();
        }

        void on_edit(fan::opengl::context_t* context) {
          reset_edit();
        }

        void reset_edit() {
          m_min_edit = 0xffffffff;
          m_max_edit = 0x00000000;

          m_edit_index = fan::uninitialized;
        }

        uint32_t m_edit_index;

        uint32_t m_min_edit;
        uint32_t m_max_edit;
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
          common.open(context);
          common.buffer_bytes_size = sizeof(type_t);
          fan::opengl::core::write_glbuffer(context, common.m_vbo, 0, sizeof(type_t) * element_size, op.usage, op.target);
        }

        void close(fan::opengl::context_t* context, uniform_write_queue_t* write_queue) {
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
          std::memmove(&buffer[common.m_size], (void*)&data, common.buffer_bytes_size);
          common.m_size += sizeof(type_t);
        }

        // uniform block preallocates
        /*void write_vram_all(fan::opengl::context_t* context) {

        common.m_vao.bind(context);

        this->bind(context);

        fan::opengl::core::write_glbuffer(context, common.m_vbo, (void*)buffer, common.m_size, op.usage, op.target);
        }*/

        type_t* get_instance(fan::opengl::context_t* context, uint32_t i) {
          return (type_t*)&buffer[i * sizeof(type_t)];
        }
        void get_vram_instance(fan::opengl::context_t* context, type_t* data, uint32_t i) {
          fan::opengl::core::get_glbuffer(context, data, common.m_vbo, sizeof(type_t), i * sizeof(type_t), op.target);
        }
        void edit_ram_instance(fan::opengl::context_t* context, uint32_t i, const void* data, uint32_t byte_offset, uint32_t sizeof_data) {
          #if fan_debug >= fan_debug_low
          if (i + byte_offset + sizeof_data > common.m_size) {
            fan::throw_error("invalid access");
          }
          #endif
          std::memmove(buffer + i * sizeof(type_t) + byte_offset, data, sizeof_data);
        }

        void init_uniform_block(fan::opengl::context_t* context, uint32_t program, const char* name, uint32_t buffer_index = 0) {
          uint32_t index = context->opengl.call(context->opengl.glGetUniformBlockIndex, program, name);
          #if fan_debug >= fan_debug_low
          if (index == fan::uninitialized) {
            fan::throw_error(std::string("failed to initialize uniform block:") + name);
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
          return common.m_size / sizeof(type_t);
        }

        uniform_block_common_t common;
        uint8_t buffer[element_size * sizeof(type_t)];
      };

      uint32_t uniform_write_queue_t::push_back(fan::opengl::core::uniform_block_common_t* block){
        return write_queue.push_back(block);
      }
      void uniform_write_queue_t::process(fan::opengl::context_t* context){
        uint32_t it = write_queue.begin();

        while (it != write_queue.end()) {

          uint64_t src = write_queue[it]->m_min_edit;
          uint64_t dst = write_queue[it]->m_max_edit;

          uint8_t* buffer = (uint8_t*)&write_queue[it][1];

          buffer += src;

          fan::opengl::core::edit_glbuffer(context, write_queue[it]->m_vbo, buffer, src, dst - src, fan::opengl::GL_UNIFORM_BUFFER);

          write_queue[it]->on_edit(context);

          it = write_queue.next(it);
        }

        write_queue.clear();
      }
    }
  }
}