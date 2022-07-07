#pragma once

#include _FAN_PATH(graphics/opengl/gl_core.h)
#include _FAN_PATH(graphics/opengl/gl_shader.h)
#include _FAN_PATH(graphics/shared_graphics.h)
#include _FAN_PATH(physics/collision/rectangle.h)

namespace fan_2d {
  namespace opengl {

    template <typename T_user_global_data, typename T_user_instance_data>
    struct rectangle_t {

      using user_global_data_t = T_user_global_data;
      using user_instance_data_t = T_user_instance_data;

      using move_cb_t = void(*)(rectangle_t*, uint32_t src, uint32_t dst, user_instance_data_t*);

      struct instance_t {
        fan::vec2 position = 0;
        fan::vec2 size = 0;
        fan::color color = fan::colors::white;
        fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
        f32_t angle = 0;
        fan::vec2 rotation_point = 0;
        fan::vec2 pad;
      };

      struct properties_t : instance_t {

        using type_t = rectangle_t;

        user_instance_data_t data;
      };

      static constexpr uint32_t max_instance_size = 256;

      struct id_t{
        id_t(uint32_t id) {
          block = id / max_instance_size;
          instance = id % max_instance_size;
        }
        uint32_t block;
        uint32_t instance;
      };

      void open(fan::opengl::context_t* context, move_cb_t move_cb_, const user_global_data_t& gd) {
        m_shader.open(context);

        m_shader.set_vertex(
          context,
#include _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.vs)
        );

        m_shader.set_fragment(
          context,
#include _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.fs)
        );

        m_shader.compile(context);

        blocks.open();

        m_draw_node_reference = fan::uninitialized;

        user_global_data = gd;
        move_cb = move_cb_;
      }
      void close(fan::opengl::context_t* context) {
        m_shader.close(context);
        for (uint32_t i = 0; i < blocks.size(); i++) {
          blocks[i].uniform_buffer.close(context);
        }
        blocks.close();
      }

      uint32_t push_back(fan::opengl::context_t* context, const properties_t& p) {
        instance_t it = p;

        uint32_t i = 0;

        for (; i < blocks.size(); i++) {
          if (blocks[i].uniform_buffer.size() != max_instance_size) {
            break;
          }
        }

        if (i == blocks.size()) {
          blocks.push_back({});
          blocks[i].uniform_buffer.open(context);
          blocks[i].uniform_buffer.bind_uniform_block(context, m_shader.id, "instance_t");
        }

        blocks[i].uniform_buffer.push_ram_instance(context, it);
        blocks[i].user_instance_data[blocks[i].uniform_buffer.size() - 1] = p.data;

        blocks[i].uniform_buffer.write_vram_all(context); // TODO fix

        return i * max_instance_size + (blocks[i].uniform_buffer.size() - 1);
      }
      void erase(fan::opengl::context_t* context, uint32_t id) {

        uint32_t block_id = id / max_instance_size;
        uint32_t instance_id = id % max_instance_size;

        if (block_id == blocks.size() - 1 && instance_id == blocks.ge()->uniform_buffer.size() - 1) {
          blocks[block_id].uniform_buffer.common.m_size -= blocks[block_id].uniform_buffer.common.buffer_bytes_size;
          if (blocks[block_id].uniform_buffer.size() == 0) {
            blocks[block_id].uniform_buffer.close(context);
            blocks.m_size -= 1;
          }
          return;
        }

        uint32_t last_block_id = blocks.size() - 1;
        uint32_t last_instance_id = blocks[last_block_id].uniform_buffer.size() - 1;

        instance_t* data = blocks[block_id].uniform_buffer.get_instance(context, last_instance_id);

        blocks[block_id].uniform_buffer.edit_ram_instance(
          context,
          instance_id,
          data,
          0,
          sizeof(instance_t)
        );

        blocks[block_id].uniform_buffer.common.edit(
          context,
          instance_id,
          instance_id + 1
        );

        blocks[last_block_id].uniform_buffer.common.m_size -= blocks[block_id].uniform_buffer.common.buffer_bytes_size;

        blocks[block_id].user_instance_data[instance_id] = blocks[last_block_id].user_instance_data[last_instance_id];

        if (blocks[last_block_id].uniform_buffer.size() == 0) {
          blocks[last_block_id].uniform_buffer.close(context);
          blocks.m_size -= 1;
        }

        move_cb(
          this,
          last_instance_id + last_block_id * max_instance_size,
          id,
          &blocks[block_id].user_instance_data[instance_id]
        );
      }

      void enable_draw(fan::opengl::context_t* context) {
        this->draw(context);

#if fan_debug >= fan_debug_low
        if (m_draw_node_reference != fan::uninitialized) {
          fan::throw_error("trying to call enable_draw twice");
        }
#endif

        m_draw_node_reference = context->enable_draw(this, [](fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
      }
      void disable_draw(fan::opengl::context_t* context) {
#if fan_debug >= fan_debug_low
        if (m_draw_node_reference == fan::uninitialized) {
          fan::throw_error("trying to disable unenabled draw call");
        }
#endif
        context->disable_draw(m_draw_node_reference);
      }

      void draw(fan::opengl::context_t* context) {
        m_shader.use(context);

        for (uint32_t i = 0; i < blocks.size(); i++) {
          blocks[i].uniform_buffer.bind_buffer_range(context, blocks[i].uniform_buffer.size());

          blocks[i].uniform_buffer.draw(
            context,
            0,
            blocks[i].uniform_buffer.size() * 6
          );
        }
      }

      void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
        m_shader.bind_matrices(context, matrices);
      }

      template <typename T>
      T get(fan::opengl::context_t* context, const id_t& id, T instance_t::*member) {
        return blocks[id.block].uniform_buffer.get_instance(context, id.instance)->*member;
      }
      template <typename T>
      void set(fan::opengl::context_t* context, const id_t& id, T instance_t::*member, const T& value) {
        blocks[id.block].uniform_buffer.edit_ram_instance(context, id.instance, &value, fan::ofof<instance_t, T>(member), sizeof(T));
      }

      void set_user_instance_data(fan::opengl::context_t* context, const id_t& id, const user_instance_data_t& user_instance_data) {
        blocks[id.block].user_instance_data[id.instance] = user_instance_data;
      }

      user_instance_data_t get_user_instance_data(const id_t& id) {
        return blocks[id.block].user_instance_data[id.instance];
      }
      user_global_data_t get_user_global_data() {
        return user_global_data;
      }

      fan::shader_t m_shader;

      struct block_t {
        fan::opengl::core::uniform_block_t<instance_t, max_instance_size> uniform_buffer;
        user_instance_data_t user_instance_data[max_instance_size];
      };
      uint32_t m_draw_node_reference;

      fan::hector_t<block_t> blocks;

      user_global_data_t user_global_data;
      move_cb_t move_cb;
    };
  }
}