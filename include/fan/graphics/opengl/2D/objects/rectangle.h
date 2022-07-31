#pragma once

#include _FAN_PATH(graphics/opengl/gl_core.h)
#include _FAN_PATH(graphics/opengl/gl_shader.h)
#include _FAN_PATH(graphics/shared_graphics.h)
#include _FAN_PATH(physics/collision/rectangle.h)

namespace fan_2d {
  namespace opengl {

    struct rectangle_t {

      template<typename T>
      inline void freeContainer(T& p_container)
      {
        T empty;
        using std::swap;
        swap(p_container, empty);
      }

      using draw_cb_t = void(*)(fan::opengl::context_t* context, rectangle_t*, void*);

      struct instance_t {
        fan::vec3 position = 0;
      private:
        f32_t pad[1];
      public:
        fan::vec2 size = 0;
        fan::vec2 rotation_point = 0;
        fan::color color = fan::colors::white;
        fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
        f32_t angle = 0;
      };

      struct properties_t : instance_t {

        fan::opengl::matrices_t* matrix;

        using type_t = rectangle_t;
      };

      // todo remove, pointers change
      static constexpr uint32_t reserve = 4000;

      static constexpr uint32_t blocks_array_size = 0x01000000;
      static constexpr uint32_t max_instance_size = 256;

      struct id_t{
        id_t(fan::opengl::cid_t* cid) {
          ba_id = cid->id / blocks_array_size;
          block = (cid->id & (blocks_array_size - 1)) / max_instance_size;
          instance = cid->id % max_instance_size;
        }
        uint32_t ba_id;
        uint32_t block;
        uint32_t instance;
      };

      void open(fan::opengl::context_t* context) {
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

        blocks_array.open();
        // todo block pointers are changing help
        blocks_array.reserve(reserve);

        m_draw_node_reference = fan::uninitialized;
        draw_cb = [](fan::opengl::context_t* context, rectangle_t*, void*) {};

        query = new std::unordered_map<uintptr_t, uint32_t>;
      }
      void close(fan::opengl::context_t* context) {
        m_shader.close(context);
        for (auto& it : blocks_array) {
          for (uint32_t i = 0; i < blocks_array.size(); i++) {
            it.blocks[i].uniform_buffer.close(context);
          }
        }
        blocks_array.close();
        delete query;
      }

      void push_back(fan::opengl::context_t* context, fan::opengl::cid_t* cid, const properties_t& p) {
        auto found = query->find((uintptr_t)p.matrix);
        if (found == query->end()) {
          query->insert({ (uintptr_t)p.matrix, blocks_array.size() });
          fan::hector_t<block_t> h;
          h.open();
          h.reserve(reserve);
          blocks_array_t ba;
          ba.blocks = h;
          ba.matrix = p.matrix;
          blocks_array.push_back(ba);
          found = query->find((uintptr_t)p.matrix);
        }
        uint32_t ba_id = found->second;
        fan::hector_t<block_t> &blocks = blocks_array[ba_id].blocks;

        instance_t it = p;

        uint32_t block_id = blocks.size() - 1;

        if (block_id == (uint32_t)-1 || blocks[block_id].uniform_buffer.size() == max_instance_size) {
          blocks.push_back({});
          block_id++;
          blocks[block_id].uniform_buffer.open(context);
          blocks[block_id].uniform_buffer.init_uniform_block(context, m_shader.id, "instance_t");
        }

        blocks[block_id].uniform_buffer.push_ram_instance(context, it);

        const uint32_t instance_id = blocks[block_id].uniform_buffer.size() - 1;

        blocks[block_id].cid[instance_id] = cid;

        blocks[block_id].uniform_buffer.common.edit(
          context,
          instance_id * sizeof(instance_t),
          instance_id * sizeof(instance_t) + sizeof(instance_t)
        );

        cid->id = ba_id * blocks_array_size + block_id * max_instance_size + instance_id;
      }
      void erase(fan::opengl::context_t* context, const id_t& id) {
        
       /* #if fan_debug >= fan_debug_medium
        if (block_id >= blocks.size()) {
          fan::throw_error("invalid access");
        }
        if (instance_id >= blocks[block_id].uniform_buffer.size()) {
          fan::throw_error("invalid access");
        }
        #endif*/

        if (id.block == blocks_array.size() - 1 && id.instance == blocks_array[id.ba_id].blocks.ge()->uniform_buffer.size() - 1) {
          blocks_array[id.ba_id].blocks[id.block].uniform_buffer.common.m_size -= blocks_array[id.ba_id].blocks[id.block].uniform_buffer.common.buffer_bytes_size;
          if (blocks_array[id.ba_id].blocks[id.block].uniform_buffer.size() == 0) {
            blocks_array[id.ba_id].blocks[id.block].uniform_buffer.close(context);
            blocks_array.m_size -= 1;
          }
          return;
        }

        uint32_t last_block_id = blocks_array.size() - 1;
        uint32_t last_instance_id = blocks_array[id.ba_id].blocks[last_block_id].uniform_buffer.size() - 1;

        instance_t* last_instance_data = blocks_array[id.ba_id].blocks[last_block_id].uniform_buffer.get_instance(context, last_instance_id);

        blocks_array[id.ba_id].blocks[id.block].uniform_buffer.edit_ram_instance(
          context,
          id.instance,
          last_instance_data,
          0,
          sizeof(instance_t)
        );

        blocks_array[id.ba_id].blocks[last_block_id].uniform_buffer.common.m_size -= blocks_array[id.ba_id].blocks[last_block_id].uniform_buffer.common.buffer_bytes_size;

        blocks_array[id.ba_id].blocks[id.block].cid[id.instance] = blocks_array[id.ba_id].blocks[last_block_id].cid[last_instance_id];
        blocks_array[id.ba_id].blocks[id.block].cid[id.instance]->id = id.ba_id * blocks_array_size + id.block * max_instance_size + id.instance;

        if (blocks_array[id.ba_id].blocks[last_block_id].uniform_buffer.size() == 0) {
          blocks_array[id.ba_id].blocks[last_block_id].uniform_buffer.close(context);
          blocks_array[id.ba_id].blocks.m_size -= 1;
        }
        blocks_array[id.ba_id].blocks[id.block].uniform_buffer.common.edit(
          context,
          id.instance * sizeof(instance_t),
          id.instance * sizeof(instance_t) + sizeof(instance_t)
        );
      }

      void enable_draw(fan::opengl::context_t* context) {

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

        if (draw_cb) {
          draw_cb(context, this, draw_userdata);
        }

        for (auto& it : blocks_array) {
          m_shader.set_matrices(context, it.matrix);
          for (uint32_t i = 0; i < it.blocks.size(); i++) {
            it.blocks[i].uniform_buffer.bind_buffer_range(context, it.blocks[i].uniform_buffer.size());

            it.blocks[i].uniform_buffer.draw(
              context,
              0 * 6,
              it.blocks[i].uniform_buffer.size() * 6
            );
          }
        }
      }

      template <typename T>
      T get(fan::opengl::context_t* context, const id_t& id, T instance_t::*member) {
        return blocks_array[id.ba_id].blocks[id.block].uniform_buffer.get_instance(context, id.instance)->*member;
      }
      template <typename T, typename T2>
      void set(fan::opengl::context_t* context, const id_t& id, T instance_t::*member, const T2& value) {
        blocks_array[id.ba_id].blocks[id.block].uniform_buffer.edit_ram_instance(context, id.instance, (T*)&value, fan::ofof<instance_t, T>(member), sizeof(T));
        blocks_array[id.ba_id].blocks[id.block].uniform_buffer.common.edit(
          context,
          id.instance * sizeof(instance_t) + fan::ofof<instance_t, T>(member),
          id.instance * sizeof(instance_t) + fan::ofof<instance_t, T>(member) + sizeof(T)
        );
      }

      void set_vertex(fan::opengl::context_t* context, const std::string& str) {
        m_shader.set_vertex(context, str);
      }
      void set_fragment(fan::opengl::context_t* context, const std::string& str) {
        m_shader.set_fragment(context, str);
      }
      void compile(fan::opengl::context_t* context) {
        m_shader.compile(context);
      }

      void set_draw_cb(fan::opengl::context_t* context, draw_cb_t draw_cb_, void* userptr = 0) {
        draw_cb = draw_cb_;
        if (userptr != nullptr) {
          draw_userdata = userptr;
        }
      }
      void set_draw_cb_userptr(fan::opengl::context_t* context, void* userptr) {
        draw_userdata = userptr;
      }

      fan::shader_t m_shader;


      struct block_t {
        fan::opengl::core::uniform_block_t<instance_t, max_instance_size> uniform_buffer;
        fan::opengl::cid_t* cid[max_instance_size];
      };
      uint32_t m_draw_node_reference;

      struct blocks_array_t {
        fan::opengl::matrices_t* matrix;
        fan::hector_t<block_t> blocks;
      };

      fan::hector_t<blocks_array_t> blocks_array;

      std::unordered_map<uintptr_t, uint32_t>* query;

      draw_cb_t draw_cb;
      void* draw_userdata;
    };
  }
}