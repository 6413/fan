#pragma once

#include _FAN_PATH(graphics/opengl/gl_core.h)
#include _FAN_PATH(graphics/opengl/gl_shader.h)
#include _FAN_PATH(graphics/shared_graphics.h)
#include _FAN_PATH(physics/collision/rectangle.h)

namespace fan_2d {
  namespace opengl {

    struct rectangle_t {

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

      static constexpr uint32_t max_instance_size = 128;

      struct block_properties_t {
        fan::opengl::matrices_list_NodeReference_t matrices_reference;
      };

      struct properties_t : instance_t {
        fan::opengl::matrices_t* matrices;

        union {
          struct {
            // sb block properties contents come here
            fan::opengl::matrices_list_NodeReference_t matrices_reference;
          };
          block_properties_t block_properties;
        };
      };

      void draw(fan::opengl::context_t* context) {
        m_shader.use(context);

        for (uint32_t i = 0; i < blocks.size(); i++) {
          blocks[i].uniform_buffer.bind_buffer_range(context, blocks[i].uniform_buffer.size());

          blocks[i].uniform_buffer.draw(
            context,
            0 * 6,
            blocks[i].uniform_buffer.size() * 6
          );
        }
      }

      #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.vs)
      #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.fs)
      #include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)
    };
  }
}