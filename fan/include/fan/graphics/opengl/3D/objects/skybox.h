#pragma once

#include <fan/graphics/opengl/gl_core.h>
#include <fan/graphics/shared_graphics.h>

namespace fan_3d {
  namespace opengl {

    struct skybox {

      struct loaded_skybox_properties_t {
        std::string_view left;
        std::string_view right;
        std::string_view top;
        std::string_view bottom;
        std::string_view front;
        std::string_view back;
      };

      struct loaded_skybox_t {
        uint32_t texture;
      };

      static loaded_skybox_t load_skybox(const loaded_skybox_properties_t& p) {
        loaded_skybox_t loaded_skybox;

        glGenTextures(1, &loaded_skybox.texture);
        glBindTexture(GL_TEXTURE_CUBE_MAP, loaded_skybox.texture);

        auto load_side = [] (std::string_view path, uint8_t i) {
          fan::webp::image_info_t info = fan::webp::load_image(path);
          glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA, info.size.x, info.size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, info.data);
          fan::webp::free_image(info.data);
        };

        load_side(p.right, 0);
        load_side(p.left, 1);
        load_side(p.top, 2);
        load_side(p.bottom, 3);
        load_side(p.front, 4);
        load_side(p.back, 5);

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        return loaded_skybox;
      }

      struct properties_t {
        loaded_skybox_t loaded_skybox;
      };

      void open(fan::opengl::context_t* context) {
        m_draw_node_reference = fan::uninitialized;
        #if fan_debug >= fan_debug_low
          m_texture = 0;
        #endif

        m_shader.open();

        m_shader.set_vertex(
        #include <fan/graphics/glsl/opengl/3D/objects/skybox.vs>
        );

        m_shader.set_fragment(
        #include <fan/graphics/glsl/opengl/3D/objects/skybox.fs>
        );

        m_shader.compile();
      }
      void close(fan::opengl::context_t* context) {
        #if fan_debug >= fan_debug_low
        if (m_texture == 0) {
          fan::throw_error("trying to delete invalid texture");
        }
        #endif

        glDeleteTextures(1, &m_texture);
      }

      void set(fan::opengl::context_t* context, const properties_t& properties) {
        m_texture = properties.loaded_skybox.texture;
      }

      void enable_draw(fan::opengl::context_t* context) {
        m_draw_node_reference = context->enable_draw(this, [](fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
      }
      void disable_draw(fan::opengl::context_t* context) {
        #if fan_debug >= fan_debug_low
        if (m_draw_node_reference == fan::uninitialized) {
          fan::throw_error("trying to disable non enabled draw");
        }
        #endif
        context->disable_draw(m_draw_node_reference);
      }

    protected:

      void draw(fan::opengl::context_t* context) {

        context->set_depth_test(true);

        glDepthFunc(GL_LEQUAL);
        fan::mat4 projection(1);
        projection = fan::math::perspective<fan::mat4>(fan::math::radians(90.0), (f32_t)context->viewport_size.x / (f32_t)context->viewport_size.y, 0.1f, 1000.0f);

        fan::mat4 view(1);
        view = context->camera.get_view_matrix();

        m_shader.use();
        m_shader.set_projection(projection);
        m_shader.set_view(view);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, m_texture);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glDepthFunc(GL_LESS);
      }

      uint32_t m_texture;
      uint32_t m_draw_node_reference;
      fan::shader_t m_shader;
    };

  }
}