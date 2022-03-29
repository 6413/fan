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

      static loaded_skybox_t load_skybox(fan::opengl::context_t* context, const loaded_skybox_properties_t& p) {
        loaded_skybox_t loaded_skybox;

        context->opengl.glGenTextures(1, &loaded_skybox.texture);
        context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_CUBE_MAP, loaded_skybox.texture);

        auto load_side = [] (fan::opengl::context_t* context, std::string_view path, uint8_t i) {
          fan::webp::image_info_t info = fan::webp::load_image(path);
          context->opengl.glTexImage2D(fan::opengl::GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, fan::opengl::GL_RGBA, info.size.x, info.size.y, 0, fan::opengl::GL_RGBA, fan::opengl::GL_UNSIGNED_BYTE, info.data);
          fan::webp::free_image(info.data);
        };

        load_side(context, p.right, 0);
        load_side(context, p.left, 1);
        load_side(context, p.top, 2);
        load_side(context, p.bottom, 3);
        load_side(context, p.front, 4);
        load_side(context, p.back, 5);

        context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_CUBE_MAP, fan::opengl::GL_TEXTURE_MIN_FILTER, fan::opengl::GL_LINEAR);
        context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_CUBE_MAP, fan::opengl::GL_TEXTURE_MAG_FILTER, fan::opengl::GL_LINEAR);
        context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_CUBE_MAP, fan::opengl::GL_TEXTURE_WRAP_S, fan::opengl::GL_CLAMP_TO_EDGE);
        context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_CUBE_MAP, fan::opengl::GL_TEXTURE_WRAP_T, fan::opengl::GL_CLAMP_TO_EDGE);
        context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_CUBE_MAP, fan::opengl::GL_TEXTURE_WRAP_R, fan::opengl::GL_CLAMP_TO_EDGE);

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

        m_shader.open(context);

        m_shader.set_vertex(
          context, 
          #include <fan/graphics/glsl/opengl/3D/objects/skybox.vs>
        );

        m_shader.set_fragment(
          context, 
          #include <fan/graphics/glsl/opengl/3D/objects/skybox.fs>
        );

        m_shader.compile(context);
      }
      void close(fan::opengl::context_t* context) {
        #if fan_debug >= fan_debug_low
        if (m_texture == 0) {
          fan::throw_error("trying to delete invalid texture");
        }
        #endif

        context->opengl.glDeleteTextures(1, &m_texture);
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

        context->opengl.glDepthFunc(fan::opengl::GL_LEQUAL);
        fan::mat4 projection(1);
        projection = fan::math::perspective<fan::mat4>(fan::math::radians(90.0), (f32_t)context->viewport_size.x / (f32_t)context->viewport_size.y, 0.1f, 1000.0f);

        fan::mat4 view(1);
        view = context->camera.get_view_matrix();

        m_shader.use(context);
        m_shader.set_projection(context, projection);
        m_shader.set_view(context, view);

        context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
        context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_CUBE_MAP, m_texture);
        context->opengl.glDrawArrays(fan::opengl::GL_TRIANGLES, 0, 36);
        context->opengl.glDepthFunc(fan::opengl::GL_LESS);
      }

      uint32_t m_texture;
      uint32_t m_draw_node_reference;
      fan::shader_t m_shader;
    };

  }
}