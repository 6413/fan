module;

#if defined (FAN_WINDOW)

#if defined(FAN_3D)

#include <fan/utility.h>
#include <fan/graphics/gl_api.h>

#endif

#endif

export module fan.graphics.opengl3D.objects.model;

#if defined (FAN_WINDOW)

#if defined(FAN_3D)

import std;

import fan.types;
import fan.types.vector;
import fan.types.matrix;
import fan.time;
import fan.print.error;
import fan.print;
import fan.graphics.common_context;
import fan.graphics.image_load;
import fan.graphics.fms;
#if defined(FAN_GUI)
  import fan.graphics.gui.base;
#endif
import fan.graphics.opengl.core;
import fan.graphics.loco;

#include <fan/graphics/opengl/init.h>

using namespace fan::opengl;

export namespace fan::model {
  GLuint upload_texture_array(const std::vector<fan::model::cpu_texture_t>& textures) {
    if (textures.empty()) { return 0; }
    fan::vec2ui layer_size = choose_texture_array_size(textures);
    GLuint tex_array = 0;
    glGenTextures(1, &tex_array);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex_array);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    glPixelStorei(GL_UNPACK_SKIP_IMAGES, 0);
    int levels = 1 + int(std::floor(std::log2(f32_t(std::max(layer_size.x, layer_size.y)))));
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, levels, GL_RGBA8, layer_size.x, layer_size.y, textures.size());
    for (int i = 0; i < int(textures.size()); ++i) {
      const auto& t = textures[i];
      std::vector<std::uint8_t> resized;
      const std::uint8_t* pixels = t.data.get();
      if (t.size.x != layer_size.x || t.size.y != layer_size.y || t.channels != 4) {
        resized = resize_rgba_nearest(t, layer_size);
        pixels = resized.data();
      }
      glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, layer_size.x, layer_size.y, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    }
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
    return tex_array;
  }
}

export namespace fan {
  namespace graphics {
    std::vector<fan::model::mesh_t> load_meshes(const fan::model::fms_t::properties_t& p, std::source_location callers_path = std::source_location::current()) {
      fan::model::fms_t fms(p, callers_path);
      return std::move(fms.meshes);
    }

    struct model_t : fan::model::fms_t{
      struct properties_t : fan::model::fms_t::properties_t {
        fan::graphics::camera_t camera = fan::graphics::get_perspective_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_perspective_render_view().viewport;
      };

      model_t(const properties_t& p, std::source_location callers_path = std::source_location::current()) : fms_t(p, callers_path) {
        camera_nr = p.camera;
        viewport_nr = p.viewport;
        mesh_images.resize(meshes.size());
        for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
          for (std::size_t ti = 0; ti < fan::model::texture_max; ++ti) {
            const auto& name = meshes[mi].texture_names[ti];
            mesh_images[mi][ti] = name.empty() ? fan::graphics::get_default_texture() : fan::graphics::image_load(name);
          }
        }
        std::string vs = fan::graphics::read_shader(fan::shader_paths::gl::model3d_vs);
        std::string fs = fan::graphics::read_shader(fan::shader_paths::gl::model3d_fs);
        m_shader = fan::graphics::shader_create();
        fan::graphics::shader_set_vertex(m_shader, fan::shader_paths::gl::model3d_vs, vs);
        fan::graphics::shader_set_fragment(m_shader, fan::shader_paths::gl::model3d_fs, fs);
        fan::graphics::shader_compile(m_shader);
        for (int i = 0; i < meshes.size(); ++i) {
          setup_mesh_buffers(meshes[i], i);
          meshes[i].vertices.clear();
          meshes[i].indices.clear();
        }
      }

      void setup_mesh_buffers(fan::model::mesh_t& mesh, int mesh_index) {
        if (gl_datas.size() <= mesh_index) {
          gl_datas.resize(mesh_index + 1);
        }
        gl_t& gl = gl_datas[mesh_index];
        gl.vao.open(fan::graphics::get_gl_context());
        gl.vbo.open(fan::graphics::get_gl_context(), GL_ARRAY_BUFFER);

        fan_opengl_call(glGenBuffers(1, &gl.ebo));

        gl.vao.bind(fan::graphics::get_gl_context());

        gl.vbo.bind(fan::graphics::get_gl_context());
        fan_opengl_call(glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(fan::model::vertex_t), &mesh.vertices[0], GL_STATIC_DRAW));

        fan_opengl_call(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl.ebo));
        fan_opengl_call(glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), &mesh.indices[0], GL_STATIC_DRAW));

        fan::opengl::context_t::shader_t shader = gloco()->shader_get(m_shader).gl;

        int location = (fan::graphics::get_gl_context().opengl.major == 2 && fan::graphics::get_gl_context().opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_position")) : 0;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(fan::model::vertex_t), (GLvoid*)offsetof(fan::model::vertex_t, position)));

        location = (fan::graphics::get_gl_context().opengl.major == 2 && fan::graphics::get_gl_context().opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_normal")) : 1;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(fan::model::vertex_t), (GLvoid*)offsetof(fan::model::vertex_t, normal)));

        location = (fan::graphics::get_gl_context().opengl.major == 2 && fan::graphics::get_gl_context().opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_uv")) : 2;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 2, GL_FLOAT, GL_FALSE, sizeof(fan::model::vertex_t), (GLvoid*)offsetof(fan::model::vertex_t, uv)));

        location = (fan::graphics::get_gl_context().opengl.major == 2 && fan::graphics::get_gl_context().opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_bone_ids")) : 3;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribIPointer(location, 4, GL_INT, sizeof(fan::model::vertex_t), (void*)offsetof(fan::model::vertex_t, bone_ids)));

        location = (fan::graphics::get_gl_context().opengl.major == 2 && fan::graphics::get_gl_context().opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_bone_weights")) : 4;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, sizeof(fan::model::vertex_t), (GLvoid*)offsetof(fan::model::vertex_t, bone_weights)));

        location = (fan::graphics::get_gl_context().opengl.major == 2 && fan::graphics::get_gl_context().opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_tangent")) : 5;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(fan::model::vertex_t), (GLvoid*)offsetof(fan::model::vertex_t, tangent)));

        location = (fan::graphics::get_gl_context().opengl.major == 2 && fan::graphics::get_gl_context().opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_bitangent")) : 6;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(fan::model::vertex_t), (GLvoid*)offsetof(fan::model::vertex_t, bitangent)));

        location = (fan::graphics::get_gl_context().opengl.major == 2 && fan::graphics::get_gl_context().opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_color")) : 7;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, sizeof(fan::model::vertex_t), (GLvoid*)offsetof(fan::model::vertex_t, color)));

        fan_opengl_call(glBindVertexArray(0));
      }
      void upload_modified_vertices() {
        if (p.use_cpu == 1) {
          fan::throw_error("implement calculated_meshes here");
        }
      }
      fan::graphics::image_t get_tex_id(std::size_t mesh_index, std::size_t tex_index) const {
        auto image = mesh_images[mesh_index][tex_index];
        return image.iic() ? fan::graphics::get_default_texture() : image;
      }
      void draw(const fan::mat4& model_transform = fan::mat4(1), const std::vector<fan::mat4>& bone_transforms = {}) {
        auto viewport = fan::graphics::viewport_get(viewport_nr);
        fan::graphics::viewport_set(viewport.position, viewport.size);
        fan::graphics::get_gl_context().shader_set_value(m_shader, "model", m_transform * user_transform);
        fan::graphics::get_gl_context().shader_set_value(m_shader, "use_cpu", p.use_cpu);
        fan::graphics::get_gl_context().shader_set_camera(m_shader, camera_nr);
        fan::graphics::get_gl_context().shader_set_value(m_shader, "light_position", light_position);
        fan::graphics::get_gl_context().shader_set_value(m_shader, "light_color", light_color);
        fan::graphics::get_gl_context().shader_set_value(m_shader, "light_intensity", light_intensity);
        fan::graphics::get_gl_context().set_depth_test(true);
        fan_opengl_call(glDisable(GL_BLEND));
        for (int mesh_index = 0; mesh_index < meshes.size(); ++mesh_index) {
          fan::opengl::context_t::shader_t shader = gloco()->shader_get(m_shader).gl;
          {
            auto location = fan_opengl_call(
              glGetUniformLocation(
              shader.id, 
              "material_colors"
            ));
            fan_opengl_call(glUniform4fv(
              location, 
              std::size(material_data_vector[mesh_index].color),
              &material_data_vector[mesh_index].color[0][0]
            ));
          }
          gl_datas[mesh_index].vao.bind(fan::graphics::get_gl_context());
          fan::vec3 camera_position = fan::graphics::camera_get_position(camera_nr);
          fan::graphics::get_gl_context().shader_set_value(m_shader, "view_p", camera_position);
          { // texture binding
            int tex_index = 0;
            
            for (auto& tex : meshes[mesh_index].texture_names) {
              std::ostringstream oss;
              oss << "_t" << std::setw(2) << std::setfill('0') << (int)tex_index;

              fan_opengl_call(glActiveTexture(GL_TEXTURE0 + tex_index));
              fan::graphics::get_gl_context().shader_set_value(m_shader, oss.str(), tex_index);
              fan::graphics::image_bind(mesh_images[mesh_index][tex_index]);

              ++tex_index;
            }
          }
          fan::graphics::get_gl_context().shader_set_value(m_shader, "bone_count", (int)std::min((std::size_t)200, bone_transforms.size()));
          GLint location = fan_opengl_call(glGetUniformLocation(shader.id, "bone_transforms"));
          fan_opengl_call(glUniformMatrix4fv(
            location,
            std::min((std::size_t)200, bone_transforms.size()),
            GL_FALSE,
            (f32_t*)bone_transforms.data()
          ));
    
          fan_opengl_call(
            glDrawElements(
              GL_TRIANGLES, 
              meshes[mesh_index].indices_len, 
              GL_UNSIGNED_INT,
              0
          ));
        }
      }
#if defined(FAN_GUI)
      void draw_cached_images() {
        using namespace fan::graphics;
        gui::begin("test");
        auto& style = gui::get_style();
        f32_t cursor_pos_x = 64 + style.ItemSpacing.x;

        for (auto& mesh : mesh_images) {
          for (auto image : mesh) {
            if (image.iic()) {
              continue;
            }

            fan::vec2 image_size(64, 64);
            fan::graphics::gui::image(image, image_size);

            if (cursor_pos_x + image_size.x > gui::get_content_region_avail().x) {
              gui::new_line();
              cursor_pos_x = image_size.x + style.ItemSpacing.x;
            }
            else {
              gui::same_line();
              cursor_pos_x += image_size.x + style.ItemSpacing.x;
            }
          }
        }
        gui::end();
      }
#endif

      //temp
      // user transform
      fan::mat4 user_transform{1};
      // 
      // user transform
      fan::vec3 light_position{3.46f, 1.94f, 6.22f};
      fan::vec3 light_color{.8f,.8f,.8f};
      f32_t light_intensity{1.f};
      struct gl_t {
        fan::opengl::core::vao_t vao;
        fan::opengl::core::gpu_buffer_t vbo;
        GLuint ebo;
      };
      std::vector<gl_t> gl_datas;
      fan::graphics::shader_t m_shader;
      GLuint envMapTexture;
      fan::graphics::camera_t camera_nr;
      fan::graphics::viewport_t viewport_nr;
      std::vector<std::array<fan::graphics::image_t, fan::model::texture_max>> mesh_images;
    };
  }
}
#endif

#endif