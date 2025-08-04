module;

#include <iomanip>

#include <fan/graphics/opengl/init.h>
#include <fan/imgui/imgui.h>

export module fan.graphics.opengl3D.objects.model;

export import fan.graphics.opengl3D.objects.fms;
export import fan.graphics.gui;

namespace fan_3d {
  namespace model {
    inline static std::unordered_map<std::string, fan::graphics::image_t> cached_images;
  }
}

export namespace fan {
  namespace graphics {
    using namespace opengl;
    struct model_t : fan_3d::model::fms_t{
      struct properties_t : fan_3d::model::fms_t::properties_t {
        loco_t::camera_t camera = gloco->perspective_render_view.camera;
        loco_t::viewport_t viewport = gloco->perspective_render_view.viewport;
      };
      model_t(const properties_t& p) : fms_t(p) {
        camera_nr = p.camera;
        viewport_nr = p.viewport;
        std::string vs = loco_t::read_shader("shaders/opengl/3D/objects/model.vs");
        std::string fs = loco_t::read_shader("shaders/opengl/3D/objects/model.fs");
        m_shader = gloco->shader_create();
        gloco->shader_set_vertex(m_shader, vs);
        gloco->shader_set_fragment(m_shader, fs);
        gloco->shader_compile(m_shader);
        
        // load textures
        for (fan_3d::model::mesh_t& mesh : meshes) {
          for (const std::string& name : mesh.texture_names) {
            if (name.empty()) {
              continue;
            }
            auto found = fan_3d::model::cached_images.find(name);
            if (found != fan_3d::model::cached_images.end()) { // check if texture has already been loaded for this cahce
             continue;
            }
            fan::image::info_t ii;
            auto& td = fan_3d::model::cached_texture_data[name];
            ii.data = td.data.data();
            ii.size = td.size;
            ii.channels = td.channels;
            fan::graphics::image_load_properties_t ilp;
            // other implementations i saw, only used these channels
            constexpr uint32_t gl_formats[] = {
              0,                      // index 0 unused
              fan::graphics::image_format::r8_unorm,    // index 1 for 1 channel
              0,                      // index 2 unused
              fan::graphics::image_format::rg8_unorm,    // index 3 for 3 channels
              fan::graphics::image_format::rgba_unorm    // index 4 for 4 channels
            };
            if (ii.channels < std::size(gl_formats) && gl_formats[ii.channels]) {
              ilp.format = ilp.internal_format = gl_formats[ii.channels];
              fan_3d::model::cached_images[name] = gloco->image_load(ii, ilp); // insert new texture, since old doesnt exist
            }
            else if (ii.data == nullptr) {
              continue;
            }
            else {
              fan::print("unimplemented channel", ii.channels);
              fan_3d::model::cached_images[name] = gloco->default_texture; // insert default (missing) texture, since old doesnt exist
            }
          }
          setup_mesh_buffers(mesh);
          mesh.vertices.clear();
          mesh.indices.clear();
        }
      }
      void setup_mesh_buffers(fan_3d::model::mesh_t& mesh) {
        mesh.vao.open(gloco->context.gl);
        mesh.vbo.open(gloco->context.gl, GL_ARRAY_BUFFER);
        fan_opengl_call(glGenBuffers(1, &mesh.ebo));

        mesh.vao.bind(gloco->context.gl);

        mesh.vbo.bind(gloco->context.gl);
        fan_opengl_call(glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(fan_3d::model::vertex_t), &mesh.vertices[0], GL_STATIC_DRAW));

        fan_opengl_call(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo));
        fan_opengl_call(glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), &mesh.indices[0], GL_STATIC_DRAW));

        fan::opengl::context_t::shader_t shader = gloco->shader_get(m_shader).gl;

        int location = (gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_position")) : 0;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(fan_3d::model::vertex_t), (GLvoid*)offsetof(fan_3d::model::vertex_t, position)));

        location = (gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_normal")) : 1;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(fan_3d::model::vertex_t), (GLvoid*)offsetof(fan_3d::model::vertex_t, normal)));

        location = (gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_uv")) : 2;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 2, GL_FLOAT, GL_FALSE, sizeof(fan_3d::model::vertex_t), (GLvoid*)offsetof(fan_3d::model::vertex_t, uv)));

        location = (gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_bone_ids")) : 3;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribIPointer(location, 4, GL_INT, sizeof(fan_3d::model::vertex_t), (void*)offsetof(fan_3d::model::vertex_t, bone_ids)));

        location = (gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_bone_weights")) : 4;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, sizeof(fan_3d::model::vertex_t), (GLvoid*)offsetof(fan_3d::model::vertex_t, bone_weights)));

        location = (gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_tangent")) : 5;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(fan_3d::model::vertex_t), (GLvoid*)offsetof(fan_3d::model::vertex_t, tangent)));

        location = (gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_bitangent")) : 6;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, sizeof(fan_3d::model::vertex_t), (GLvoid*)offsetof(fan_3d::model::vertex_t, bitangent)));

        location = (gloco->context.gl.opengl.major == 2 && gloco->context.gl.opengl.minor == 1) ?
          fan_opengl_call(glGetAttribLocation(shader.id, "in_color")) : 7;
        fan_opengl_call(glEnableVertexAttribArray(location));
        fan_opengl_call(glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE, sizeof(fan_3d::model::vertex_t), (GLvoid*)offsetof(fan_3d::model::vertex_t, color)));

        fan_opengl_call(glBindVertexArray(0));
      }
      void upload_modified_vertices() {
        if (p.use_cpu == 1) {
          fan::throw_error("implement calculated_meshes here");
        }
      }
      void draw(const fan::mat4& model_transform = fan::mat4(1), const std::vector<fan::mat4>& bone_transforms = {}) {
        auto viewport = gloco->viewport_get(viewport_nr);
        gloco->viewport_set(viewport.viewport_position, viewport.viewport_size);
        gloco->shader_set_value(m_shader, "model", m_transform * user_transform);
        gloco->shader_set_value(m_shader, "use_cpu", p.use_cpu);
        gloco->shader_set_camera(m_shader, camera_nr);
        gloco->shader_set_value(m_shader, "light_position", light_position);
        gloco->shader_set_value(m_shader, "light_color", light_color);
        gloco->shader_set_value(m_shader, "light_intensity", light_intensity);
        gloco->context.gl.set_depth_test(true);
        auto& context = gloco->get_context();
        fan_opengl_call(glDisable(GL_BLEND));
        for (int mesh_index = 0; mesh_index < meshes.size(); ++mesh_index) {
          fan::opengl::context_t::shader_t shader = gloco->shader_get(m_shader).gl;
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
          meshes[mesh_index].vao.bind(context.gl);
          fan::vec3 camera_position = gloco->camera_get_position(camera_nr);
          gloco->shader_set_value(m_shader, "view_p", camera_position);
          { // texture binding
            int tex_index = 0;
            
            for (auto& tex : meshes[mesh_index].texture_names) {
              std::ostringstream oss;
              oss << "_t" << std::setw(2) << std::setfill('0') << (int)tex_index;
              if (tex.empty()) {
                fan_opengl_call(glActiveTexture(GL_TEXTURE0 + tex_index));
                gloco->shader_set_value(m_shader, oss.str(), tex_index);
                gloco->image_bind(gloco->default_texture);
                ++tex_index;
                continue;
              }

              //tex.second.texture_datas
              fan_opengl_call(glActiveTexture(GL_TEXTURE0 + tex_index));
              gloco->shader_set_value(m_shader, oss.str(), tex_index);
              if (fan_3d::model::cached_images[tex].iic()) {
                fan_3d::model::cached_images[tex] = gloco->default_texture;
              }
              gloco->image_bind(fan_3d::model::cached_images[tex]);
              ++tex_index;
            }
          }
          gloco->shader_set_value(m_shader, "bone_count", (int)std::min((std::size_t)200, bone_transforms.size()));
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
#if defined(fan_gui)
      void draw_cached_images() {
        ImGui::Begin("test");
        float cursor_pos_x = 64 + ImGui::GetStyle().ItemSpacing.x;

        for (auto& i : fan_3d::model::cached_images) {
          ImVec2 imageSize(64, 64);
          fan::graphics::gui::image(i.second, imageSize);

          if (cursor_pos_x + imageSize.x > ImGui::GetContentRegionAvail().x) {
            ImGui::NewLine();
            cursor_pos_x = imageSize.x + ImGui::GetStyle().ItemSpacing.x;
          }
          else {
            ImGui::SameLine();
            cursor_pos_x += imageSize.x + ImGui::GetStyle().ItemSpacing.x;
          }
        }
        ImGui::End();
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
      loco_t::shader_t m_shader;
      GLuint envMapTexture;
      loco_t::camera_t camera_nr;
      loco_t::viewport_t viewport_nr;
    };
  }
}