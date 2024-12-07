#pragma once

#define fms_use_opengl
#include "fms.h"

namespace fan_3d {
  namespace model {
    inline static std::unordered_map<std::string, loco_t::image_t> cached_images;
  }
}

namespace fan {
  namespace graphics {
    using namespace opengl;
    struct model_t : fan_3d::model::fms_t{
      struct properties_t : fan_3d::model::fms_t::properties_t {
        loco_t::camera_t camera = gloco->perspective_camera.camera;
        loco_t::viewport_t viewport = gloco->perspective_camera.viewport;
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
            fan::image::image_info_t ii;
            auto& td = fan_3d::model::cached_texture_data[name];
            ii.data = td.data.data();
            ii.size = td.size;
            ii.channels = td.channels;
            fan::opengl::context_t::image_load_properties_t ilp;
            // other implementations i saw, only used these channels
            constexpr uint32_t gl_formats[] = {
              0,                      // index 0 unused
              fan::opengl::GL_RED,    // index 1 for 1 channel
              0,                      // index 2 unused
              fan::opengl::GL_RGB,    // index 3 for 3 channels
              fan::opengl::GL_RGBA    // index 4 for 4 channels
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
        mesh.vao.open(*gloco);
        mesh.vbo.open(*gloco, GL_ARRAY_BUFFER);
        gloco->opengl.glGenBuffers(1, &mesh.ebo);

        mesh.vao.bind(*gloco);

        mesh.vbo.bind(*gloco);
        gloco->opengl.glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(fan_3d::model::vertex_t), &mesh.vertices[0], GL_STATIC_DRAW);

        gloco->opengl.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
        gloco->opengl.glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), &mesh.indices[0], GL_STATIC_DRAW);

        gloco->opengl.glEnableVertexAttribArray(0);
        gloco->opengl.glVertexAttribPointer(0, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, position));
        gloco->opengl.glEnableVertexAttribArray(1);
        gloco->opengl.glVertexAttribPointer(1, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, normal));
        gloco->opengl.glEnableVertexAttribArray(2);
        gloco->opengl.glVertexAttribPointer(2, 2, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, uv));
        gloco->opengl.glEnableVertexAttribArray(3);
        // glVertexAttribIPointer opengl 3.0
        gloco->opengl.glVertexAttribIPointer(3, 4, fan::opengl::GL_INT, sizeof(fan_3d::model::vertex_t), (void*)offsetof(fan_3d::model::vertex_t, bone_ids));
        gloco->opengl.glEnableVertexAttribArray(4);
        gloco->opengl.glVertexAttribPointer(4, 4, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, bone_weights));
        gloco->opengl.glEnableVertexAttribArray(5);
        gloco->opengl.glVertexAttribPointer(5, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, tangent));
        gloco->opengl.glEnableVertexAttribArray(6);
        gloco->opengl.glVertexAttribPointer(6, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, bitangent));
        gloco->opengl.glEnableVertexAttribArray(7);
        gloco->opengl.glVertexAttribPointer(7, 4, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, color));
        gloco->opengl.glBindVertexArray(0);
      }
      void upload_modified_vertices() {
        if (p.use_cpu == 1) {
          fan::throw_error("implement calculated_meshes here");
        }
      }
      void draw(const fan::mat4& model_transform = fan::mat4(1), const std::vector<fan::mat4>& bone_transforms = {}) {
        auto viewport = gloco->viewport_get(viewport_nr);
        gloco->viewport_set(viewport.viewport_position, viewport.viewport_size, gloco->window.get_size());
        gloco->shader_set_value(m_shader, "model",
          m_transform * fan::translation_matrix(user_position) *
          fan::rotation_quat_matrix(fan::quat::from_angles(user_rotation)) *
          fan::scaling_matrix(user_scale)
        );
        gloco->shader_set_value(m_shader, "use_cpu", p.use_cpu);
        gloco->shader_set_camera(m_shader, camera_nr);
        gloco->shader_set_value(m_shader, "light_position", light_position);
        gloco->shader_set_value(m_shader, "light_color", light_color);
        gloco->shader_set_value(m_shader, "light_intensity", light_intensity);
        gloco->set_depth_test(true);
        auto& context = gloco->get_context();
        context.opengl.glDisable(fan::opengl::GL_BLEND);
        for (int mesh_index = 0; mesh_index < meshes.size(); ++mesh_index) {
          fan::opengl::context_t::shader_t& shader = gloco->shader_get(m_shader);
          {
            auto location = gloco->opengl.call(
              gloco->opengl.glGetUniformLocation, 
              shader.id, 
              "material_colors"
            );
            gloco->opengl.glUniform4fv(
              location, 
              std::size(material_data_vector[mesh_index].color),
              &material_data_vector[mesh_index].color[0][0]
            );
          }
          meshes[mesh_index].vao.bind(context);
          fan::vec3 camera_position = gloco->camera_get_position(camera_nr);
          gloco->shader_set_value(m_shader, "view_p", camera_position);
          { // texture binding
            uint8_t tex_index = 0;
            
            for (auto& tex : meshes[mesh_index].texture_names) {
              if (tex.empty()) {
                continue;
              }
              std::ostringstream oss;
              oss << "_t" << std::setw(2) << std::setfill('0') << (int)tex_index;

              //tex.second.texture_datas
              gloco->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + tex_index);

              gloco->shader_set_value(m_shader, oss.str(), tex_index);
              if (fan_3d::model::cached_images[tex].iic()) {
                fan_3d::model::cached_images[tex] = gloco->default_texture;
              }
              gloco->image_bind(fan_3d::model::cached_images[tex]);
              ++tex_index;
            }
          }
          gloco->shader_set_value(m_shader, "bone_count", (int)std::min((std::size_t)200, bone_transforms.size()));
          fan::opengl::GLint location = gloco->opengl.call(gloco->opengl.glGetUniformLocation, shader.id, "bone_transforms");
          gloco->opengl.glUniformMatrix4fv(
            location,
            std::min((std::size_t)200, bone_transforms.size()),
            GL_FALSE,
            (f32_t*)bone_transforms.data()
          );
    
          gloco->opengl.glDrawElements(
              GL_TRIANGLES, 
              meshes[mesh_index].indices_len, 
              GL_UNSIGNED_INT,
              0
          );
        }
      }
      void draw_cached_images() {
        ImGui::Begin("test");
        float cursor_pos_x = 64 + ImGui::GetStyle().ItemSpacing.x;

        for (auto& i : fan_3d::model::cached_images) {
          ImVec2 imageSize(64, 64);
          ImGui::Image(i.second, imageSize);

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

      //temp
      // user transform
      fan::vec3 user_position = 0;
      fan::vec3 user_rotation = 0;
      fan::vec3 user_scale = 1;
      // 
      // user transform
      fan::vec3 light_position{3.46f, 1.94f, 6.22f};
      fan::vec3 light_color{.8f,.8f,.8f};
      f32_t light_intensity{1.f};
      loco_t::shader_t m_shader;
      fan::opengl::GLuint envMapTexture;
      loco_t::camera_t camera_nr;
      loco_t::viewport_t viewport_nr;
    };
  }
}