#pragma once

#include "fms.h"

namespace fan {
  namespace graphics {
    using namespace opengl;
    struct model_cpu_t {

      struct properties_t : fan_3d::model::fms_model_info_t {

      };
      model_cpu_t(const properties_t& p) : fms(p) {
        std::string vs = loco_t::read_shader("shaders/opengl/3D/objects/model_cpu.vs");
        std::string fs = loco_t::read_shader("shaders/opengl/3D/objects/model_cpu.fs");
        m_shader = gloco->shader_create();
        gloco->shader_set_vertex(m_shader, vs);
        gloco->shader_set_fragment(m_shader, fs);
        gloco->shader_compile(m_shader);

        // load textures
        for (fan_3d::model::mesh_t& mesh : fms.meshes) {
          for (const std::string& name : mesh.texture_names) {
            if (name.empty()) {
              continue;
            }
            auto found = cached_images.find(name);
            if (found != cached_images.end()) { // check if texture has already been loaded for this cahce
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
              cached_images[name] = gloco->image_load(ii, ilp); // insert new texture, since old doesnt exist
            } 
            else {
              fan::print("unimplemented channel", ii.channels);
              cached_images[name] = gloco->default_texture; // insert default (missing) texture, since old doesnt exist
            }
          }
          SetupMeshBuffers(mesh);
        }
      }

      // kinda illegal here
      void SetupMeshBuffers(fan_3d::model::mesh_t& mesh) {
        mesh.VAO.open(*gloco);
        mesh.VBO.open(*gloco, GL_ARRAY_BUFFER);
        gloco->opengl.glGenBuffers(1, &mesh.EBO);

        mesh.VAO.bind(*gloco);

        mesh.VBO.bind(*gloco);
        gloco->opengl.glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(fan_3d::model::vertex_t), &mesh.vertices[0], GL_STATIC_DRAW);

        gloco->opengl.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
        gloco->opengl.glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), &mesh.indices[0], GL_STATIC_DRAW);


        gloco->opengl.glEnableVertexAttribArray(0);
        gloco->opengl.glVertexAttribPointer(0, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, position));
        gloco->opengl.glEnableVertexAttribArray(1);
        gloco->opengl.glVertexAttribPointer(1, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, normal));
        gloco->opengl.glEnableVertexAttribArray(2);
        gloco->opengl.glVertexAttribPointer(2, 2, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, uv));
        gloco->opengl.glEnableVertexAttribArray(3);
        gloco->opengl.glVertexAttribPointer(3, 4, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, bone_ids));
        gloco->opengl.glEnableVertexAttribArray(4);
        gloco->opengl.glVertexAttribPointer(4, 4, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, bone_weights));
        gloco->opengl.glEnableVertexAttribArray(5);
        gloco->opengl.glVertexAttribPointer(5, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, tangent));
        gloco->opengl.glEnableVertexAttribArray(6);
        gloco->opengl.glVertexAttribPointer(6, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, bitangent));
        gloco->opengl.glBindVertexArray(0);
      }

      void upload_modified_vertices() {
        for (uint32_t i = 0; i < fms.meshes.size(); ++i) {
          fms.meshes[i].VAO.bind(gloco->get_context());
          fms.meshes[i].VBO.write_buffer(
            gloco->get_context(),
            &fms.calculated_meshes[i].vertices[0],
            sizeof(fan_3d::model::vertex_t) * fms.calculated_meshes[i].vertices.size()
          );
        }
      }


      void draw(const fan::mat4& model_transform = fan::mat4(1)) {
        gloco->shader_use(m_shader);

        gloco->shader_set_camera(m_shader, &gloco->camera_get(gloco->perspective_camera.camera));

        gloco->set_depth_test(true);

        auto& context = gloco->get_context();
        context.opengl.glDisable(fan::opengl::GL_BLEND);
        //context.opengl.call(context.opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
        for (int mesh_index = 0; mesh_index < fms.meshes.size(); ++mesh_index) {
          
          // only for gpu vs
          gloco->shader_set_value(m_shader, "model", model_transform);

          fms.meshes[mesh_index].VAO.bind(context);
          fan::vec3 camera_position = gloco->camera_get_position(gloco->perspective_camera.camera);
          gloco->shader_set_value(m_shader, "view_p", camera_position);
          { // texture binding
            uint8_t tex_index = 0;
            
            for (auto& tex : fms.meshes[mesh_index].texture_names) {
              if (tex.empty()) {
                continue;
              }
              std::ostringstream oss;
              oss << "_t" << std::setw(2) << std::setfill('0') << (int)tex_index;

              //tex.second.texture_datas
              gloco->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + tex_index);

              gloco->shader_set_value(m_shader, oss.str(), tex_index);
              gloco->image_bind(cached_images[tex]);
              ++tex_index;
            }
          }
          gloco->opengl.glDrawElements(GL_TRIANGLES, fms.meshes[mesh_index].indices.size(), GL_UNSIGNED_INT, 0);
        }
      }

      void draw_cached_images() {
        ImGui::Begin("test");
        float cursor_pos_x = 64 + ImGui::GetStyle().ItemSpacing.x;

        for (auto& i : cached_images) {
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


      fan_3d::model::fms_t fms;

      loco_t::shader_t m_shader;
      // should be stored globally among all models
      inline static std::unordered_map<std::string, loco_t::image_t> cached_images;

      fan::opengl::GLuint envMapTexture;
    };
  }
}