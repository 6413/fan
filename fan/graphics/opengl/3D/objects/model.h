#pragma once

#include "fms.h"

namespace fan {
  namespace graphics {
    using namespace opengl;
    struct model_t {


      struct use_flag_e {
        enum {
          model_cpu,
          model_gpu
        };
      };

      struct properties_t : fan_3d::model::fms_model_info_t {
        fan::mat4 model{ 1 };
        uint8_t use_flag = use_flag_e::model_gpu;
      };

      // cpp badness.. compiler thinks this is not compiletime
      std::vector<std::array<std::string, 2>> vertex_shader_paths = {
        {
          "shaders/opengl/3D/objects/model_cpu.vs",
          "shaders/opengl/3D/objects/model_cpu.fs"
        },
        {
          "shaders/opengl/3D/objects/model_gpu.vs",
          "shaders/opengl/3D/objects/model_cpu.fs"
        },
      };

      model_t(const properties_t& p) : fms(p) {
        m_shader = gloco->shader_create();
        std::string vs = loco_t::read_shader(vertex_shader_paths[p.use_flag][0]);
        std::string fs = loco_t::read_shader(vertex_shader_paths[p.use_flag][1]);
        gloco->shader_set_vertex(m_shader, vs);
        gloco->shader_set_fragment(m_shader, fs);

        gloco->shader_compile(m_shader);
      }

      void upload_modified_vertices(uint32_t i) {
        fms.meshes[i].VAO.bind(gloco->get_context());

        fms.meshes[i].VBO.write_buffer(
          gloco->get_context(),
          &fms.calculated_meshes[i].vertices[0],
          sizeof(fan_3d::model::vertex_t) * fms.calculated_meshes[i].vertices.size()
        );
      }


      void draw() {
        gloco->shader_use(m_shader);

        gloco->get_context().shader_set_camera(m_shader, &gloco->camera_get(gloco->perspective_camera.camera));

        //gloco->shader_set_value(m_shader, "projection", projection);
        //gloco->shader_set_value(m_shader, "view", view);

        gloco->get_context().set_depth_test(true);

        gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE3);
        gloco->get_context().opengl.glBindTexture(fan::opengl::GL_TEXTURE_CUBE_MAP, envMapTexture);
        gloco->shader_set_value(m_shader, "envMap", 3);
        gloco->shader_set_value(m_shader, "m", m);

        auto& context = gloco->get_context();
        context.opengl.glDisable(fan::opengl::GL_BLEND);
        //context.opengl.call(context.opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
        for (int mesh_index = 0; mesh_index < fms.meshes.size(); ++mesh_index) {
          
          // only for gpu vs
          fan::mat4 model(1);
          gloco->shader_set_value(m_shader, "model", model);

          fms.meshes[mesh_index].VAO.bind(context);
          fan::vec3 camera_position = gloco->camera_get_position(gloco->perspective_camera.camera);
          gloco->shader_set_value(m_shader, "view_p", camera_position);
          { // texture binding
            uint8_t tex_index = 0;
            uint8_t valid_tex_index = 0;
            
            //for (auto& tex : fms.parsed_model.model_data.mesh_data[mesh_index].names) { // i think think this doesnt make sense
            //  std::ostringstream oss;
            //  oss << "_t" << std::setw(2) << std::setfill('0') << (int)tex_index;

            //  //tex.second.texture_datas
            //  gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + tex_index);
            //  if (tex.empty()) {
            //    continue;
            //  }

            //  gloco->shader_set_value(m_shader, oss.str(), tex_index);
            //  gloco->image_bind(cached_images[tex]);

            //  if (tex_index == aiTextureType_NORMALS) { // whats this?
            //    gloco->shader_set_value(m_shader, "has_normal", 1);
            //  }
            //  else {
            //    gloco->shader_set_value(m_shader, "has_normal", 0);
            //  }
            //  ++tex_index;
            //}
          }
          gloco->opengl.glDrawElements(GL_TRIANGLES, fms.meshes[mesh_index].indices.size(), GL_UNSIGNED_INT, 0);
        }
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
      std::unordered_map<std::string, loco_t::image_t> cached_images;

      static constexpr uint8_t axis_count = 3;
      loco_t::shape_t joint_controls[axis_count];
      fan::opengl::GLuint envMapTexture;
      fan::mat4 m{ 1 };
    };
  }
}