#pragma once

#include _FAN_PATH(graphics/opengl/gl_core.h)
#include _FAN_PATH(graphics/shared_graphics.h)

#include _FAN_PATH(graphics/opengl/3D/objects/skybox.h)

extern "C" {
  #define FAST_OBJ_IMPLEMENTATION
  #include _FAN_PATH(graphics/fast_obj/fast_obj.h)
}

namespace fan_3d {
  namespace opengl {

    struct model_t {

      struct textures_t {
        fan::opengl::image_t diffuse;
        fan::opengl::image_t depth;
      };

      struct loaded_model_t {
        fan::hector_t<fan::vec3> vertices;
        //fan::hector_t<uint32_t> indices;
        //fan::hector_t<fan::vec3> normals;
        fan::hector_t<fan::vec2> texture_coordinates;
        //fan::hector_t<fan::vec3> tangets;
        textures_t m_textures;
      };

      struct model_instance_t {
        fan::vec3 position;
        fan::vec3 size;
        f32_t angle;
        fan::vec3 rotation_point;
        fan::vec3 rotation_vector;
        uint32_t m_ebo;
        fan::opengl::core::glsl_buffer_t m_glsl_buffer;
        fan::hector_t<uint32_t> m_indices;
        uint32_t is_light;
        fan::vec3 light_position = 0;
        fan_3d::opengl::skybox::loaded_skybox_t skybox;
      };

      static loaded_model_t* load_model(fan::opengl::context_t* context, const std::string& path) {
        
        fastObjMesh* mesh = fast_obj_read(path.c_str());

        if (mesh == nullptr)
        {
          fan::throw_error("error loading model \"" + path + "\"");
        }

        loaded_model_t* model = new loaded_model_t;
        model->vertices.open();
        //model->indices.open();
        //model->normals.open();
        //model->texture_coordinates.open();
        //model->tangets.open();

        parse_model(model, mesh);
       // aiMaterial* material = scene->mMaterials[scene->mMeshes[0]->mMaterialIndex];
        model->m_textures = parse_model_texture(context, mesh);

        return model;
      }

      struct properties_t {
        fan::vec3 position;
        fan::vec3 size = 1;
        f32_t angle = 0;
        fan::vec3 rotation_point = 0;
        fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
        loaded_model_t* loaded_model;
        uint32_t is_light = false;
        fan_3d::opengl::skybox::loaded_skybox_t skybox;
        fan::vec3* camera_position;
      };
      struct instance_t {
        fan::vec3 position;
        fan::vec2 texture_coordinate;
        fan::vec3 normal;
        fan::vec3 tanget;
      };

      static constexpr uint32_t offset_position = offsetof(instance_t, position);
      static constexpr uint32_t offset_texture_coordinate = offsetof(instance_t, texture_coordinate);
      //static constexpr uint32_t offset_normal = offsetof(instance_t, normal);
      //static constexpr uint32_t offset_tanget = offsetof(instance_t, tanget);
      static constexpr uint32_t element_byte_size = offset_texture_coordinate + sizeof(fan::vec3);

      void open(fan::opengl::context_t* context) {
        m_shader.open(context);

        m_shader.set_vertex(
          context, 
          #include _FAN_PATH(graphics/glsl/opengl/3D/objects/model.vs)
        );

        m_shader.set_fragment(
          context, 
          #include _FAN_PATH(graphics/glsl/opengl/3D/objects/model.fs)
        );

        m_shader.compile(context);

        m_draw_node_reference = fan::uninitialized;

        m_model_instance.m_indices.open();
        m_model_instance.m_glsl_buffer.open(context);
        m_model_instance.m_glsl_buffer.init(context, m_shader.id, element_byte_size);
      }
      void close(fan::opengl::context_t* context) {

        m_shader.close(context);

        if (m_draw_node_reference == fan::uninitialized) {
          return;
        }

        context->disable_draw(m_draw_node_reference);
        m_draw_node_reference = fan::uninitialized;
        context->opengl.glDeleteBuffers(1, &m_model_instance.m_ebo);
        m_model_instance.m_indices.close();
      }

      void set(fan::opengl::context_t* context, properties_t properties) {
        m_model_instance.position = properties.position;
        m_model_instance.size = properties.size;
        m_model_instance.angle = properties.angle;
        m_model_instance.rotation_point = properties.rotation_point;
        m_model_instance.rotation_vector = properties.rotation_vector;
        m_model_instance.is_light = properties.is_light;
        m_model_instance.skybox = properties.skybox;
        camera_position = properties.camera_position;
        m_textures = properties.loaded_model->m_textures;

        instance_t instance;
        for (int i = 0; i < properties.loaded_model->vertices.size(); i++) {
          instance.position = properties.loaded_model->vertices[i];
          //instance.normal = properties.loaded_model->normals[i];
          instance.texture_coordinate = properties.loaded_model->texture_coordinates[i];
          //if (i < properties.loaded_model->tangets.size()) {
            //instance.tanget = properties.loaded_model->tangets[i];
         // }
          m_model_instance.m_glsl_buffer.push_ram_instance(context, &instance, element_byte_size);
        }

        m_model_instance.m_glsl_buffer.write_vram_all(context);
        this->draw(context);
      }

      fan::vec3 get_position(fan::opengl::context_t* context, uint32_t i) const;
      void set_position(fan::opengl::context_t* context, const fan::vec3& position) {
        m_model_instance.position = position;
      }

      void set_light_position(fan::opengl::context_t* context, const fan::vec3& position) {
        m_model_instance.light_position = position;
      }

      fan::vec3 get_size(fan::opengl::context_t* context, uint32_t i) const;
      void set_size(fan::opengl::context_t* context, uint32_t i, const fan::vec3& size);

      f32_t get_angle(fan::opengl::context_t* context, uint32_t i) const;
      void set_angle(fan::opengl::context_t* context, f32_t angle) {
        m_model_instance.angle = angle;
      }

      fan::vec3 get_rotation_vector(fan::opengl::context_t* context, uint32_t i) const;
      void set_rotation_vector(fan::opengl::context_t* context, uint32_t i, const fan::vec3& rotation_vector) {
        m_model_instance.rotation_vector = rotation_vector;
      }

      void enable_draw(fan::opengl::context_t* context) {
        m_draw_node_reference = context->enable_draw(this, [](fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
      }
      void disable_draw(fan::opengl::context_t* context);

      void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
        m_shader.bind_matrices(context, matrices);
      }

    protected:

      static textures_t parse_model_texture(fan::opengl::context_t* context, fastObjMesh* mesh) {

        textures_t textures;

        static auto path_maker = [](std::string p) -> std::string {

          auto found = p.find_last_of('\\');
          p = p.erase(0, found + 1);
          found = p.find('.');
          p.replace(found, p.size(), ".webp");

          return p;
        };

        #if fan_debug >= fan_debug_low
          /*if (mesh->material_count > 1) {
            fan::throw_error("multi texture models not supported");
          }*/
        #endif

        if (mesh->materials->map_Kd.path == nullptr) {
          fan::throw_error("no diffuse map");
          textures.diffuse.load(context, "models/missing_texture.webp");
        }
        else {
          std::string diffuse_path = mesh->materials->map_Kd.path;

          textures.diffuse.load(context, "models/" + path_maker(diffuse_path));
        }
       /* if (mesh->materials->map_bump.path == nullptr) {
          textures.depth.load(context, "models/missing_depth.webp");
        }
        else {
          std::string depth_path = mesh->materials->map_bump.path;
          textures.depth.load(context, "models/" + path_maker(depth_path));
        }*/
        return textures;
      }

      static void parse_model(loaded_model_t* lm, fastObjMesh* m) {

        lm->vertices.open();
        lm->texture_coordinates.open();

        for (unsigned int ii = 0; ii < m->group_count; ii++)
        {
          const fastObjGroup& grp = m->groups[ii];

          // reference https://github.com/thisistherk/fast_obj/blob/master/test/test.cpp

          uint32_t idx = 0;
          for (uint32_t i = 0; i < grp.face_count; i++) {

            uint32_t fv = m->face_vertices[grp.face_offset + i];

            for (uint32_t j = 0; j < fv; j++) {
              fastObjIndex mi = m->indices[grp.index_offset + idx];

              lm->vertices.push_back(fan::vec3(
                m->positions[3 * mi.p + 0],
                m->positions[3 * mi.p + 1],
                m->positions[3 * mi.p + 2]
              ));

              lm->texture_coordinates.push_back(fan::vec2(
                m->texcoords[2 * mi.t + 0],
                m->texcoords[2 * mi.t + 1]
              ));

              idx++;
            }
          }
        }
      }

      void draw(fan::opengl::context_t* context) {
        context->set_depth_test(true);

        m_shader.use(context);

        m_model_instance.m_glsl_buffer.m_vao.bind(context);

        fan::mat4 m[2];
        std::memset(m, 0, sizeof(m));
        m[0][0][0] = m_model_instance.position.x;
        m[0][0][1] = m_model_instance.position.y;
        m[0][0][2] = m_model_instance.position.z;
        m[0][0][3] = m_model_instance.size.x;

        m[0][1][0] = m_model_instance.size.y;
        m[0][1][1] = m_model_instance.size.z;
        m[0][1][2] = m_model_instance.angle;
        m[0][1][3] = m_model_instance.rotation_point.x;

        m[0][2][0] = m_model_instance.rotation_point.y;
        m[0][2][1] = m_model_instance.rotation_point.z;
        m[0][2][2] = m_model_instance.rotation_vector.x;
        m[0][2][3] = m_model_instance.rotation_vector.y;

        m[0][3][0] = m_model_instance.rotation_vector.z;
        m[0][3][1] = m_model_instance.is_light;
        // camera position
        m[0][3][2] = camera_position->x;
        m[0][3][3] = camera_position->y;
        m[1][0][0] = camera_position->z;

        /*#if fan_debug >= fan_debug_low
          if (m_textures.diffuse.texture == 0) {
            fan::throw_error("invalid diffuse texture");
          }
          if (m_textures.depth.texture == 0) {
            fan::throw_error("invalid depth texture");
          }
        #endif
        */
        m_shader.set_int(context, "texture_diffuse", 0);
        context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
        context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_textures.diffuse.texture);
        /*
        m_shader.set_int(context, "texture_depth", 1);
        context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + 1);
        context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_textures.depth.texture);*/

       /* m_shader.set_int(context, "skybox", 2);
        context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + 2);
        context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_model_instance.skybox.texture);*/

        m_shader.set_mat4(context, "model_input", &m[0][0][0], std::size(m));
       // m_shader.set_float(context, "s", s);
        //m_shader.set_vec3(context, "p", m_model_instance.light_position);

        context->opengl.glDrawArrays(fan::opengl::GL_TRIANGLES, 0, m_model_instance.m_glsl_buffer.m_buffer.size() / element_byte_size);
      }

      model_instance_t m_model_instance;

      fan::opengl::shader_t m_shader;
      uint32_t m_draw_node_reference;
      textures_t m_textures;

      fan::vec3* camera_position;

      public:
        f32_t s = 2;
    };

  }
}