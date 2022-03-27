#pragma once

#include <fan/graphics/opengl/gl_core.h>
#include <fan/graphics/shared_graphics.h>

#include <fan/graphics/opengl/3D/objects/skybox.h>

extern "C" {
  #define FAST_OBJ_IMPLEMENTATION
  #include <fan/graphics/fast_obj/fast_obj.h>
}

namespace fan_3d {
  namespace opengl {

    struct model_t {

      struct textures_t {
        uint32_t diffuse;
        uint32_t depth;
      };

      struct loaded_model_t {
        fan::hector_t<fan::vec3> vertices;
        fan::hector_t<uint32_t> indices;
        fan::hector_t<fan::vec3> normals;
        fan::hector_t<fan::vec2> texture_coordinates;
        fan::hector_t<fan::vec3> tangets;
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
        fan::vec3 light_position;
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
        model->indices.open();
        model->normals.open();
        model->texture_coordinates.open();
        model->tangets.open();

        parse_model(model, mesh);
       // aiMaterial* material = scene->mMaterials[scene->mMeshes[0]->mMaterialIndex];
        model->m_textures = parse_model_texture(context, mesh);

        return model;
      }

      struct properties_t {
        fan::vec3 position;
        fan::vec3 size = 1;
        f32_t angle = 0;
        fan::vec3 rotation_point;
        fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
        loaded_model_t* loaded_model;
        uint32_t is_light = false;
        fan_3d::opengl::skybox::loaded_skybox_t skybox;
      };
      struct instance_t {
        fan::vec3 position;
        fan::vec2 texture_coordinate;
        fan::vec3 normal;
        fan::vec3 tanget;
      };

      static constexpr uint32_t offset_position = offsetof(instance_t, position);
      static constexpr uint32_t offset_texture_coordinate = offsetof(instance_t, texture_coordinate);
      static constexpr uint32_t offset_normal = offsetof(instance_t, normal);
      static constexpr uint32_t offset_tanget = offsetof(instance_t, tanget);
      static constexpr uint32_t element_byte_size = offset_tanget + sizeof(fan::vec3);

      void open(fan::opengl::context_t* context) {
        m_shader.open();

        m_shader.set_vertex(
        #include <fan/graphics/glsl/opengl/3D/objects/model.vs>
        );

        m_shader.set_fragment(
        #include <fan/graphics/glsl/opengl/3D/objects/model.fs>
        );

        m_shader.compile();

        m_draw_node_reference = fan::uninitialized;

        m_model_instance.m_indices.open();
        m_model_instance.m_glsl_buffer.open();
        m_model_instance.m_glsl_buffer.init(m_shader.id, element_byte_size);
      }
      void close(fan::opengl::context_t* context) {

        m_shader.close();

        if (m_draw_node_reference == fan::uninitialized) {
          return;
        }

        context->disable_draw(m_draw_node_reference);
        m_draw_node_reference = fan::uninitialized;
        glDeleteBuffers(1, &m_model_instance.m_ebo);
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
        m_textures = properties.loaded_model->m_textures;

        instance_t instance;
        for (int i = 0; i < properties.loaded_model->vertices.size(); i++) {
          instance.position = properties.loaded_model->vertices[i];
          instance.normal = properties.loaded_model->normals[i];
          instance.texture_coordinate = properties.loaded_model->texture_coordinates[i];
          instance.tanget = properties.loaded_model->tangets[i];
          m_model_instance.m_glsl_buffer.push_ram_instance(&instance, element_byte_size);
        }

        for (int i = 0; i < properties.loaded_model->indices.size(); i++) {
          m_model_instance.m_indices.push_back(properties.loaded_model->indices[i]);
        }

        glGenBuffers(1, &m_model_instance.m_ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_model_instance.m_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * m_model_instance.m_indices.size(), m_model_instance.m_indices.data(), GL_STATIC_DRAW);

        m_model_instance.m_glsl_buffer.write_vram_all();
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
      void set_rotation_vector(fan::opengl::context_t* context, uint32_t i, const fan::vec3& rotation_vector);

      void enable_draw(fan::opengl::context_t* context) {
        m_draw_node_reference = context->enable_draw(this, [](fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
      }
      void disable_draw(fan::opengl::context_t* context);

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
          if (mesh->material_count > 1) {
            fan::throw_error("multi texture models not supported");
          }
        #endif

        if (mesh->materials->map_Kd.path == nullptr) {
          textures.diffuse = fan::opengl::load_image(context, "models/missing_texture.webp")->texture;
        }
        else {
          std::string diffuse_path = mesh->materials->map_Kd.path;

          textures.diffuse = fan::opengl::load_image(context, "models/" + path_maker(diffuse_path))->texture;
        }
        if (mesh->materials->map_bump.path == nullptr) {
          textures.depth = fan::opengl::load_image(context, "models/missing_depth.webp")->texture;
        }
        else {
          std::string depth_path = mesh->materials->map_bump.path;
          textures.depth = fan::opengl::load_image(context, "models/" + path_maker(depth_path))->texture;
        }
        return textures;
      }

      static void parse_model(loaded_model_t* m, fastObjMesh* obj) {
        uint32_t i = 0;
        auto push_tangets = [&]{
           if (!(i % 3) && i) {
            fan::vec3 pos1 = m->vertices[i - 3];
            fan::vec3 pos2 = m->vertices[i - 2];
            fan::vec3 pos3 = m->vertices[i - 1];

            fan::vec2 uv1 = m->texture_coordinates[i - 3];
            fan::vec2 uv2 = m->texture_coordinates[i - 2];
            fan::vec2 uv3 = m->texture_coordinates[i - 1];

            fan::vec3 tangent1;

            fan::vec3 edge1 = pos2 - pos1;
            fan::vec3 edge2 = pos3 - pos1;
            fan::vec2 deltaUV1 = uv2 - uv1;
            fan::vec2 deltaUV2 = uv3 - uv1;

            
            f32_t f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

            tangent1.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
            tangent1.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
            tangent1.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);


            m->tangets.push_back(tangent1);
            m->tangets.push_back(tangent1);
            m->tangets.push_back(tangent1);
          }
        };

        for (; i < obj->index_count; i++) {

          push_tangets();

          m->vertices.push_back(
            fan::vec3(
              obj->positions[obj->indices[i].p * 3 + 0],
              obj->positions[obj->indices[i].p * 3 + 1],
              obj->positions[obj->indices[i].p * 3 + 2]
            )
          );
          m->normals.push_back(
            fan::vec3(
              obj->normals[obj->indices[i].n * 3 + 0],
              obj->normals[obj->indices[i].n * 3 + 1],
              obj->normals[obj->indices[i].n * 3 + 2]
            )
          );
          m->texture_coordinates.push_back(
            fan::vec2(
              obj->texcoords[obj->indices[i].t * 2 + 0],
              obj->texcoords[obj->indices[i].t * 2 + 1]
            )
          );
        }
        push_tangets();
      }

      void draw(fan::opengl::context_t* context) {
        context->set_depth_test(true);

        m_shader.use();

        const fan::vec2 viewport_size = context->viewport_size;

        fan::mat4 projection(1);
        projection = fan::math::perspective<fan::mat4>(fan::math::radians(90.0), (f32_t)context->viewport_size.x / (f32_t)context->viewport_size.y, 0.1f, 1000.0f);

        fan::mat4 view(1);
        view = context->camera.get_view_matrix();

        m_shader.use();
        m_shader.set_projection(projection);
        m_shader.set_view(view);

        m_model_instance.m_glsl_buffer.m_vao.bind();

        fan::mat4 m[2];
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
        m[0][3][2] = context->camera.get_position().x;
        m[0][3][3] = context->camera.get_position().y;

        m[1][0][0] = context->camera.get_position().z;

        #if fan_debug >= fan_debug_low
          if (m_textures.diffuse == 0) {
            fan::throw_error("invalid diffuse texture");
          }
          if (m_textures.depth == 0) {
            fan::throw_error("invalid depth texture");
          }
        #endif

        m_shader.set_int("texture_diffuse", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_textures.diffuse);

        m_shader.set_int("texture_depth", 1);
        glActiveTexture(GL_TEXTURE0 + 1);
        glBindTexture(GL_TEXTURE_2D, m_textures.depth);

        m_shader.set_int("skybox", 2);
        glActiveTexture(GL_TEXTURE0 + 2);
        glBindTexture(GL_TEXTURE_2D, m_model_instance.skybox.texture);

        m_shader.set_mat4("model_input", &m[0][0][0], std::size(m));
        m_shader.set_float("s", s);
        m_shader.set_vec3("p", m_model_instance.light_position);

        glDrawArrays(GL_TRIANGLES, 0, m_model_instance.m_glsl_buffer.m_buffer.size() / element_byte_size);
      }

      model_instance_t m_model_instance;

      fan::shader_t m_shader;
      uint32_t m_draw_node_reference;
      textures_t m_textures;
      public:
        f32_t s = 2;
    };

  }
}