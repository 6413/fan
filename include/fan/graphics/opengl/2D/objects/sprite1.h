#pragma once

#include _FAN_PATH(graphics/opengl/gl_core.h)
#include _FAN_PATH(graphics/opengl/gl_shader.h)
#include _FAN_PATH(graphics/shared_graphics.h)
#include _FAN_PATH(graphics/opengl/texture_pack.h)
#include _FAN_PATH(physics/collision/rectangle.h)

namespace fan_2d {
  namespace opengl {

    struct sprite1_t {

      typedef void(*erase_cb_t)(void*, uint64_t, uint32_t);

      sprite1_t() = default;

      struct open_properties_t {
        open_properties_t() {};
      };

      void open(fan::opengl::context_t* context, const open_properties_t& p, void* user_ptr, erase_cb_t erase_cb) {

        m_shader.open(context);

        m_shader.set_vertex(
          context,
          #include _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite1.vs)
        );

        m_shader.set_fragment(
          context,
          #include _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite1.fs)
        );

        m_shader.compile(context);

        m_store_sprite.open();
        m_glsl_buffer.open(context);
        m_glsl_buffer.init(context, m_shader.id, element_byte_size);
        m_queue_helper.open();

        m_draw_node_reference = fan::uninitialized;

        m_erase_cb = erase_cb;
				m_user_ptr = user_ptr;
      }

      void close(fan::opengl::context_t* context)
      {
        m_glsl_buffer.close(context);
        m_queue_helper.close(context);
        m_shader.close(context);

        if (m_draw_node_reference == fan::uninitialized) {
          return;
        }

        context->disable_draw(m_draw_node_reference);
        m_draw_node_reference = fan::uninitialized;

        m_store_sprite.close();
      }

      struct properties_t {

        fan::color color = fan::color(1, 1, 1, 1);
        fan::vec2 position = 0;
        fan::vec2 size = 0;
        f32_t angle = 0;
        fan::vec2 rotation_point = 0;
        fan::vec3 rotation_vector = 0;

        fan::_vec4<fan::vec2> texture_coordinates = fan::opengl::default_texture_coordinates;

        fan::opengl::image_t image{(uint32_t)fan::uninitialized, fan::uninitialized};
        fan::opengl::image_t light_map{(uint32_t)fan::uninitialized, fan::uninitialized};

        uint64_t id = fan::uninitialized;

        void load_texturepack(fan::opengl::context_t* context, fan::opengl::texturepack* texture_packd, fan::opengl::texturepack::ti_t* ti) {
          image = texture_packd->pixel_data_list[ti->pack_id].image;
          const fan::vec2 texture_position = fan::cast<f32_t>(ti->position) / image.size;
          const fan::vec2 texture_size = fan::cast<f32_t>(ti->size) / image.size;
          texture_coordinates = {
            fan::vec2(texture_position.x, texture_position.y), // top left
            fan::vec2(texture_position.x + texture_size.x, texture_position.y), // top right
            fan::vec2(texture_position.x + texture_size.x, texture_position.y + texture_size.y), // bottom right
            fan::vec2(texture_position.x, texture_position.y + texture_size.y) // bottom left
          };
        }
      };

    private:

      struct instance_t {

        fan::color color;
        fan::vec2 position;
        fan::vec2 size;
        f32_t angle;
        fan::vec2 rotation_point;
        fan::vec3 rotation_vector;

        fan::vec2 texture_coordinates;
      };

    public:

      static constexpr uint32_t offset_color = offsetof(instance_t, color);
      static constexpr uint32_t offset_position = offsetof(instance_t, position);
      static constexpr uint32_t offset_size = offsetof(instance_t, size);
      static constexpr uint32_t offset_angle = offsetof(instance_t, angle);
      static constexpr uint32_t offset_rotation_point = offsetof(instance_t, rotation_point);
      static constexpr uint32_t offset_rotation_vector = offsetof(instance_t, rotation_vector);
      static constexpr uint32_t offset_texture_coordinates = offsetof(instance_t, texture_coordinates);
      static constexpr uint32_t element_byte_size = offset_texture_coordinates + sizeof(fan::vec2);

      static constexpr uint32_t vertex_count = 6;

      // fan::opengl::load::texture
      void push_back(fan::opengl::context_t* context, const properties_t& properties) {

        instance_t instance;
        instance.color = properties.color;
        instance.position = properties.position;
        instance.size = properties.size;
        instance.angle = properties.angle;
        instance.rotation_point = properties.rotation_point;
        instance.rotation_vector = properties.rotation_vector;

        for (int i = 0; i < vertex_count; i++) {
          instance.texture_coordinates = fan_2d::opengl::convert_tc_4_2_6(&properties.texture_coordinates, i);
          m_glsl_buffer.push_ram_instance(context, &instance, sizeof(instance));
        }
        m_queue_helper.edit(
          context,
          (this->size(context) - 1) * vertex_count * element_byte_size,
          (this->size(context)) * vertex_count * element_byte_size,
          &m_glsl_buffer
        );

        if (properties.image.texture == fan::uninitialized) {
          fan::throw_error("a");
        }
        if (properties.light_map.texture == fan::uninitialized) {
          fan::throw_error("a");
        }
        m_store_sprite.resize(m_store_sprite.size() + 1);
        m_store_sprite[m_store_sprite.size() - 1].image[0] = properties.image;
        m_store_sprite[m_store_sprite.size() - 1].image[1] = properties.light_map;
        m_store_sprite[m_store_sprite.size() - 1].m_id = properties.id;
      }

      void insert(fan::opengl::context_t* context, uint32_t i, const properties_t& properties) {

        instance_t instance;
        instance.color = properties.color;
        instance.position = properties.position;
        instance.size = properties.size;
        instance.angle = properties.angle;
        instance.rotation_point = properties.rotation_point;
        instance.rotation_vector = properties.rotation_vector;

        for (int j = 0; j < vertex_count; j++) {
          instance.texture_coordinates = fan_2d::opengl::convert_tc_4_2_6(&properties.texture_coordinates, j);
          m_glsl_buffer.insert_ram_instance(context, i * vertex_count + j, &instance, sizeof(instance));
        }
        m_queue_helper.edit(
          context,
          i * vertex_count * element_byte_size,
          (this->size(context)) * vertex_count * element_byte_size,
          &m_glsl_buffer
        );

        store_sprite_t sst;
        sst.image[0] = properties.image;
        sst.image[1] = properties.light_map;
        sst.m_id = properties.id;

        m_store_sprite.insert(i, sst);
      }

      void load_texturepack(fan::opengl::context_t* context, uint32_t i, fan::opengl::texturepack* texture_packd, fan::opengl::texturepack::ti_t* ti) {
        m_store_sprite[i].image[0] = texture_packd->pixel_data_list[ti->pack_id].image;
        const fan::vec2 texture_position = fan::cast<f32_t>(ti->position) / texture_packd->pixel_data_list[ti->pack_id].image.size;
        const fan::vec2 texture_size = fan::cast<f32_t>(ti->size) / texture_packd->pixel_data_list[ti->pack_id].image.size;
        this->set_texture_coordinates(context, i, {
          fan::vec2(texture_position.x, texture_position.y), // top left
          fan::vec2(texture_position.x + texture_size.x, texture_position.y), // top right
          fan::vec2(texture_position.x + texture_size.x, texture_position.y + texture_size.y), // bottom right
          fan::vec2(texture_position.x, texture_position.y + texture_size.y) // bottom left
        });
      }

     /* void reload_sprite(fan::opengl::context_t* context, uint32_t i, fan::opengl::image_t image) {
        m_store_sprite[i].image = image.texture;
      }*/

      std::array<fan::vec2, 4> get_texture_coordinates(fan::opengl::context_t* context, uint32_t i) {
        fan::vec2* coordinates = (fan::vec2*)m_glsl_buffer.get_instance(context, i * vertex_count, element_byte_size, offset_texture_coordinates);

        return std::array<fan::vec2, 4>{
          coordinates[0],
            coordinates[1],
            coordinates[2],
            coordinates[5]
        };
      }
      // set texture coordinates before position or size
      void set_texture_coordinates(fan::opengl::context_t* context, uint32_t i, const fan::_vec4<fan::vec2>& texture_coordinates) {

        for (uint32_t j = 0; j < vertex_count; j++) {
          fan::vec2 tc = fan_2d::opengl::convert_tc_4_2_6(&texture_coordinates, j);
          m_glsl_buffer.edit_ram_instance(
            context,
            i * vertex_count + j,
            &tc,
            element_byte_size,
            offset_texture_coordinates,
            sizeof(instance_t::texture_coordinates)
          );
        }
        m_queue_helper.edit(
          context,
          i * vertex_count * element_byte_size,
          (i + 1) * (vertex_count)*element_byte_size,
          &m_glsl_buffer
        );
      }

      void erase(fan::opengl::context_t* context, uint32_t i) {

        if (i != this->size(context) - 1) {
					if ((f32_t)m_glsl_buffer.m_buffer.size() / vertex_count / element_byte_size != m_glsl_buffer.m_buffer.size() / vertex_count / element_byte_size) {
						fan::print("problem");
					}

					std::memmove(
						m_glsl_buffer.m_buffer.begin() + i * vertex_count * element_byte_size, 
						m_glsl_buffer.m_buffer.end() - element_byte_size * vertex_count, 
						element_byte_size * vertex_count
					);

					m_glsl_buffer.erase_instance(context, (this->size(context) - 1) * vertex_count, 1, element_byte_size, sprite1_t::vertex_count);

					m_erase_cb(m_user_ptr, m_store_sprite[m_store_sprite.size() - 1].m_id, i);

					m_store_sprite[i] = *(m_store_sprite.end() - 1);

					m_store_sprite.pop_back();

					uint32_t to = m_glsl_buffer.m_buffer.size();
					if (to == 0) {
						// erase queue if there will be no objects left (special case)
						return;
					}
					m_queue_helper.edit(
						context,
						i * vertex_count * element_byte_size,
						to,
						&m_glsl_buffer
					);
				}
				else {

          m_glsl_buffer.erase_instance(context, i * vertex_count, 1, element_byte_size, vertex_count);

          uint32_t to = m_glsl_buffer.m_buffer.size();
          m_store_sprite.erase(i);
          if (to == 0) {
            // erase queue if there will be no objects left (special case)
            return;
          }

          m_queue_helper.edit(
            context,
            i * vertex_count * element_byte_size,
            to,
            &m_glsl_buffer
          );
        }
      }

      // removes everything
      void clear(fan::opengl::context_t* context) {
        m_glsl_buffer.clear_ram(context);
        m_store_sprite.clear();
      }

      fan_2d::opengl::rectangle_corners_t get_corners(fan::opengl::context_t* context, uint32_t i) const {
        auto position = this->get_position(context, i);
        auto size = this->get_size(context, i);

        fan::vec2 mid = position;

        auto corners = fan_2d::opengl::get_rectangle_corners_no_rotation(position, size);

        f32_t angle = -this->get_angle(context, i);

        fan::vec2 top_left = fan_2d::opengl::get_transformed_point(corners[0] - mid, angle) + mid;
        fan::vec2 top_right = fan_2d::opengl::get_transformed_point(corners[1] - mid, angle) + mid;
        fan::vec2 bottom_left = fan_2d::opengl::get_transformed_point(corners[2] - mid, angle) + mid;
        fan::vec2 bottom_right = fan_2d::opengl::get_transformed_point(corners[3] - mid, angle) + mid;

        return { top_left, top_right, bottom_left, bottom_right };
      }

      const fan::color get_color(fan::opengl::context_t* context, uint32_t i) const {
        return *(fan::color*)m_glsl_buffer.get_instance(context, i * vertex_count, element_byte_size, offset_color);
      }
      void set_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color) {
        for (int j = 0; j < vertex_count; j++) {
          m_glsl_buffer.edit_ram_instance(
            context,
            i * vertex_count + j,
            &color,
            element_byte_size,
            offset_color,
            sizeof(fan::color)
          );
        }

        m_queue_helper.edit(
          context,
          i * vertex_count * element_byte_size,
          (i + 1) * (vertex_count)*element_byte_size,
          &m_glsl_buffer
        );
      }

      fan::vec2 get_position(fan::opengl::context_t* context, uint32_t i) const {
        return *(fan::vec2*)m_glsl_buffer.get_instance(context, i * vertex_count, element_byte_size, offset_position);
      }
      void set_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) {
        for (int j = 0; j < vertex_count; j++) {
          m_glsl_buffer.edit_ram_instance(
            context,
            i * vertex_count + j,
            &position,
            element_byte_size,
            offset_position,
            sizeof(properties_t::position)
          );
        }
        m_queue_helper.edit(
          context,
          i * vertex_count * element_byte_size,
          (i + 1) * (vertex_count)*element_byte_size,
          &m_glsl_buffer
        );
      }

      fan::vec2 get_size(fan::opengl::context_t* context, uint32_t i) const {
        return *(fan::vec2*)m_glsl_buffer.get_instance(context, i * vertex_count, element_byte_size, offset_size);
      }
      void set_size(fan::opengl::context_t* context, uint32_t i, const fan::vec2& size) {
        for (int j = 0; j < vertex_count; j++) {
          m_glsl_buffer.edit_ram_instance(
            context,
            i * vertex_count + j,
            &size,
            element_byte_size,
            offset_size,
            sizeof(properties_t::size)
          );
        }
        m_queue_helper.edit(
          context,
          i * vertex_count * element_byte_size,
          (i + 1) * (vertex_count)*element_byte_size,
          &m_glsl_buffer
        );
      }

      f32_t get_angle(fan::opengl::context_t* context, uint32_t i) const {
        return *(f32_t*)m_glsl_buffer.get_instance(context, i * vertex_count, element_byte_size, offset_angle);
      }
      void set_angle(fan::opengl::context_t* context, uint32_t i, f32_t angle) {
        f32_t a = fmod(angle, fan::math::pi * 2);

        for (int j = 0; j < vertex_count; j++) {
          m_glsl_buffer.edit_ram_instance(
            context,
            i * vertex_count + j,
            &a,
            element_byte_size,
            offset_angle,
            sizeof(properties_t::angle)
          );
        }
        m_queue_helper.edit(
          context,
          i * vertex_count * element_byte_size,
          (i + 1) * (vertex_count)*element_byte_size,
          &m_glsl_buffer
        );
      }

      fan::vec2 get_rotation_point(fan::opengl::context_t* context, uint32_t i) const {
        return *(fan::vec2*)m_glsl_buffer.get_instance(context, i * vertex_count, element_byte_size, offset_rotation_point);
      }
      void set_rotation_point(fan::opengl::context_t* context, uint32_t i, const fan::vec2& rotation_point) {
        for (int j = 0; j < vertex_count; j++) {
          m_glsl_buffer.edit_ram_instance(
            context,
            i * vertex_count + j,
            &rotation_point,
            element_byte_size,
            offset_rotation_point,
            sizeof(properties_t::rotation_point)
          );
        }
        m_queue_helper.edit(
          context,
          i * vertex_count * element_byte_size,
          (i + 1) * (vertex_count)*element_byte_size,
          &m_glsl_buffer
        );
      }

      fan::vec3 get_rotation_vector(fan::opengl::context_t* context, uint32_t i) const {
        return *(fan::vec3*)m_glsl_buffer.get_instance(context, i * vertex_count, element_byte_size, offset_rotation_vector);
      }
      void set_rotation_vector(fan::opengl::context_t* context, uint32_t i, const fan::vec3& rotation_vector) {
        for (int j = 0; j < vertex_count; j++) {
          m_glsl_buffer.edit_ram_instance(
            context,
            i * vertex_count + j,
            &rotation_vector,
            element_byte_size,
            offset_rotation_vector,
            sizeof(properties_t::rotation_vector)
          );
        }
        m_queue_helper.edit(
          context,
          i * vertex_count * element_byte_size,
          (i + 1) * (vertex_count)*element_byte_size,
          &m_glsl_buffer
        );
      }

      uint32_t size(fan::opengl::context_t* context) const {
        return m_glsl_buffer.m_buffer.size() / element_byte_size / vertex_count;
      }

      bool inside(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) const {

        auto corners = get_corners(context, i);

        return fan_2d::collision::rectangle::point_inside(
          corners[0],
          corners[1],
          corners[2],
          corners[3],
          position
        );
      }

      void enable_draw(fan::opengl::context_t* context) {
        this->draw(context);

        #if fan_debug >= fan_debug_low
        if (m_draw_node_reference != fan::uninitialized) {
          fan::throw_error("trying to call enable_draw twice");
        }
        #endif

        m_draw_node_reference = context->enable_draw(this, [](fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
      }
      void disable_draw(fan::opengl::context_t* context) {
        #if fan_debug >= fan_debug_low
        if (m_draw_node_reference == fan::uninitialized) {
          fan::throw_error("trying to disable unenabled draw call");
        }
        #endif
        context->disable_draw(m_draw_node_reference);
      }

      fan::mat4 projection;
      fan::mat4 view;

      void draw(fan::opengl::context_t* context) {
        m_shader.use(context);

        uint32_t texture_id = fan::uninitialized;
        uint32_t from = 0;
        uint32_t to = 0;
        for (uint32_t i = 0; i < this->size(context); i++) {
          if (texture_id != m_store_sprite[i].image[0].texture) {
            if (to) {
              m_glsl_buffer.draw(
                context,
                (from)*vertex_count,
                (from + to) * vertex_count
              );
            }
            from = i;
            to = 0;
            texture_id = m_store_sprite[i].image[0].texture;
            m_shader.set_int(context, "texture_sampler", 0);
            context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
            context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, texture_id);

            texture_id = m_store_sprite[i].image[1].texture;
            m_shader.set_int(context, "texture_light_map", 1);
            context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
            context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, texture_id);
          }
          to++;
        }

        if (to) {
          fan::opengl::GLint viewport[4];
          context->opengl.glGetIntegerv(fan::opengl::GL_VIEWPORT, viewport);
          m_shader.set_vec2(context, "viewport_size", fan::vec2(viewport[2], viewport[3]));

          m_glsl_buffer.draw(
            context,
            (from)*vertex_count,
            (from + to) * vertex_count
          );
        }

      }

      struct store_sprite_t {
        fan::opengl::image_t image[2];
        uint64_t m_id;
      };

      void* m_user_ptr;

			erase_cb_t m_erase_cb;

      fan::hector_t<store_sprite_t> m_store_sprite;

      fan::shader_t m_shader;
      fan::opengl::core::glsl_buffer_t m_glsl_buffer;
      fan::opengl::core::queue_helper_t m_queue_helper;
      uint32_t m_draw_node_reference;
    };

  }
}