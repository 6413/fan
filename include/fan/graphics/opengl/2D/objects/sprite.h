#pragma once

#include _FAN_PATH(graphics/opengl/gl_core.h)
#include _FAN_PATH(graphics/opengl/gl_shader.h)
#include _FAN_PATH(graphics/shared_graphics.h)
#include _FAN_PATH(graphics/opengl/texture_pack.h)
#include _FAN_PATH(physics/collision/rectangle.h)

namespace fan_2d {
  namespace opengl {

    struct sprite_t {

      sprite_t() = default;

      struct open_properties_t {

        open_properties_t() {};
      };

      void open(fan::opengl::context_t* context, open_properties_t p = open_properties_t()) {

        m_shader.open(context);

        m_shader.set_vertex(
          context,
          #include _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.vs)
        );

        m_shader.set_fragment(
          context,
          #include _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.fs)
        );

        m_shader.compile(context);

        m_store_sprite.open();
        m_glsl_buffer.open(context);
        m_glsl_buffer.init(context, m_shader.id, element_byte_size);
        m_queue_helper.open();

        m_draw_node_reference = fan::uninitialized;
      }
      void close(fan::opengl::context_t* context) {

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

        fan::opengl::image_t image;

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

        if (properties.image.texture != fan::uninitialized) {
          m_store_sprite.resize(m_store_sprite.size() + 1);
          m_store_sprite[m_store_sprite.size() - 1].image = properties.image;
        }
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
        sst.image = properties.image;

        m_store_sprite.insert(i, sst);
      }

      void load_texturepack(fan::opengl::context_t* context, uint32_t i, fan::opengl::texturepack* texture_packd, fan::opengl::texturepack::ti_t* ti) {
        m_store_sprite[i].image = texture_packd->pixel_data_list[ti->pack_id].image;
        const fan::vec2 texture_position = fan::cast<f32_t>(ti->position) / texture_packd->pixel_data_list[ti->pack_id].image.size;
        const fan::vec2 texture_size = fan::cast<f32_t>(ti->size) / texture_packd->pixel_data_list[ti->pack_id].image.size;
        this->set_texture_coordinates(context, i, {
          fan::vec2(texture_position.x, texture_position.y), // top left
          fan::vec2(texture_position.x + texture_size.x, texture_position.y), // top right
          fan::vec2(texture_position.x + texture_size.x, texture_position.y + texture_size.y), // bottom right
          fan::vec2(texture_position.x, texture_position.y + texture_size.y) // bottom left
        });
      }

      fan::opengl::image_t reload_sprite(fan::opengl::context_t* context, uint32_t i, fan::opengl::image_t image) {
        fan::opengl::image_t old = m_store_sprite[i].image;
        m_store_sprite[i].image = image;
        return old;
      }

      fan::opengl::image_t get_image(fan::opengl::context_t* context, uint32_t i) const {
        return m_store_sprite[i].image;
      }

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
      void erase(fan::opengl::context_t* context, uint32_t begin, uint32_t end) {
        m_glsl_buffer.erase_instance(context, begin * vertex_count, end - begin, element_byte_size, vertex_count);

        uint32_t to = m_glsl_buffer.m_buffer.size();
        m_store_sprite.erase(begin, end);
        if (to == 0) {
          // erase queue if there will be no objects left (special case)
          return;
        }
        else {
          m_queue_helper.edit(
            context,
            begin * vertex_count * element_byte_size,
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
            sizeof(rectangle_t::properties_t::color)
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

      void bind_matrices(fan::opengl::context_t* context, fan::opengl::matrices_t* matrices) {
				m_shader.bind_matrices(context, matrices);
			}

      void write_out(fan::opengl::context_t* context, FILE* f) const {
				uint64_t buffer_size = m_glsl_buffer.m_buffer.size();
				fwrite(&buffer_size, sizeof(uint64_t), 1, f);
				fwrite(m_glsl_buffer.m_buffer.data(), buffer_size, 1, f);

        for (uint32_t i = 0; i < this->size(context); i++) {
          uint8_t* data;
          m_store_sprite[i].image.get_pixel_data(context, &data);
          uint8_t* ptr;
          uint32_t ptr_size = fan::webp::encode_rgba(data, m_store_sprite[i].image.size, 100, &ptr);
          fwrite(&ptr_size, sizeof(ptr_size), 1, f);
          fwrite(ptr, ptr_size, 1, f);
          WebPFree(ptr);
        }
			}
			void write_in(fan::opengl::context_t* context, FILE* f) {
				uint64_t to_read;
				fread(&to_read, sizeof(uint64_t), 1, f);
				m_glsl_buffer.m_buffer.resize(to_read);
				fread(m_glsl_buffer.m_buffer.data(), to_read, 1, f);
				m_glsl_buffer.write_vram_all(context);
        
        m_store_sprite.resize(this->size(context));

        for (uint32_t i = 0; i < this->size(context); i++) {
          uint32_t image_size;
          fread(&image_size, sizeof(image_size), 1, f);
          uint8_t* data = (uint8_t*)malloc(image_size);

          fan::webp::image_info_t image_info;
          image_info.data = WebPDecodeRGBA(
            data,
            image_size,
            &image_info.size.x,
            &image_info.size.y
          );
          m_store_sprite[i].image.load(context, image_info);

          free(data);
          fan::webp::free_image(image_info.data);
        }
			}

      void write_out_texturepack(fan::opengl::context_t* context, FILE* f) const {
				uint64_t buffer_size = m_glsl_buffer.m_buffer.size();
				fwrite(&buffer_size, sizeof(uint64_t), 1, f);
				fwrite(m_glsl_buffer.m_buffer.data(), buffer_size, 1, f);
			}
      // a bit design problem, need to provei 
      void write_in_texturepack(fan::opengl::context_t* context, FILE* f, fan::opengl::texturepack* tp, uint32_t pack_id) {
				uint64_t to_read;
				fread(&to_read, sizeof(uint64_t), 1, f);
				m_glsl_buffer.m_buffer.resize(to_read);
				fread(m_glsl_buffer.m_buffer.data(), to_read, 1, f);
				m_glsl_buffer.write_vram_all(context);
        m_store_sprite.resize(this->size(context));

        for (uint32_t i = 0; i < this->size(context); i++) {
          m_store_sprite[i].image = tp->get_pixel_data(pack_id).image;
        }
			}

      fan::mat4 projection;
      fan::mat4 view;

      void draw(fan::opengl::context_t* context) {
        m_shader.use(context);

        uint32_t texture_id = fan::uninitialized;
        uint32_t from = 0;
        uint32_t to = 0;
        for (uint32_t i = 0; i < this->size(context); i++) {
          if (texture_id != m_store_sprite[i].image.texture) {
            if (to) {
              m_glsl_buffer.draw(
                context,
                (from)*vertex_count,
                (from + to) * vertex_count
              );
            }
            from = i;
            to = 0;
            texture_id = m_store_sprite[i].image.texture;
            m_shader.set_int(context, "texture_sampler", 0);
            context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
            context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, texture_id);
          }
          to++;
        }

        if (to) {
          m_glsl_buffer.draw(
            context,
            (from)*vertex_count,
            (from + to) * vertex_count
          );
        }

      }

      struct store_sprite_t {
        fan::opengl::image_t image;
      };

      fan::hector_t<store_sprite_t> m_store_sprite;

      fan::shader_t m_shader;
      fan::opengl::core::glsl_buffer_t m_glsl_buffer;
      fan::opengl::core::queue_helper_t m_queue_helper;
      uint32_t m_draw_node_reference;
    };

  }
}