#pragma once

#include _FAN_PATH(graphics/opengl/gl_shader.h)
#include _FAN_PATH(graphics/shared_graphics.h)

namespace fan_2d {
  namespace opengl {

		struct yuv420p_renderer_t :
			public fan_2d::opengl::sprite_t {

			struct properties_t : public fan_2d::opengl::sprite_t::properties_t {
				fan_2d::opengl::pixel_data_t pixel_data;
				uint32_t filter = fan::opengl::image_load_properties_defaults::filter;
				uint32_t visual_output = fan::opengl::image_load_properties_defaults::visual_output;
			};

			void open(fan::opengl::context_t* context) {

				m_shader.open(context);

				m_shader.set_vertex(
					context, 
					#include _FAN_PATH(graphics/glsl/opengl/2D/objects/yuv420p_renderer.vs)
				);

				m_shader.set_fragment(
					context, 
					#include _FAN_PATH(graphics/glsl/opengl/2D/objects/yuv420p_renderer.fs)
				);

				m_shader.compile(context);

				m_store_sprite.open();
				m_glsl_buffer.open(context);
				m_glsl_buffer.init(context, m_shader.id, element_byte_size);
				m_queue_helper.open();

				m_draw_node_reference = fan::uninitialized;
			}

			void push_back(fan::opengl::context_t* context, const yuv420p_renderer_t::properties_t& properties) {

				m_store_sprite.resize(m_store_sprite.size() + 3);

				context->opengl.glGenTextures(1, &m_store_sprite[m_store_sprite.size() - 3].m_texture);
				context->opengl.glGenTextures(1, &m_store_sprite[m_store_sprite.size() - 2].m_texture);
				context->opengl.glGenTextures(1, &m_store_sprite[m_store_sprite.size() - 1].m_texture);

				context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_store_sprite[m_store_sprite.size() - 3].m_texture);

				context->opengl.glTexImage2D(fan::opengl::GL_TEXTURE_2D, 0, fan::opengl::GL_LUMINANCE, properties.pixel_data.size.x, properties.pixel_data.size.y, 0, fan::opengl::GL_LUMINANCE, fan::opengl::GL_UNSIGNED_BYTE, properties.pixel_data.pixels[0]);
				//glGenerateMipmap(GL_TEXTURE_2D);

				context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_S, properties.visual_output);
				context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_T, properties.visual_output);
				context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MIN_FILTER, properties.filter);
				context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MAG_FILTER, properties.filter);

				context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_store_sprite[m_store_sprite.size() - 2].m_texture);

				context->opengl.glTexImage2D(fan::opengl::GL_TEXTURE_2D, 0, fan::opengl::GL_LUMINANCE, properties.pixel_data.size.x / 2, properties.pixel_data.size.y / 2, 0, fan::opengl::GL_LUMINANCE, fan::opengl::GL_UNSIGNED_BYTE, properties.pixel_data.pixels[1]);
				//	glGenerateMipmap(GL_TEXTURE_2D);

				context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_S, properties.visual_output);
				context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_T, properties.visual_output);
				context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MIN_FILTER, properties.filter);
				context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MAG_FILTER, properties.filter);

				context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_store_sprite[m_store_sprite.size() - 1].m_texture);

				context->opengl.glTexImage2D(fan::opengl::GL_TEXTURE_2D, 0, fan::opengl::GL_LUMINANCE, properties.pixel_data.size.x / 2, properties.pixel_data.size.y / 2, 0, fan::opengl::GL_LUMINANCE, fan::opengl::GL_UNSIGNED_BYTE, properties.pixel_data.pixels[2]);

				//glGenerateMipmap(GL_TEXTURE_2D);

				context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_S, properties.visual_output);
				context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_T, properties.visual_output);
				context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MIN_FILTER, properties.filter);
				context->opengl.glTexParameteri(fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MAG_FILTER, properties.filter);

				context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, 0);

				sprite_t::properties_t property;
				property.position = properties.position;
				property.size = properties.size;
				property.angle = properties.angle;
				property.rotation_point = properties.rotation_point;
				property.rotation_vector = properties.rotation_vector;
				property.texture_coordinates = properties.texture_coordinates;
				property.image = nullptr;

				fan_2d::opengl::sprite_t::push_back(context, property);

				image_size.emplace_back(properties.pixel_data.size);
			}

			void reload_pixels(fan::opengl::context_t* context, uint32_t i, const fan_2d::opengl::pixel_data_t& pixel_data) {
				context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_store_sprite[i * 3].m_texture);
				context->opengl.glTexImage2D(fan::opengl::GL_TEXTURE_2D, 0, fan::opengl::GL_LUMINANCE, pixel_data.size.x, pixel_data.size.y, 0, fan::opengl::GL_LUMINANCE, fan::opengl::GL_UNSIGNED_BYTE, pixel_data.pixels[0]);
				context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_store_sprite[i * 3 + 1].m_texture);
				context->opengl.glTexImage2D(fan::opengl::GL_TEXTURE_2D, 0, fan::opengl::GL_LUMINANCE, pixel_data.size.x / 2, pixel_data.size.y / 2, 0, fan::opengl::GL_LUMINANCE, fan::opengl::GL_UNSIGNED_BYTE, pixel_data.pixels[1]);
				context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_store_sprite[i * 3 + 2].m_texture);
				context->opengl.glTexImage2D(fan::opengl::GL_TEXTURE_2D, 0, fan::opengl::GL_LUMINANCE, pixel_data.size.x / 2, pixel_data.size.y / 2, 0, fan::opengl::GL_LUMINANCE, fan::opengl::GL_UNSIGNED_BYTE, pixel_data.pixels[2]);

				image_size[i] = pixel_data.size;
			}

			fan::vec2ui get_image_size(fan::opengl::context_t* context, uint32_t i) const
			{
				return this->image_size[i];
			}

			void enable_draw(fan::opengl::context_t* context)
			{
				m_draw_node_reference = context->enable_draw(this, [](fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
			}

		private:

			void draw(fan::opengl::context_t* context)
			{
				m_shader.use(context);

				m_shader.set_int(context, "sampler_y", 0);
				m_shader.set_int(context, "sampler_u", 1);
				m_shader.set_int(context, "sampler_v", 2);

				for (int i = 0; i < sprite_t::size(context); i++) {
					context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + 0);
					context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_store_sprite[i * 3].m_texture);

					context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + 1);
					context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_store_sprite[i * 3 + 1].m_texture);

					context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + 2);
					context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_store_sprite[i * 3 + 2].m_texture);

					m_shader.use(context);

					m_glsl_buffer.draw(
						context,
						i * vertex_count,
						(i + 1) * vertex_count
					);
				}
			}

			std::vector<fan::vec2ui> image_size;

			static constexpr auto layout_y = "layout_y";
			static constexpr auto layout_u = "layout_u";
			static constexpr auto layout_v = "layout_v";

		};
  }
}