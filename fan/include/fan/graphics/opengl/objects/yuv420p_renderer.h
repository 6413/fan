#pragma once

namespace fan_2d {
  namespace graphics {

		struct yuv420p_renderer :
			public fan_2d::graphics::sprite {

			struct properties_t : public fan_2d::graphics::sprite::properties_t {
				fan_2d::graphics::pixel_data_t pixel_data;
			};

			void push_back(fan::opengl::context_t* context, const yuv420p_renderer::properties_t& properties) {

				m_store_sprite.resize(m_store_sprite.size() + 3);

				glGenTextures(1, &m_store_sprite[m_store_sprite.size() - 3].m_texture);
				glGenTextures(1, &m_store_sprite[m_store_sprite.size() - 2].m_texture);
				glGenTextures(1, &m_store_sprite[m_store_sprite.size() - 1].m_texture);

				glBindTexture(GL_TEXTURE_2D, m_store_sprite[m_store_sprite.size() - 3].m_texture);

				glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, properties.pixel_data.size.x, properties.pixel_data.size.y, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, properties.pixel_data.pixels[0]);
				//glGenerateMipmap(GL_TEXTURE_2D);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

				glBindTexture(GL_TEXTURE_2D, m_store_sprite[m_store_sprite.size() - 2].m_texture);

				glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, properties.pixel_data.size.x / 2, properties.pixel_data.size.y / 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, properties.pixel_data.pixels[1]);
				//	glGenerateMipmap(GL_TEXTURE_2D);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

				glBindTexture(GL_TEXTURE_2D, m_store_sprite[m_store_sprite.size() - 1].m_texture);

				glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, properties.pixel_data.size.x / 2, properties.pixel_data.size.y / 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, properties.pixel_data.pixels[2]);

				//glGenerateMipmap(GL_TEXTURE_2D);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

				glBindTexture(GL_TEXTURE_2D, 0);

				sprite::sprite::properties_t property;
				property.position = properties.position;
				property.size = properties.size;
				property.angle = properties.angle;
				property.rotation_point = properties.rotation_point;
				property.rotation_vector = properties.rotation_vector;

				fan_2d::graphics::sprite::push_back(context, property);

				image_size.emplace_back(properties.pixel_data.size);
			}

			void reload_pixels(fan::opengl::context_t* context, uint32_t i, const fan_2d::graphics::pixel_data_t& pixel_data) {
				glBindTexture(GL_TEXTURE_2D, m_store_sprite[i * 3].m_texture);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

				glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, pixel_data.size.x, pixel_data.size.y, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel_data.pixels[0]);

				//	glGenerateMipmap(GL_TEXTURE_2D);

				glBindTexture(GL_TEXTURE_2D, m_store_sprite[i * 3 + 1].m_texture);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

				glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, pixel_data.size.x / 2, pixel_data.size.y / 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel_data.pixels[1]);

				//glGenerateMipmap(GL_TEXTURE_2D);

				glBindTexture(GL_TEXTURE_2D, m_store_sprite[i * 3 + 2].m_texture);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

				glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, pixel_data.size.x / 2, pixel_data.size.y / 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel_data.pixels[2]);

				//glGenerateMipmap(GL_TEXTURE_2D);

				glBindTexture(GL_TEXTURE_2D, 0);

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
				m_shader.use();

				m_shader.set_int("sampler_y", 0);
				m_shader.set_int("sampler_u", 1);
				m_shader.set_int("sampler_v", 2);

				for (int i = 0; i < sprite::size(context); i++) {
					glActiveTexture(GL_TEXTURE0 + 0);
					glBindTexture(GL_TEXTURE_2D, m_store_sprite[i * 3].m_texture);

					glActiveTexture(GL_TEXTURE0 + 1);
					glBindTexture(GL_TEXTURE_2D, m_store_sprite[i * 3 + 1].m_texture);

					glActiveTexture(GL_TEXTURE0 + 2);
					glBindTexture(GL_TEXTURE_2D, m_store_sprite[i * 3 + 2].m_texture);

					m_glsl_buffer.draw(
						context,
						m_shader,
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