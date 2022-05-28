#pragma once

//#include <fan/graphics/opengl/objects/text_renderer_raw.h>

//namespace fan_2d {
//	namespace graphics {
//		namespace gui {
//
//			struct text_renderer {
//
//				struct properties_t {
//					fan::utf16_string text;
//					f32_t font_size;
//					fan::vec2 position = 0;
//					fan::color text_color;
//					fan::color outline_color = fan::color(0, 0, 0, 0);
//					f32_t outline_size = 0.3;
//					fan::vec2 rotation_point = 0;
//					f32_t angle = 0;
//				};
//
//				using open_t = text_renderer_raw::open_t;
//
//				void open(fan::opengl::context_t* context, const open_t& open_properties = open_t()) {
//					m_tr_raw.open();
//				}
//				void close(fan::opengl::context_t* context) {
//					for (int i = 0; i < m_tr_raw.size(); i++) {
//						m_tr_raw[i].close(context);
//					}
//					m_tr_raw.close();
//				}
//
//				void push_back(fan::opengl::context_t* context, properties_t properties) {
//					m_tr_raw.push_back(text_renderer_raw());
//					auto& instance = m_tr_raw[m_tr_raw.size() - 1];
//					instance.open(context);
//
//					text_renderer_raw::properties_t p;
//					p.angle = properties.angle;
//					p.font_size = properties.font_size;
//					p.outline_color = properties.outline_color;
//					p.outline_size = properties.outline_size;
//					p.rotation_point = properties.rotation_point;
//					p.text_color = properties.text_color;
//
//					fan::vec2 text_size = instance.font.get_text_size(properties.text, properties.font_size);
//
//					f32_t left = properties.position.x - text_size.x * 0.5;
//					properties.position.y += instance.font.size * instance.font.convert_font_size(properties.font_size) * 0.5;
//
//					for (int i = 0; i < properties.text.size(); i++) {
//
//						p.position = fan::vec2(left, properties.position.y);
//						p.font_id = instance.font.get_font_index(properties.text[i]);
//						instance.push_back(context, p);
//						left += instance.font.advance(p.font_id, p.font_size);
//					}
//				}
//
//				void erase(fan::opengl::context_t* context, uint32_t i) {
//					uint32_t s = m_tr_raw[i].size(context);
//					for (int j = 0; j < s; j++) {
//						m_tr_raw[i].erase(context, s - j);
//					}
//				}
//
//				void enable_draw(fan::opengl::context_t* context, uint32_t i) {
//					m_tr_raw[i].enable_draw(context);
//				}
//				void disable_draw(fan::opengl::context_t* context, uint32_t i) {
//					m_tr_raw[i].disable_draw(context);
//				}
//
//				uint32_t size(fan::opengl::context_t* context) const {
//					return m_tr_raw.size();
//				}
//
//				f32_t get_line_height(f32_t font_size) const {
//					return m_tr_raw[0].font.get_line_height(font_size);
//				}
//
//				fan::hector_t<text_renderer_raw> m_tr_raw;
//			};
//
//		}
//	}
//}