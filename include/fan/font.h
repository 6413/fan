
#pragma once

#include <fan/io/file.h>

namespace fan {

	namespace font {

		// -y offset from top
		static constexpr auto highest_character = -2;

		struct mapping_t {
			uint32_t parse_index;
		};

		// physical letter info
		struct metrics_info_t {
			fan::vec2 size;
			fan::vec2 offset;
			f32_t advance;
		};

		// image info
		struct glyph_info_t {
			fan::vec2 position;
			fan::vec2 size;
			f32_t border;
		};

		struct single_info_t {
			mapping_t mapping;
			metrics_info_t metrics;
			glyph_info_t glyph;
		};

		using characters_t = std::unordered_map<uint32_t, single_info_t>;

		struct font_t {
			f32_t size;
			characters_t characters;
			f32_t line_height;

			
			auto get_character(wchar_t c) {
				auto found = characters.find(c);
				#if fan_debug >= fan_debug_low
					if (found == characters.end()) {
						fan::throw_error("failed to find character from font");
					}
				#endif
				return found->second;
			}

			uint16_t get_font_index(fan::character_t character) const {
				auto found = characters.find(character.c);
			#if fan_debug >= fan_debug_low
				if (found == characters.end()) {
					fan::throw_error("failed to find character from font");
				}
			#endif
				return std::distance(characters.begin(), found);
			}
			characters_t::const_iterator get_font_instance(uint16_t font_index) const {
				fan::font::characters_t::const_iterator found = characters.begin();
				std::advance(found, font_index);
				return found;
			}
			f32_t convert_font_size(f32_t font_size) const {
				return font_size / this->size;
			}
			fan::font::single_info_t get_letter_info(uint16_t font_index, f32_t font_size) const {

				auto found = get_font_instance(font_index);

			#if fan_debug >= fan_debug_low
				if (found == characters.end()) {
					throw std::runtime_error("failed to find character with font index: " + std::to_string(font_index));
				}
			#endif

				f32_t converted_size = convert_font_size(font_size);

				fan::font::single_info_t font_info;
				font_info.metrics.size = found->second.metrics.size * converted_size;
				font_info.metrics.offset = found->second.metrics.offset * converted_size;
				font_info.metrics.advance = (found->second.metrics.advance * converted_size);

				font_info.glyph = found->second.glyph;
				font_info.mapping = found->second.mapping;

				return font_info;
			}

			fan::font::single_info_t get_letter_info(wchar_t c, f32_t font_size) const {
				auto found = this->characters.find(c);

			#if fan_debug >= fan_debug_low
				if (found == characters.end()) {
					throw std::runtime_error("failed to find character: " + std::to_string((int)c));
				}
			#endif

				f32_t converted_size = convert_font_size(font_size);

				fan::font::single_info_t font_info;
				font_info.metrics.size = found->second.metrics.size * converted_size;
				font_info.metrics.offset = found->second.metrics.offset * converted_size;
				font_info.metrics.advance = (found->second.metrics.advance * converted_size);

				font_info.glyph = found->second.glyph;
				font_info.mapping = found->second.mapping;

				return font_info;
			}
			f32_t get_line_height(f32_t font_size) const {
				return line_height * convert_font_size(font_size);
			}
			fan::vec2 get_text_size(const fan::utf16_string& text, f32_t font_size) {
				fan::vec2 text_size = 0;

				text_size.y = line_height;

				f32_t width = 0;

				for (int i = 0; i < text.size(); i++) {

					switch (text[i]) {
					case '\n': {
						text_size.x = std::max(width, text_size.x);
						text_size.y += line_height;
						width = 0;
						continue;
					}
					}

					auto letter = characters[text[i]];

					if (i == text.size() - 1) {
						width += letter.glyph.size.x;
					}
					else {
						width += letter.metrics.advance;
					}
				}

				text_size.x = std::max(width, text_size.x);

				return text_size * convert_font_size(font_size);
			}

			f32_t advance(uint16_t font_index, f32_t font_size) const {
				return get_font_instance(font_index)->second.metrics.advance * convert_font_size(font_size);
			}
		};

		enum class parse_stage_e {
			mapping,
			metrics_info,
			glyph_info
		};

		struct line_t {
			uint32_t utf;
			single_info_t font_info;
		};

		static line_t parse_line(std::unordered_multimap<uint32_t, uint32_t>* reverse_mapping, const std::string& line, parse_stage_e stage) {
			switch (stage) {
				case parse_stage_e::mapping: {

					line_t l;

					auto r = fan::io::file::get_string_valuei_n(line);

					l.utf = r.value;

					r = fan::io::file::get_string_valuei_n(line, r.end);

					l.font_info.mapping.parse_index = r.value;

					reverse_mapping->insert(std::pair(r.value, l.utf));

					return l;
				}
				case parse_stage_e::metrics_info: {
					line_t l;

					auto r = fan::io::file::get_string_valuei_n(line);

					auto utf = reverse_mapping->find(r.value);

					if (utf == reverse_mapping->end()) {
						throw std::runtime_error("utf was not found from map index");
					}

					l.utf = utf->second;

					auto r2 = fan::io::file::get_string_valuevec2i_n(line, r.end);

					l.font_info.metrics.size = r2.value;

					r2 = fan::io::file::get_string_valuevec2i_n(line, r2.end);

					l.font_info.metrics.offset = r2.value;

					r = fan::io::file::get_string_valuei_n(line, r2.end);

					l.font_info.metrics.advance = r.value;

					return l;
				}
				case parse_stage_e::glyph_info: {
					line_t l;

					auto r = fan::io::file::get_string_valuei_n(line);

					auto utf = reverse_mapping->find(r.value);

					if (utf == reverse_mapping->end()) {
						throw std::runtime_error("utf was not found from map index");
					}

					l.utf = utf->second;

					auto r2 = fan::io::file::get_string_valuevec2i_n(line, r.end);

					l.font_info.glyph.position = r2.value;

					r2 = fan::io::file::get_string_valuevec2i_n(line, r2.end);

					l.font_info.glyph.size = r2.value;

					r = fan::io::file::get_string_valuei_n(line, r2.end);

					l.font_info.glyph.border = r.value;

					return l;
				}
			}
		}

		static font_t parse_font(const std::string& path) {
			if (!fan::io::file::exists(path)) {
				fan::print("font not found", path);
				exit(1);
			}

			std::ifstream file(path);

			std::vector<std::string> lines;
			std::string line;

			while (std::getline(file, line)) {
				lines.push_back(line);
			}

			f32_t flowest = -fan::math::inf;
			f32_t fhighest = fan::math::inf;

			f32_t lowest = 0, highest = 0;

			char previous_c = '0';

			std::size_t iline = 0;

			font_t font;

			while (lines[iline].substr(0, 4) != "font") {
				iline++;
			}

			auto r = fan::io::file::get_string_valuei_n(lines[iline]);

			r = fan::io::file::get_string_valuei_n(lines[iline], r.end);
			r = fan::io::file::get_string_valuei_n(lines[iline], r.end);

			font.size = r.value;

			font.line_height = font.size * 1.5;

			while (lines[iline++].find("# code index") == std::string::npos) {
			}

			parse_stage_e stage = parse_stage_e::mapping;

			std::unordered_multimap<uint32_t, uint32_t> reverse_mapping;

			while (1) {
				if (lines[iline] == "") {
					stage = parse_stage_e::metrics_info;
					break;
				}

				auto line = parse_line(&reverse_mapping, lines[iline], stage);

				font.characters[line.utf].mapping = line.font_info.mapping;

				previous_c = lines[iline][0];
				iline++;
			}

			while (lines[iline++].find("# index width height offset_x offset_y advance") == std::string::npos) {}

			while (1) {
				if (lines[iline] == "") {
					stage = parse_stage_e::glyph_info;
					break;
				}

				auto line = parse_line(&reverse_mapping, lines[iline], stage);

				font.characters[line.utf].metrics = line.font_info.metrics;

				previous_c = lines[iline][0];
				iline++;
			}

			while (lines[iline++].find("# index x y width height border") == std::string::npos) {}

			while (1) {

				if (lines[iline] == "") {

					font.characters[L'\n'].glyph.position = 0;
					font.characters[L'\n'].glyph.size = 0;
					font.characters[L'\n'].metrics.advance = 0;
					font.characters[L'\n'].metrics.offset = 0;
					font.characters[L'\n'].metrics.size = 0;

					for (auto& i : font.characters) {
						i.second.metrics.size = i.second.glyph.size;
					}

					return font;
				}

				auto line = parse_line(&reverse_mapping, lines[iline], stage);

				font.characters[line.utf].glyph = line.font_info.glyph;

				previous_c = lines[iline][0];
				iline++;
			}
		}
	}

}