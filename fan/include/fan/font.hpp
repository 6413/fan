#pragma once

#include <fan/io/file.hpp>

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

		using font_info_t = std::unordered_map<uint32_t, single_info_t>;

		struct font_t {
			f32_t size;
			font_info_t font;
			f32_t line_height;
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

			while (lines[iline++] != "# code index") {}

			parse_stage_e stage = parse_stage_e::mapping;

			std::unordered_multimap<uint32_t, uint32_t> reverse_mapping;

			while (1) {
				if (lines[iline] == "") {
					stage = parse_stage_e::metrics_info;
					break;
				}

				auto line = parse_line(&reverse_mapping, lines[iline], stage);

				font.font[line.utf].mapping = line.font_info.mapping;

				previous_c = lines[iline][0];
				iline++;
			}

			while (lines[iline++] != "# index width height offset_x offset_y advance") {}

			while (1) {
				if (lines[iline] == "") {
					stage = parse_stage_e::glyph_info;
					break;
				}

				auto line = parse_line(&reverse_mapping, lines[iline], stage);

				font.font[line.utf].metrics = line.font_info.metrics;

				previous_c = lines[iline][0];
				iline++;
			}

			while (lines[iline++] != "# index x y width height border") {}

			while (1) {

				if (lines[iline] == "") {

					font.font[L'\n'].glyph.position = 0;
					font.font[L'\n'].glyph.size = 0;
					font.font[L'\n'].metrics.advance = 0;
					font.font[L'\n'].metrics.offset = 0;
					font.font[L'\n'].metrics.size = 0;

					for (auto& i : font.font) {
						i.second.metrics.size = i.second.glyph.size;
					}

					return font;
				}

				auto line = parse_line(&reverse_mapping, lines[iline], stage);

				font.font[line.utf].glyph = line.font_info.glyph;

				previous_c = lines[iline][0];
				iline++;
			}
		}

	}

}