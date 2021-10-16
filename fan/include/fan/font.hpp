#pragma once

#include <fan/io/file.hpp>

namespace fan {

	namespace font {

		// -y offset from top
		static constexpr auto highest_character = -2;

		struct font_t {
			static constexpr std::size_t char_offset = 4;
			static constexpr std::size_t struct_size = 7;

			fan::vec2 position;
			fan::vec2 size;
			fan::vec2 offset;
			fan::vec2::value_type advance;
		};

		struct font_info {
			f32_t size;
			std::unordered_map<uint16_t, font_t> font;
			f32_t padding;
			f32_t lowest;
			f32_t highest;
			f32_t line_height;
		};

		static font_info parse_font(const std::string& path) {
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

			int amount_of_chars = fan::io::file::get_string_valuei(lines[3], "chars count");

			std::unordered_map<uint16_t, font_t> font_info_vector;

			f32_t flowest = -fan::math::inf;
			f32_t fhighest = fan::math::inf;

			f32_t lowest = 0, highest = 0;

			auto padding = (f32_t)fan::io::file::get_string_valuei(lines[0], "padding");

			for (std::size_t iline = font_t::char_offset; iline < amount_of_chars; iline++) {
				if (lines[iline][0] != 'c') {
					break;
				}
				font_t font_info;
				fan::io::file::str_int_t value_info;
				value_info = fan::io::file::get_string_valuei_n(lines[iline], 0);
				uint16_t character = value_info.value;
				for (std::size_t i = 0; i < font_t::struct_size; i++) {
					value_info = fan::io::file::get_string_valuei_n(lines[iline], value_info.end);
					if (iline - font_t::char_offset == 5) {
						if (value_info.value < highest_character) {
							value_info.value = highest_character;
						}
					}
					((fan::vec2::value_type*)&font_info)[i] = value_info.value;
				}

				// padding 8
				font_info.position += padding;
				font_info.size -= padding;
				font_info.advance -= padding * 2;
				font_info.offset -= padding * 2;
				font_info_vector[character] = font_info;
				if (flowest < font_info.offset.y) {
					flowest = font_info.offset.y;
					lowest = character;
				}
				if (fhighest > font_info.offset.y) {
					if (fhighest > highest_character) {
						fhighest = font_info.offset.y;
						highest = character;
					}
				}
			}

			// doesn't set space automatically, need to set in load part
			return {
				(f32_t)(fan::io::file::get_string_valuei(lines[0], "size")),
				font_info_vector,
				padding,
				lowest,
				highest,
				(f32_t)(fan::io::file::get_string_valuei(lines[1], "lineHeight"))
			};
		}

	}

}