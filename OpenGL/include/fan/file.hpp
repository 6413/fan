#pragma once

#include <fan/types/types.hpp>

#include <fstream>
#include <string>

namespace fan {
	namespace io {
		namespace file {
			inline bool exists(const std::string& name) {
				std::ifstream file(name);
				return file.good();
			}

			inline void write(
				std::string path,
				const std::string& data,
				decltype(std::ios_base::binary | std::ios_base::app) mode = std::ios_base::binary | std::ios_base::app
			) {
				std::ofstream ofile(path, mode);
				ofile << data;
				ofile.close();
			}

			template <typename T>
			static inline void write(
				std::string path,
				const std::vector<T>& vector,
				decltype(std::ios_base::binary | std::ios_base::app) mode = std::ios_base::binary | std::ios_base::app
			) {
				std::ofstream ofile(path, mode);
				ofile.write(reinterpret_cast<const char*>(&vector[0]), vector.size() * sizeof(T));
				ofile.close();
			}

			static std::string read(const std::string& path) {
				std::ifstream file(path.c_str(), std::ifstream::ate | std::ifstream::binary);
				if (!file.good()) {
					fan::print("path does not exist", path);
					exit(1);
				}
				std::string data;
				data.resize(file.tellg());
				file.seekg(0, std::ios::beg);
				file.read(&data[0], data.size());
				file.close();
				return data;
			}

			template <typename T>
			static std::vector<T> read(const std::string& path) {
				std::ifstream file(path.c_str(), std::ifstream::ate | std::ifstream::binary);
				if (!file.good()) {
					fan::print("path does not exist", path);
					exit(1);
				}
				std::vector<T> vector;
				uint64_t size = file.tellg();
				vector.resize(size / sizeof(T));
				file.seekg(0, std::ios::beg);
				file.read(reinterpret_cast<char*>(&vector[0]), size);
				file.close();
				return vector;
			}

			struct str_int_t {
				std::size_t begin, end;
				int value;
			};

			static const char* digits = "0123456789";

			static int get_string_valuei(const std::string& str, const std::string& find, std::size_t offset = 0) {

				std::size_t found = str.find(find, offset);

				std::size_t begin = str.find_first_of(digits, found);

				bool negative = 0;

				if (begin - 1 >= 0) {
					if (str[begin - 1] == '-') {
						negative = 1;
					}
				}

				std::size_t end = str.find_first_not_of(digits, begin);

				if (end == std::string::npos) {
					end = str.size();
				}

				return std::stoi(std::string(str.begin() + begin - negative, str.begin() + end));
			}

			static str_int_t get_string_valuei_n(const std::string& str, std::size_t offset = 0) {

				std::size_t begin = str.find_first_of(digits, offset);

				bool negative = 0;

				if (begin - 1 >= 0) {
					if (str[begin - 1] == '-') {
						negative = 1;
					}
				}

				std::size_t end = str.find_first_not_of(digits, begin);
				if (end == std::string::npos) {
					end = str.size();
				}

				return { (std::size_t)begin, end, std::stoi(std::string(str.begin() + begin - negative, str.begin() + end)) };
			}

			struct font_t {
				static constexpr std::size_t char_offset = 4;
				static constexpr std::size_t struct_size = 7;

				fan::vec2 m_position;
				fan::vec2 m_size;
				fan::vec2 m_offset;
				fan::vec2::type m_advance;
			};

			struct font_info {
				unsigned int m_size;
				std::unordered_map<uint16_t, font_t> m_font;
				uint_t m_lowest;
				uint_t m_highest;
			};

			static font_info parse_font(const std::string& path)  {
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

				f_t flowest = -fan::inf;
				f_t fhighest = fan::inf;

				uint_t lowest = 0, highest = 0;

				for (std::size_t iline = font_t::char_offset; iline < amount_of_chars + font_t::char_offset + 1; iline++) {
					if (lines[iline][0] != 'c') {
						break;
					}
					font_t font_info;
					fan::io::file::str_int_t value_info;
					value_info =  fan::io::file::get_string_valuei_n(lines[iline], 0);
					uint16_t character = value_info.value;
					for (std::size_t i = 0; i < font_t::struct_size; i++) {
						value_info =  fan::io::file::get_string_valuei_n(lines[iline], value_info.end);
						((fan::vec2::type*)&font_info)[i] = value_info.value;
					}
					font_info_vector[character] = font_info;
					if (flowest > font_info.m_offset.y) {
						flowest = font_info.m_offset.y;
						lowest = character;
					}
					if (fhighest < font_info.m_offset.y) {
						fhighest = font_info.m_offset.y;
						highest = character;
					}
				}

				// doesn't set space automatically, need to set in load part
				return { static_cast<unsigned int>(fan::io::file::get_string_valuei(lines[0], "size")), font_info_vector, lowest, highest };
			}

		}
	}
}
