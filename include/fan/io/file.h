#pragma once

#include _FAN_PATH(types/types.h)
#include _FAN_PATH(types/vector.h)

#include <fstream>
#include <string>
#include <filesystem>

namespace fan {
	namespace io {
		namespace file {

			using file_t = FILE;
			
			struct properties_t {
				const char* mode;
			};

			static bool close(file_t* f) {
				int ret = fclose(f);
				#if fan_debug >= fan_debug_low
					if (ret != 0) {
						fan::print_warning("failed to close file stream");
						return 1;
					}
				#endif
				return 0;
			}
			static bool open(file_t** f, const char* path, const properties_t& p) {
				*f = fopen(path, p.mode);
				if (f == nullptr) {
					fan::print_warning(std::string("failed to open file:") + path);
					close(*f);
					return 1;
				}
				return 0;
			}

			static bool write(file_t* f, void* data, uint64_t size, uint64_t elements) {
				uint64_t ret = fwrite(data, size, elements, f);
				#if fan_debug >= fan_debug_low
					if (ret != elements && size != 0) {
						fan::print_warning("failed to write from file stream");
						return 1;
					}
				#endif
				return 0;
			}
			static bool read(file_t* f, void* data, uint64_t size, uint64_t elements) {
				uint64_t ret = fread(data, size, elements, f);
				#if fan_debug >= fan_debug_low
					if (ret != elements && size != 0) {
						fan::print_warning("failed to read from file stream");
						return 1;
					}
				#endif
				return 0;
			}

			inline uint64_t file_size(const std::string& filename)
			{
				std::ifstream f(filename, std::ifstream::ate | std::ifstream::binary);
				return f.tellg(); 
			}

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

			static std::vector<std::string> read_line(const std::string& path) {

				std::ifstream file(path.c_str(), std::ifstream::binary);
				if (file.fail()) {
					fan::throw_error("path does not exist " + path);
				}
				std::vector<std::string> data;
				for (std::string line; std::getline(file, line); ) {
					data.push_back(line);
				}
				return data;
			}

			static bool read(const std::string& path, std::string* str) {

				std::ifstream file(path.c_str(), std::ifstream::ate | std::ifstream::binary);
				if (file.fail()) {
					fan::print_warning_no_space("path does not exist:" + path);
					return 1;
				}
				str->resize(file.tellg());
				file.seekg(0, std::ios::beg);
				file.read(&(*str)[0], str->size());
				file.close();
				return 0;
			}

			template <typename T>
			static std::vector<T> read(const std::string& path) {
				std::ifstream file(path.c_str(), std::ifstream::ate | std::ifstream::binary);
				if (!file.good()) {
					fan::print("path does not exist " + path);
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

			struct str_vec2i_t {
				std::size_t begin, end;
				fan::vec2i value;
			};

			static const char* digits = "0123456789";

			static int get_string_valuei(const std::string& str, const std::string& find, std::size_t offset = 0) {

				std::size_t found = str.find(find, offset);

				int64_t begin = str.find_first_of(digits, found);

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

				int64_t begin = str.find_first_of(digits, offset);

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

			static str_vec2i_t get_string_valuevec2i_n(const std::string& str, std::size_t offset = 0) {

				fan::vec2i v;

				std::size_t begin, end;

				auto r = get_string_valuei_n(str, offset);

				begin = r.begin;

				v.x = r.value;

				r = get_string_valuei_n(str, r.end);

				v.y = r.value;
				
				end = r.end;

				return { begin, end, v };
			}

		}
	}
}
