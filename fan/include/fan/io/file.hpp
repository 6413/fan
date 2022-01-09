#pragma once

#include <fan/types/types.hpp>

#include <fan/types/vector.hpp>

#include <fstream>
#include <string>
#include <filesystem>

namespace fan {
	namespace io {

		static bool directory_exists(const std::string& directory) {
			return std::filesystem::exists(directory);
		}

		static void iterate_directory(
			const std::string& path, 
			const std::function<void(const std::string& path)>& function
		) {

			if (!directory_exists(path)) {
				fan::throw_error("directory does not exist");
			}

			for (const auto & entry : std::filesystem::directory_iterator(path)) {
				if (entry.is_directory()) {
					iterate_directory(entry.path().string(), function);
					continue;
				}
				function(entry.path().string());
			}
		}

		namespace file {

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

			static std::string read(const std::string& path) {

				if (!exists(path)) {
					fan::print("path does not exist", path);
					exit(1);
				}

				std::ifstream file(path.c_str(), std::ifstream::ate | std::ifstream::binary);
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
