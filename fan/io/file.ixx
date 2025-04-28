module;

#include <fan/types/types.h>

import fan.types.print;
import fan.types.vector;

#include <fstream>
#include <string>
#include <sstream>
#undef index // xos.h

export module fan.io.file;

export namespace fan {
	namespace io {
		namespace file {

			std::string extension(const std::string& file_path) {
				size_t dotPosition = file_path.find_last_of('.');
				size_t sepPosition = file_path.find_last_of("/\\");

				if (dotPosition != std::string::npos && (sepPosition == std::string::npos || dotPosition > sepPosition)) {
					return file_path.substr(dotPosition);
				}
				else {
					return "";
				}
			}

			bool exists(const std::string& name) {
				std::ifstream file(name.c_str());
				return file.good();
			}


			bool rename(const std::string& from, const std::string& to) {
				return std::rename(from.c_str(), to.c_str());
			}

			struct fstream {
				fstream() = default;
				fstream(const std::string& path) {
					file_name = path;
					open(path);
				}
				fstream(const std::string& path, std::string* str) {
					file_name = path;
					open(path);
					read(str);
				}

				bool open(const std::string& path) {
					file_name = path;
					auto flags = std::ios::in | std::ios::out | std::ios::binary;

					if (!exists(path)) {
						flags |= std::ios::trunc;
					}

					file = std::fstream(path, flags);
					return !file.good();
				}

				bool read(const std::string& path, std::string* str) {
					file_name = path;
					open(path);
					return read(str);
				}

				bool read(std::string* str) {
					if (file.is_open()) {
						file.seekg(0, std::ios::end);
						str->resize(file.tellg());
						file.seekg(0, std::ios::beg);
						file.read(&(*str)[0], str->size());
					}
					else {
						fan::print_warning("file is not opened:");
						return 1;
					}
					return 0;
				}

				bool write(std::string* str) {
					auto flags = std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc;

					file = std::fstream(file_name, flags);
					if (file.is_open()) {
						file.write(&(*str)[0], str->size());
						file.flush();
					}
					else {
						fan::print_warning("file is not opened:");
						return 1;
					}
					flags &= ~std::ios::trunc;
					file = std::fstream(file_name, flags);
					return 0;
				}

				std::fstream file;
				std::string file_name;
			};

			using file_t = FILE;

			struct properties_t {
				const char* mode;
			};

			bool open(file_t** f, const std::string& path, const properties_t& p) {
				*f = fopen(path.c_str(), p.mode);
				if (*f == nullptr) {
					return 1;
				}
				return 0;
			}
			bool close(file_t* f) {
				int ret = fclose(f);
#if fan_debug >= fan_debug_low
				if (ret != 0) {
					fan::print_warning("failed to close file stream");
					return 1;
				}
#endif
				return 0;
			}

			bool read(file_t* f, void* data, uint64_t size, uint64_t elements) {
				uint64_t ret = fread(data, size, elements, f);
#if fan_debug >= fan_debug_low
				if (ret != elements && size != 0) {
					fan::print_warning("failed to read from file stream");
					return 1;
				}
#endif
				return 0;
			}

			bool write(file_t* f, void* data, uint64_t size, uint64_t elements) {
				uint64_t ret = fwrite(data, size, elements, f);
#if fan_debug >= fan_debug_low
				if (ret != elements && size != 0) {
					fan::print_warning("failed to write from file stream");
					return 1;
				}
#endif
				return 0;
			}

			uint64_t file_size(const std::string& filename) {
				std::ifstream f(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
				return f.tellg();
			}

			bool write(
				std::string path,
				const std::string& data,
				decltype(std::ios_base::binary | std::ios_base::app) mode = std::ios_base::binary | std::ios_base::app
			) {
				std::ofstream ofile(path.c_str(), mode);
				if (ofile.fail()) {
					fan::print_warning("failed to write to:" + path);
					return 0;
				}
				ofile.write(data.c_str(), data.size());
				return 1;
			}

			template <typename T>
			void write(
				std::string path,
				const std::vector<T>& vector,
				decltype(std::ios_base::binary | std::ios_base::app) mode = std::ios_base::binary | std::ios_base::app
			) {
				std::ofstream ofile(path.c_str(), mode);
				if (ofile.fail()) {
					fan::throw_error("failed to write to:" + path);
				}
				ofile.write(reinterpret_cast<const char*>(&vector[0]), vector.size() * sizeof(T));
			}

			std::vector<std::string> read_line(const std::string& path) {

				std::ifstream file(path.c_str(), std::ifstream::binary);
				if (file.fail()) {
					fan::throw_error("path does not exist:" + path);
				}
				std::vector<std::string> data;
				for (std::string line; std::getline(file, line); ) {
					data.push_back(line.c_str());
				}
				return data;
			}

			bool read(const std::string& path, std::string* str) {

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
			bool read(const std::string& path, std::string* str, std::size_t length) {

				std::ifstream file(path.c_str(), std::ifstream::binary);
				if (file.fail()) {
					fan::print_warning_no_space("path does not exist:" + path);
					return 1;
				}
				str->resize(length);
				file.seekg(0, std::ios::beg);
				file.read(&(*str)[0], length);
				file.close();
				return 0;
			}

			template <typename T>
			std::vector<T> read(const std::string& path) {
				std::ifstream file(path.c_str(), std::ifstream::ate | std::ifstream::binary);
				if (!file.good()) {
					fan::print("path does not exist:" + path);
					exit(1);
				}
				std::vector<T> vector;
				uint64_t size = file.tellg();
				vector.resize(size / sizeof(T));
				file.seekg(0, std::ios::beg);
				file.read(reinterpret_cast<char*>(&vector[0]), size);
				return vector;
			}

			std::string extract_variable_type(const std::string& string_data, const std::string& var_name) {
				std::istringstream file(string_data);

				std::string type;
				std::string line;
				while (std::getline(file, line)) {
					std::istringstream iss(line);
					std::string word;
					while (iss >> word) {
						if (word.find(var_name) != std::string::npos) {
							return type;
						}
						else {
							type = word;
						}
					}
				}

				return "";
			}

			struct str_int_t {
				std::size_t begin, end;
				int64_t value;
			};

			struct str_vec2i_t {
				std::size_t begin, end;
				fan::vec2i value;
			};

			constexpr const char* digits = "0123456789";

			int get_string_valuei(const std::string& str, const std::string& find, std::size_t offset = 0) {

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

				std::string ret(str.begin() + begin - negative, str.begin() + end);
				return std::stoi(ret.data());
			}

			str_int_t get_string_valuei_n(const std::string& str, std::size_t offset = 0) {
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

				std::string ret(str.begin() + begin - negative, str.begin() + end);
				return { (std::size_t)begin, end, std::stoi(ret.c_str()) };
			}

			str_vec2i_t get_string_valuevec2i_n(const std::string& str, std::size_t offset = 0) {

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

namespace fan {
	namespace tmpl {
		template<typename>
		struct is_std_vector : std::false_type {};

		template<typename T, typename A>
		struct is_std_vector<std::vector<T, A>> : std::true_type {};
	}
}

export namespace fan {
	template <typename T>
	void write_to_file(fan::io::file::file_t* f, const T& o) {
		if constexpr (std::is_same<std::string, T>::value ||
			std::is_same<std::string, T>::value) {
			uint64_t len = o.size();
			fan::io::file::write(f, (uint8_t*)&len, sizeof(len), 1);
			fan::io::file::write(f, (uint8_t*)o.data(), len, 1);
		}
		else if constexpr (tmpl::is_std_vector<T>::value) {
			uint64_t len = o.size();
			fan::io::file::write(f, (uint8_t*)&len, sizeof(len), 1);
			fan::io::file::write(f, (uint8_t*)o.data(), len * sizeof(typename T::value_type), 1);
		}
		else {
			fan::io::file::write(f, (uint8_t*)&o, sizeof(o), 1);
		}
	}
}