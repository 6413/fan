module;

#include <fan/utility.h>

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <source_location>
#include <filesystem>

#if defined(fan_platform_windows)
  #include <Windows.h>
#elif defined(fan_platform_unix)
  #include <unistd.h>
  #include <limits.h>
#endif

export module fan.io.file;

import fan.print;
import fan.utility;

export namespace fan {
	namespace io {
		namespace file {

      template<typename T>
      concept path_t = std::same_as<T, std::filesystem::path>;

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

			uint64_t size(const std::string& filename) {
				std::ifstream f(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
				return f.tellg();
			}

      using fs_mode = decltype(std::ios_base::binary | std::ios_base::app);

			bool write(
				std::string path,
				const std::string& data,
				fs_mode mode = std::ios_base::binary | std::ios_base::app
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
				fs_mode mode = std::ios_base::binary | std::ios_base::app
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

      // writes only if file does not exist
      bool try_write(
        std::string path,
				const std::string& data,
        fs_mode mode = std::ios_base::binary | std::ios_base::app
      ) {
        if (fan::io::file::exists(path)) {
          return false;
        }
        return write(path, data, mode);
      }

      std::string get_exe_path() {
#if defined(fan_platform_windows)
        char buffer[MAX_PATH];
        GetModuleFileNameA(NULL, buffer, MAX_PATH);
        return std::filesystem::path(buffer).parent_path().string() + "/";
#elif defined(fan_platform_unix)
        char buffer[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", buffer, PATH_MAX);
        return std::filesystem::path(std::string(buffer, count)).parent_path().string() + "/";
        //#elif defined(__APPLE__)
        //      char buffer[PATH_MAX];
        //      uint32_t size = sizeof(buffer);
        //      _NSGetExecutablePath(buffer, &size);
        //      return std::filesystem::path(buffer).parent_path().string() + "/";
#endif
        fan::throw_error("not implemented");
      }

      // searches primarily from working directory and 
      // returns relative path, 
      // then checks exe's directory, 
      // then all the way recursively from path x till exe's path
      // else returns empty path
      std::filesystem::path find_relative_path(const std::string& file_path, const std::source_location& location = std::source_location::current()) {
        namespace fs = std::filesystem;
        if (file_path.empty()) {
          return {};
        }

        std::error_code ec;
        fs::path current_dir = fs::current_path(ec);
        if (ec) return {};

        if (fs::is_regular_file(file_path, ec) && !ec) {
          return file_path;
        }
        fs::path candidate = fs::path(location.file_name()).parent_path() / file_path;
        if (fs::is_regular_file(candidate, ec) && !ec) {
          fs::path relative = fs::relative(candidate, current_dir, ec);
          return ec ? candidate : relative;
        }

        candidate = current_dir / file_path;
        if (fs::is_regular_file(candidate, ec) && !ec) {
          return file_path;
        }

        fs::path exe_dir = get_exe_path();
        if (!exe_dir.empty() && exe_dir != current_dir) {
          candidate = exe_dir / file_path;
          if (fs::is_regular_file(candidate, ec) && !ec) {
            fs::path relative = fs::relative(candidate, current_dir, ec);
            return ec ? candidate : relative;
          }
        }

        if (!exe_dir.empty()) {
          fs::path search_path = current_dir;

          while (search_path.has_parent_path() && search_path != search_path.parent_path()) {
            search_path = search_path.parent_path();

            candidate = search_path / file_path;
            if (fs::is_regular_file(candidate, ec) && !ec) {
              fs::path relative = fs::relative(candidate, current_dir, ec);
              return ec ? candidate : relative;
            }

            if (fs::equivalent(search_path, exe_dir, ec) && !ec) {
              break;
            }
          }
        }

        return {};
      }
			bool read(const std::string& path, std::string* str) {

				std::ifstream file(path.c_str(), std::ifstream::ate | std::ifstream::binary);
				if (file.fail()) {
#if fan_debug >= fan_debug_insane
					fan::print_warning_no_space("path does not exist:" + path);
#endif
					return 1;
				}
				str->resize(file.tellg());
				file.seekg(0, std::ios::beg);
				file.read(&(*str)[0], str->size());
				file.close();
				return 0;
			}
      
      bool read(const path_t auto & path, std::string* str) {
        std::ifstream file(path, std::ifstream::ate | std::ifstream::binary);
        if (file.fail()) {
#if fan_debug >= fan_debug_insane
          fan::print_warning_no_space("path does not exist:" + path.string());
#endif
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
#if fan_debug >= fan_debug_insane
					fan::print_warning_no_space("path does not exist:" + path);
#endif
					return 1;
				}
				str->resize(length);
				file.seekg(0, std::ios::beg);
				file.read(&(*str)[0], length);
				file.close();
				return 0;
			}
      bool read(const path_t auto& path, std::string* str, std::size_t length) {
        std::ifstream file(path, std::ifstream::binary);
        if (file.fail()) {
#if fan_debug >= fan_debug_insane
          fan::print_warning_no_space("path does not exist:" + path.string());
#endif
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