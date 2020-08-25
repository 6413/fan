#include <iostream>
#include <string>
#include <fstream>
#include <vector>

struct File {
	File(std::string file_name) : name(file_name) {}
	File(const char* file_name) : name(file_name) {}

	bool read() {
		std::ifstream file(name.c_str(), std::ifstream::ate | std::ifstream::binary);
		if (!file.good()) {
			return 0;
		}
		data.resize(file.tellg());
		data.resize(file.tellg());
		file.seekg(0, std::ios::beg);
		file.read(&data[0], data.size());
		file.close();
		return 1;
	}

	template <typename T>
	bool read(std::vector<T>& vector) {
		std::ifstream file(name.c_str(), std::ifstream::ate | std::ifstream::binary);
		if (!file.good()) {
			return 0;
		}
		uint64_t size = file.tellg();
		vector.resize(size / sizeof(T));
		file.seekg(0, std::ios::beg);
		file.read(reinterpret_cast<char*>(&vector[0]), size);
		file.close();
		return 1;
	}

	static inline void write(
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
	static inline bool file_exists(const std::string& name) {
		std::ifstream file(name);
		return file.good();
	}
	std::string data;
	std::string name;
};