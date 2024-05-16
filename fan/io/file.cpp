#include "file.h"

bool fan::io::file::exists(const std::string& name) {
  std::ifstream file(name.c_str());
  return file.good();
}

bool fan::io::file::rename(const std::string& from, const std::string& to) {
  return std::rename(from.c_str(), to.c_str());
}

bool fan::io::file::close(file_t* f) {
  int ret = fclose(f);
  #if fan_debug >= fan_debug_low
  if (ret != 0) {
    fan::print_warning("failed to close file stream");
    return 1;
  }
  #endif
  return 0;
}

bool fan::io::file::open(file_t** f, const char* path, const properties_t& p) {
  *f = fopen(path, p.mode);
  if (f == nullptr) {
    fan::print_warning(std::string("failed to open file:") + path);
    close(*f);
    return 1;
  }
  return 0;
}

bool fan::io::file::write(file_t* f, void* data, uint64_t size, uint64_t elements) {
  uint64_t ret = fwrite(data, size, elements, f);
  #if fan_debug >= fan_debug_low
  if (ret != elements && size != 0) {
    fan::print_warning("failed to write from file stream");
    return 1;
  }
  #endif
  return 0;
}

bool fan::io::file::read(file_t* f, void* data, uint64_t size, uint64_t elements) {
  uint64_t ret = fread(data, size, elements, f);
  #if fan_debug >= fan_debug_low
  if (ret != elements && size != 0) {
    fan::print_warning("failed to read from file stream");
    return 1;
  }
  #endif
  return 0;
}

uint64_t fan::io::file::file_size(const std::string& filename)
{
  std::ifstream f(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
  return f.tellg();
}

bool fan::io::file::write(std::string path, const std::string& data, decltype(std::ios_base::binary | std::ios_base::app) mode) {
  std::ofstream ofile(path.c_str(), mode);
  if (ofile.fail()) {
    fan::print_warning("failed to write to:" + path);
    return 0;
  }
  ofile.write(data.c_str(), data.size());
  return 1;
}

std::vector<std::string> fan::io::file::read_line(const std::string& path) {

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

bool fan::io::file::read(const std::string& path, std::string* str) {

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

std::string fan::io::file::extract_variable_type(const std::string& string_data, const std::string& var_name) {
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

int fan::io::file::get_string_valuei(const std::string& str, const std::string& find, std::size_t offset) {

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

fan::io::file::str_int_t fan::io::file::get_string_valuei_n(const std::string& str, std::size_t offset) {
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

fan::io::file::str_vec2i_t fan::io::file::get_string_valuevec2i_n(const std::string& str, std::size_t offset) {

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

fan::io::file::fstream::fstream(const std::string& path) {
  file_name = path;
  open(path);
}

fan::io::file::fstream::fstream(const std::string& path, std::string* str) {
  file_name = path;
  open(path);
  read(str);
}

bool fan::io::file::fstream::open(const std::string& path) {
  file_name = path;
  auto flags = std::ios::in | std::ios::out | std::ios::binary;

  if (!exists(path)) {
    flags |= std::ios::trunc;
  }

  file = std::fstream(path, flags);
  return !file.good();
}

bool fan::io::file::fstream::read(const std::string& path, std::string* str) {
  file_name = path;
  open(path);
  return read(str);
}

bool fan::io::file::fstream::read(std::string* str) {
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

bool fan::io::file::fstream::write(std::string* str) {
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
