#include <fan/pch.h>


int main() {
  //system("make -f make_pch");
  //"C:\Program Files\Microsoft Visual Studio\2000\VC\Tools\MSVC\14.37.32705\bin\Hostx64\x64\cl.exe"
  // assumes pch is built
  fan::string copy_str = "copy \"a.exe\" \"build/here.exe\"";
  fan::string make_str = "make -f build_projects";
  //
  //system();
  fan::string prev_copy = "here";
  int x = 0;
  struct ret_t {
    fan::string name;
    std::size_t beg;
    std::size_t size;
  };
  auto extract_file_name = [](const fan::string& path) -> ret_t {
    auto found = path.find(".cpp");
    if (found == fan::string::npos) {
      return {};
    }
    //found += strlen(".cpp");
    fan::string file_name;
    std::size_t off = path.rfind("/", found);
    off += 1;
    file_name = path.substr(off, found - off);
    ret_t ret;
    ret.name = file_name;
    ret.beg = off;
    ret.size = found - off;
    return ret;
  };

  fan::io::iterate_directory("examples/graphics", [&](const fan::string path) {

    fan::string file;
    fan::io::file::read("build_projects", &file);
    fan::string file_name = extract_file_name(path).name;
    if (file_name.empty()) {
      return;
    }
    auto file_make_name = extract_file_name(file);
    if (file_make_name.name.empty()) {
      file.insert(file.begin() + file_make_name.beg, file_name.begin(), file_name.end());
    }
    else {
      file.replace_all(file_make_name.name, file_name);
    }
    fan::io::file::write("build_projects", file, std::ios_base::binary);
    system(make_str.c_str());
    fan::string temp = file_name;
    copy_str.replace_all(prev_copy, temp);
    system(copy_str.c_str());
    prev_copy = file_name;
    system("del a.exe");
  });
}