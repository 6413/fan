#include fan_pch


int main() {
  //system("make -f make_pch");
  //"C:\Program Files\Microsoft Visual Studio\2000\VC\Tools\MSVC\14.37.32705\bin\Hostx64\x64\cl.exe"
  // assumes pch is built
  fan::string copy_str = "copy \"a.exe\" \"test/here.exe\"";
  fan::string make_str = "make -f build_projects";
  //
  //system();
  fan::string prev_copy = "here";
  int x = 0;
  auto extract_file_name = [](const fan::string& path) -> fan::string {
    auto found = path.find(".cpp");
    if (found == fan::string::npos) {
      return "";
    }
    //found += strlen(".cpp");
    fan::string file_name;
    std::size_t off = path.rfind("/", found);
    off += 1;
    file_name = path.substr(off, found - off);
    return file_name;
    };

  fan::io::iterate_directory("examples/graphics", [&](const fan::string path) {

    fan::string file;
    fan::io::file::read("build_projects", &file);
    fan::string file_name = extract_file_name(path);
    fan::string file_make_name = extract_file_name(file);
    file.replace_all(file_make_name, file_name);
    fan::io::file::write("build_projects", file, std::ios_base::binary);
    system(make_str.c_str());
    fan::string temp = file_name;
    copy_str.replace_all(prev_copy, temp);
    system(copy_str.c_str());
    prev_copy = file_name;
    system("del a.exe");
    //system("make -f build_projects");
  });
}