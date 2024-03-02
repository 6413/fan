#include fan_pch

int main(int argc, char** argv) {
  if (argc != 2) {
    fan::print("usage .exe filename");
    return 1;
  }

  fan::string code;
  if (fan::io::file::read(argv[1], &code)) {
    return 1;
  }

 /* code.insert(0, R"(
#define custom_switch(x) switch(fan::get_hash(std::string_view(x)))
#define custom_case(x) case fan::get_hash(std::string_view(x))
)");

  uint64_t beg = 0;
  std::string_view keyword = "custom_case ";
  auto src = code.find(keyword, beg);
  if (src != std::string::npos) {
    auto dst = code.find(":", src + 1 + beg);
    if (dst != std::string::npos) {
      code.replace(src, keyword.size(), "custom_case(");
      code.replace(dst, 1, "):");
    }
  }*/

  std::string_view str = "print_struct(";
  auto src = code.find(str);
  if (src != std::string::npos) {
    auto dst = code.find(");", src);
    if (dst != std::string::npos) {
      auto struct_to_find = code.substr(src + str.size(), dst - (src + str.size()));
      auto struct_src = code.find(struct_to_find + " {");
      auto struct_dst = code.find("}", struct_src);
      auto struct_parsed = code.substr(struct_src, struct_dst);
      std::string line;
      std::istringstream iss(struct_parsed);
      std::vector<std::string> var_names;
      while (std::getline(iss, line)) {
        std::istringstream liss(line);
        std::string s0, s1, s2, s3;
        while (liss >> s0 >> s1 >> s2 >> s3) {
          var_names.push_back(s1);
        }
      }
      auto arr = std::to_array({ 1, 2 });
      for (auto& [idx, value] : std::views::enumerate(arr)) {

      }
      std::string final_string;
      for (auto& i : var_names) {
        final_string += fan::format("fan::print_no_space(\"{}:\", {}().{});\n", i, struct_to_find, i);
      }
      code.replace(src, dst - src + 2, final_string);
    }
  }

  fan::io::file::write("temp.cpp", code, std::ios_base::binary);
  system("make MAIN=temp.cpp");
}