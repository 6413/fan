import fan;
import std;
import fan.rjson;

int main() {
  fan::rjson_t json;
  json["grapes"] = 8;
  for (int i = 0; i < 10; ++i) {
    json["apples" + std::to_string(i)] = i;
  }
  fan::io::file::write("fruits.json", fan::rjson_dump(json, 2), std::ios::binary);
  std::string str = fan::io::file::read("fruits.json");
  fan::print(fan::rjson_parse(str));
}