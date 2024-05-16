#include <fan/pch.h>
#include <fan/io/json.h>

int main() {



  fan::string str;
  fan::io::file::read("json/test.json", &str);
  fan::json data = fan::json::parse(str);
  fan::print(data["quiz"]["sport"]);
  //json
}