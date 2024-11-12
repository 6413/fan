#include <fan/pch.h>
#include "fmt.h"

// std::quoted
#include <iomanip>

std::vector<std::string> fan::split(std::string str, std::string token) {
  std::vector<std::string>result;
  while (str.size()) {
    std::size_t index = str.find(token);
    if (index != std::string::npos) {
      result.push_back(str.substr(0, index));
      str = str.substr(index + token.size());
      if (str.size() == 0)result.push_back(str);
    }
    else {
      result.push_back(str);
      str = "";
    }
  }
  return result;
}

std::vector<std::string> fan::split_quoted(const std::string& input) {
  std::vector<std::string> args;
  std::istringstream stream(input);
  std::string arg;

  while (stream >> std::quoted(arg)) {
    args.push_back(arg);
  }

  return args;
}