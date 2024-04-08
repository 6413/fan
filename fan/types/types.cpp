#include "types.h"

void fan::throw_error_impl() {
  #ifdef fan_compiler_msvc
  system("pause");
  #endif
  #if __cpp_exceptions
  throw std::runtime_error("");
  #endif
}

void fan::assert_test(bool test) {
  if (!test) {
    fan::throw_error("assert failed");
  }
}

uint8_t __clz32(uint32_t p0)
{
  #if defined(__GNUC__)
  return __builtin_clz(p0);
  #elif defined(_MSC_VER)
  DWORD trailing_zero = 0;
  if (_BitScanReverse(&trailing_zero, p0)) {
    return uint8_t((DWORD)31 - trailing_zero);
  }
  else {
    return 0;
  }
  #else
  #error ?
  #endif
}

uint8_t __clz64(uint64_t p0) {
  #if defined(__GNUC__)
  return __builtin_clzll(p0);
  #elif defined(_MSC_VER)
  DWORD trailing_zero = 0;
  if (_BitScanReverse64(&trailing_zero, p0)) {
    return uint8_t((DWORD)63 - trailing_zero);
  }
  else {
    return 0;
  }
  #else
  #error ?
  #endif
}

std::vector<std::string> fan::split(std::string str, std::string token) {
  std::vector<std::string>result;
  while (str.size()) {
    int index = str.find(token);
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