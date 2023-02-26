#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/masterpiece.h)


#include <iostream>
#include <stdexcept>
#include <regex>

typedef std::string String;
using namespace std::literals::string_literals;

class Strings
{
public:
  static String TrimStart(const std::string& data)
  {
    String s = data;
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
      return !std::isspace(ch);
      }));
    return s;
  }

  static String TrimEnd(const std::string& data)
  {
    String s = data;
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
      return !std::isspace(ch);
      }).base(),
        s.end());
    return s;
  }

  static String Trim(const std::string& data)
  {
    return TrimEnd(TrimStart(data));
  }

  static String Replace(const String& data, const String& toFind, const String& toReplace)
  {
    String result = data;
    size_t pos = 0;
    while ((pos = result.find(toFind, pos)) != String::npos)
    {
      result.replace(pos, toFind.length(), toReplace);
      pos += toReplace.length();
      pos = result.find(toFind, pos);
    }
    return result;
  }

};

static String Nameof(const String& name)
{
  std::smatch groups;
  String str = Strings::Trim(name);
  if (std::regex_match(str, groups, std::regex(R"(^&?([_a-zA-Z]\w*(->|\.|::))*([_a-zA-Z]\w*)$)")))
  {
    if (groups.size() == 4)
    {
      return groups[3];
    }
  }
  throw std::invalid_argument(Strings::Replace(R"(nameof(#). Invalid identifier "#".)", "#", name));
}

#define nameof(name) Nameof(u8## #name ## s)
#define cnameof(name) Nameof(u8## #name ## s).c_str()

enum TokenType {
  COMMA,
  PERIOD,
  Q_MARK
};

struct MyClass
{
  enum class MyEnum : char {
    AAA = -8,
    BBB = '8',
    CCC = AAA + BBB
  };
};

int main() {
  nameof(COMMA);
}