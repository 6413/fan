#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/masterpiece.h)

#include <iostream>
#include <string>
#include <regex>
#include <tuple>
#include <utility>

template<typename T>
constexpr void assignArg(T& arg, const std::string& value) {
    if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
        arg = value.substr(1, value.size() - 2);
    }
    else if constexpr (std::is_integral_v<std::decay_t<T>>) {
        arg = std::stoi(value);
    }
}

template<typename... Args>
constexpr auto parseArgs(std::string_view str, std::tuple<Args...>& args) {
    size_t start = 0;
    while (start < str.size() && str[start] != '(') {
        ++start;
    }
    ++start;
    size_t end = start;
    while (end < str.size() && str[end] != ')') {
        ++end;
    }
    std::string argsStr(str.data() + start, end - start);

    std::regex argRe(R"("[^"]*"|\d+)");
    std::sregex_iterator it(argsStr.begin(), argsStr.end(), argRe);
    std::sregex_iterator endIt;
    size_t index = 0;
    while (it != endIt) {
        std::apply([&](auto&... arg) {
            assignArg((index == 0 ? arg : (index == 1 ? arg : (void)arg))..., it->str());
        }, args);
        ++it;
        ++index;
    }
}

int main() {
    constexpr const char* input = "f(5, \"there was a durum long time ago\")";
    constexpr size_t pos = input - std::find(input, input + sizeof(input), '(');
    constexpr std::string_view command(input, pos);

    std::tuple<int, std::string> args;
    parseArgs(input, args);

    return 0;
}