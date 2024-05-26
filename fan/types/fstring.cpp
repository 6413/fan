#include "fstring.h"

std::string fan::trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

std::vector<std::string> fan::split(const std::string& s) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, ',')) {
        std::istringstream tokenStream2(token);
        std::getline(tokenStream2, token, '=');
        tokens.push_back(fan::trim(token));
    }
    return tokens;
}