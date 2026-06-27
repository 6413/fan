export module fan.io.prompt;

import std;
import fan.fmt;

export namespace fan::io {
  bool ask_yes_no(const std::string_view question, bool default_yes = false) {
    fan::printf("{} [{}]: ", question, default_yes ? "Y/n" : "y/N");
    std::string response;
    std::getline(std::cin, response);
    return response.empty() ? default_yes : (response[0] == 'y' || response[0] == 'Y');
  }

  bool ask_override(const std::string_view target) {
    return ask_yes_no(std::format("Override {}?", target));
  }

  std::string ask_string(const std::string_view question) {
    fan::print("{}: ", question);
    std::string response;
    std::getline(std::cin, response);
    return response;
  }
}