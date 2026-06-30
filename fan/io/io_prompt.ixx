export module fan.io.prompt;

import std;

export namespace fan::io {
  bool ask_yes_no(const std::string_view question, bool default_yes = false) {
    std::cout << question << " [" << (default_yes ? "Y/n" : "y/N") << "]: " << std::flush;
    std::string response;
    std::getline(std::cin, response);
    return response.empty() ? default_yes : (response[0] == 'y' || response[0] == 'Y');
  }

  bool ask_override(const std::string_view target) {
    std::string question;
    question.reserve(10 + target.size());
    question += "Override ";
    question += target;
    question += "?";
    return ask_yes_no(question);
  }

  std::string ask_string(const std::string_view question) {
    std::cout << question << ": " << std::flush;
    std::string response;
    std::getline(std::cin, response);
    return response;
  }
}