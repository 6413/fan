#include <fan/pch.h>

struct Assignment {
  std::string type;
  std::string name;
  std::string value;
};

std::string trim(const std::string& str) {
  std::string s = str;
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
    return !std::isspace(ch);
    }));
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
    return !std::isspace(ch);
    }).base(), s.end());
  return s;
}

void set_float(const std::string& name, float value) {
  std::cout << "Setting float: " << name << " = " << value << std::endl;
}

void set_vec2(const std::string& name, float x, float y) {
  std::cout << "Setting vec2: " << name << " = (" << x << ", " << y << ")" << std::endl;
}

void set_struct(const std::string& struct_str) {
  std::vector<Assignment> assignments;
  std::istringstream iss(struct_str);
  std::string line;

  while (std::getline(iss, line)) {
    // Trim leading and trailing whitespace
    line = trim(line);

    // Skip empty lines
    if (line.empty())
      continue;

    // Find the position of the equals sign
    size_t equal_pos = line.find('=');

    // Check if the equals sign is found
    if (equal_pos == std::string::npos) {
      std::cerr << "Error: Expected '=', found end of line." << std::endl;
      continue;
    }

    // Extract type, name, and value from the line
    std::string type = trim(line.substr(0, equal_pos));
    std::string name = trim(line.substr(equal_pos + 1));

    // Check if name ends with a semicolon
    if (name.back() != ';') {
      std::cerr << "Error: Expected ';', found '" << name.back() << "'." << std::endl;
      continue;
    }

    // Remove the semicolon from the name
    name.pop_back();

    // Remove trailing semicolon from the value
    std::string value = trim(name.substr(equal_pos + 1));

    // Construct the assignment
    Assignment assignment = { type, name, value };

    // Add the assignment to the vector
    assignments.push_back(assignment);
  }

  // Process assignments
  for (const auto& assignment : assignments) {
    // Your existing processing logic here...
  }
}

struct command_t {
  fan::function_t<void()> command;
  fan::string description;
};

void connect() {
  fan::print("connect");
}

void disconnect() {
  fan::print("disconnect");
}

std::unordered_map<std::string, command_t> command_tables;

fan::string get_description(const fan::string& cmd) {
  return command_tables[cmd].description;
}

void call_command(fan::string cmd) {
  command_tables[cmd].command();
}

void register_command(
  const std::string& command_name,
  fan::function_t<void()> func,
  const fan::string& description
) {
  auto& command = command_tables[command_name];
  command.command = func;
  command.description = description;
}

int main() {

  //register_command("connect", connect, "connects to server");
  //register_command("disconnect", disconnect, "disconnect from server");

  //call_command("connect");
  //fan::print("description:" + get_description("connect"));
  //return 0;
  loco_t loco;
  
  std::string struct_str = R"(
    float blur_radius = 0.001;
    float bloom_threshold = 0;
    float bloom_softness = 5;
    float bloom_radius = 0.3;
    float bloom_strength = 2;
    vec2 something = vec2(3, 2);
  )";

  std::string current_type;
  std::string current_name;
  std::string current_value;

  std::istringstream f(struct_str);
  std::string line;
  bool finished_cycle = false;
  int token_index = 0;
  while (std::getline(f, line)) {
    std::istringstream f2(line);
    std::string word;
    while (f2 >> word) {
      if (word.find("=") != std::string::npos) {
        continue;
      }
      if (word.find("vec2(") != std::string::npos) {
        std::string word2;
        while (f2 >> word2) {
          if (token_index == 2) {
            word2.pop_back();
            current_value = word + word2;
          }
          token_index++;
          if (token_index > 2) {
            fan::print(current_type, current_name, current_value);
            token_index = 0;
          }
         // fan::print(word + word2);
          //current_value = 
        }
        continue;
      }
     // fan::print(word);
      if (token_index == 0) {
        current_type = word;
      }
      if (token_index == 1) {
        current_name = word;
      }
      if (token_index == 2) {
        word.pop_back();
        current_value = word;
      }
      token_index++;
      if (token_index > 2) {
        fan::print(current_type, current_name, current_value);
        token_index = 0;
      }
    }
  }

  loco.loop([] {

  });

  return 0;
}