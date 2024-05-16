#include <fan/pch.h>


int main() {
  auto s = std::to_array({"test", "something"})[fan::random::value_i64(0, 1)];
  custom_switch(s) {
    custom_case "test": {
      fan::print("hi");
      break;
    }
    default: {
      fan::print("no");
      break;
    }
  }
}