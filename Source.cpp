#include <string>
#include <fan/types/types.h>

#include <stddef.h>

int main() {
  std::stringstream ss[2];
  std::string test = "test";
  ss[0] << test << '\n';
}