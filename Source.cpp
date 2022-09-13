#include <fan/types/types.h>

#include <Windows.h>

void function(auto&&...args) {
  (fan::print(sizeof(args)), ...);
}

int main() {
  function(5, 5.0, 1.f);
}