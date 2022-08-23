#include <exception>
#include <iostream>

int main() {
  try {
    int x;
    throw std::runtime_error("a");
  }
  catch (...) {

  }
}