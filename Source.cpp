#include <exception>
#include <iostream>

#define x 5
#define y #x
#define x 6


int main() {

  try {
    int x;
    throw std::runtime_error("a");
  }
  catch (...) {

  }
}