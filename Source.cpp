#include <fan/types/types.h>

#include <iostream>

//#define bll_init

struct nr_t {
  using nr_type_t = uint32_t;

  void sic() {
    nr = (nr_type_t)-1;
  }
  static nr_t sic(nr_t& nr) {
    nr.nr = (nr_type_t)-1;
    return nr;
  }

  #ifdef bll_init
  nr_t() {
    sic();
  }
  #else
  nr_t() = default;
  nr_t(bool x) {
    //if 
  }
  #endif

  nr_type_t nr;
};

int main() {
  nr_t nr{};
  return nr.nr;
}