struct a_t {
  struct not_yet_known_t;
  struct b_t {
    void f(not_yet_known_t* m) { _b_t_f(this, m); }
  };
  struct c_t {
    b_t b;
    //...
  };
  struct not_yet_known_t {
    c_t c;
    //...
  };


};
static void _b_t_f(a_t::b_t* b, a_t::not_yet_known_t* m) {
  // code comes here
}
int main() {
  a_t::not_yet_known_t m;
  a_t::b_t b;
  b.f(&m);
}