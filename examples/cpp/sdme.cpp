#include <fan/types/sdme.h>

struct data0_t {
  float v;
};

struct data1_t {
  double v;
};

struct data2_t {
  int v;
};

struct data3_t {
  char v;
};


sdme_create_struct(sdme_t)
__sdme2(data0_t, a);
__sdme2(data1_t, b);
sdme_internal_end__

sdme_create_struct(sdme2_t)
__sdme2(data2_t, a);
__sdme2(data3_t, b);
sdme_internal_end__


#pragma pack(pop)

#ifndef VERIFY_SDME
#define VERIFY_SDME 1
#endif

#if VERIFY_SDME
#include <cassert>
#include <stdio.h>
#include <typeinfo>

template<typename T>
void verify_sdme(T& s) {
  assert(s.b.v == (&s.b)->v);
  assert(s.template AN<1>() == 4);
}
#endif

template<class T, std::size_t N>
constexpr std::size_t size(const T(&array)[N]) noexcept
{
  return N;
}

int main() {
  {
    static sdme_t s{ };
    static auto offsetof = s.AN<1>();
    static auto offsetof2 = s.AN<&sdme_t::b>();
    static auto* address = &s.b;
    static auto len = s.GetMemberAmount();

    int runtime_index = 1;
    s.get_value(runtime_index, []<typename T>(T & v) {
      static_assert(!std::is_same_v<T, int>);
#if VERIFY_SDME == 1
      printf("%s\n", typeid(v).name());
#endif
    });

#if VERIFY_SDME
    assert(s.beg_off == 0);
    verify_sdme(s);
    assert(s.template AN<&sdme_t::b>() == s.template AN<1>());
    assert(s.template AN<1>() == s.b.otf);
    printf("%lu %llu %lf %s %p %d %d %d %lf\n",
      s.beg_off, len, *address, typeid(decltype(*address)).name(),
      address, decltype(s.b)::otf, offsetof, offsetof2, s.b);
#endif
  }
   {
     sdme2_t s{ .a{}, .b{5} };
     static auto offsetof = s.AN<1>();
     static auto offsetof2 = s.AN<&sdme_t::b>();
     static auto* address = &s.b;
     static auto len = s.GetMemberAmount();

 #if VERIFY_SDME
     assert(s.beg_len == 2);
     assert(s.beg_off == sizeof(sdme_t));
     verify_sdme(s);
     assert(s.template AN<&sdme2_t::b>() == s.template AN<1>());
     assert(s.template AN<1>() == s.b.otf);
     printf("%lu %llu %lf %s %p %d %d %d %lf\n",
       s.beg_off, len, *address, typeid(decltype(*address)).name(),
       address, decltype(s.b)::otf, offsetof, offsetof2, s.b);
 #endif
   }
}