#undef loco_assimp
#define sdme_build_struct_arithmetic_type
#include <fan/types/sdme.h>
#include <fan/math/random.h>


struct types_e {
  enum {
    unknown = -1,
    i,
    f,
    d,
    c
  };
};

static constexpr uint32_t max_return_types = 64;

struct type_info_t {
  void* v;
  int t;
  uint64_t len;
};

struct any_ret_t {
  type_info_t ti[max_return_types]{};
};

template <typename T>
int type_to_enum(const T& v) {
  using compare_t = /*std::conditional_t < requires{ typename T::type; }, typename T::type, T > */ T;
  if (std::is_same_v<compare_t, int>) {
    return types_e::i;
  }
  else if (std::is_same_v<compare_t, float>) {
    return types_e::f;
  }
  else if (std::is_same_v<compare_t, double>) {
    return types_e::d;
  }
  else if (std::is_same_v<compare_t, char>) {
    return types_e::c;
  }
  else {
    return types_e::unknown;
  }
}

uint32_t get_type_sizeof(uint32_t i) {
  switch (i) {
  case types_e::i: {
    return sizeof(int);
  }
  case types_e::f: {
    return sizeof(float);
  }
  case types_e::d: {
    return sizeof(double);
  }
  case types_e::c: {
    return sizeof(char);
  }
  default: {
    return 0;
  }
  }
}

void print_type(const any_ret_t& r) {
  for (uint32_t type = 0; type < std::size(r.ti); ++type) {
    if (r.ti[type].len == 0) { // no more types
      break;
    }
    for (uint32_t i = 0; i < r.ti[type].len / get_type_sizeof(r.ti[type].t); ++i) {
      switch (r.ti[type].t) {
      case types_e::i: {
        fan::print(((int*)&r.ti[type].v)[i]);
        break;
      }
      case types_e::f: {
        fan::print(((float*)&r.ti[type].v)[i]);
        break;
      }
      case types_e::d: {
        fan::print(((double*)&r.ti[type].v)[i]);
        break;
      }
      case types_e::c: {
        fan::print((int)(((char*)&r.ti[type].v)[i]));
        break;
      }
      }
    }
  }
}

// d needs to be sdme
any_ret_t return_any_type(const auto& sdme) {
  any_ret_t r;
  for (int i = 0; i < sdme.size(); ++i) {
    sdme.get_value(i, [&]<typename T>(const T & v) {
      r.ti[i].v = *(void**)&v.v;
      r.ti[i].len = sizeof(T);
      r.ti[i].t = type_to_enum(v.v);
    });
  }
  return r;
}

#define create_any_type_begin [] { sdme_create_struct(temp_t)
#define create_any_type_end sdme_internal_end_manual__ }v; return v;}()

any_ret_t f() {
  switch (fan::random::value_i64(0, 1)) {
  case 0: {
    // not necessary to be sdme but just shorter and automated code
    return return_any_type(
      create_any_type_begin
      __sdme2(int, i) = { 1 };
    __sdme2(float, f) = { 2 };
    create_any_type_end
      );
  }
  case 1: {
    // not necessary to be sdme but just shorter and automated code
    return return_any_type(
      create_any_type_begin
      __sdme2(double, d) = { 3 };
    __sdme2(char, c) = { 4 };
    create_any_type_end
      );
  }
  }
  return {};
}

#pragma pack(pop)

int main() {
  for (int i = 0; i < 5; ++i) {
    print_type(f());
    fan::print("");
  }
  return 0;
}