#include <iostream>
#include <functional>
#include <cstdio>
#include <cstring>

struct sub_t {
  int z = 5;
  sub_t(){
    printf("constructor came\n");
  }
  ~sub_t() {
    printf("destructor came\n");
  }
};

#define OFFSETLESS(ptr_m, t_m, d_m) \
    ((t_m *)((uint8_t *)(ptr_m) - offsetof(t_m, d_m)))

#define fan_make_namespace_function(header, data) \
 header { \
  char buffer[sizeof(top_t) + sizeof(something_t)]; \
  struct __ : top_t, something_t { \
    void operator()() { \
      data\
    } \
  };  \
  memcpy((top_t*)buffer, &operator top_t & (), sizeof(top_t)); \
  memcpy(((uint8_t*)(top_t*)buffer) + sizeof(top_t), this, sizeof(something_t)); \
  (*(__*)buffer)(); \
  memcpy(&operator top_t & (), (top_t*)buffer, sizeof(top_t)); \
  memcpy(this, ((uint8_t*)(top_t*)buffer) + sizeof(top_t), sizeof(something_t)); \
} 
#define fan_make_namespace_inside_class(name, top, whattodo) \
   struct name##_t { \
      using top_t = top; \
      using something_t = name##_t; \
      operator top_t&() { \
        return *OFFSETLESS(this, top_t, name); \
      } \
      whattodo \
    }name; 


struct st_t{
  sub_t a;
  fan_make_namespace_inside_class(b, st_t,
    sub_t sub;
    fan_make_namespace_function(
      void f(),
      sub.z = 10;
      a.z = 5;
    );
    fan_make_namespace_function(
      void f2(),
      sub.z = 10;
      a.z = 5;
    );
  );

  void f() {
    b.f();
  }
};

int main(int argc, char **argv){
  new(sizeof(5));
  for(int i = 0; i < 1; i++){
    printf("gonna new\n");
    auto ptr = new st_t;
    printf("newed\n");
    ptr->f();
    printf("called0\n");
    ptr->f();
    printf("called1\n");
    delete ptr;
    printf("deleted\n");
  }

  printf("for ended\n");
  return 0;
}