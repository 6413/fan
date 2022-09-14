#include <fan/types/types.h>

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

int main() {
  

  //struct main_menu_data_t {
  //  f32_t a;
  //};
  //stage_common_t main_menu{
  //  .open = [menu_data = main_menu_data_t 
  //};

  //struct sortie_data_t {
  //  int b;
  //};

  //stage_common_t sortie{
  //  .open = [] {

  //  }
  //};

}