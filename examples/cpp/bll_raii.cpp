#include <fan/utility.h>
#include <functional>

import fan;

#include <fan/fan_bll_preset.h>
#define BLL_set_CPP_ConstructDestruct 1
#define BLL_set_Language 1
#define BLL_set_SafeNext 1
#define BLL_set_prefix update_callback
#define BLL_set_Link 1
#define BLL_set_NodeDataType std::function<void(update_callback_t*)>
#include <BLL/BLL.h>

int main(){
  struct test_t {
    update_callback_t bll;

    using h = fan::raii_nr_t<update_callback_t::nr_t, test_t, update_callback_t*>;
    h handle;

    void f(){
      handle = fan::add_bll_raii_struct_cb<test_t, update_callback_t>(this, &test_t::bll,
        [](update_callback_t*){

        }
      );
    }
  } test;

  update_callback_t bll;

  auto handle = fan::add_bll_raii_cb(bll, [](update_callback_t* c) {
      fan::print("called from loop", c);
    }
  );
  auto handle2 = fan::add_bll_raii_cb(bll, [](update_callback_t* c) {
    fan::print("called from loop", c);
    }
  );

  for (auto& i : bll) {
    i(&bll);
  }
}
