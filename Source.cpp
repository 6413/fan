#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH,fan/types/types.h)

#include _FAN_PATH(trees/quad_tree.h)

//template <class T, std::size_t... I>
//  requires std::copy_constructible<T>
//auto enable_if_constructible_helper(std::index_sequence<I...>){ T x{lref_constructor_t{I}...}; }
//
//template <class T, std::size_t N>
//  requires requires {
//  enable_if_constructible_helper<T>(std::make_index_sequence<N>());
//}
//using enable_if_constructible_helper_t = std::size_t;

//#include <type_traits>
//#include <concepts>
//#include <utility>
//


static constexpr std::size_t v = fan::impl::member_count<fan::trees::split_tree_t>();

int main() {
  //luokka{impl::universal_type_t{}, impl::universal_type_t{}, impl::universal_type_t{}};
  fan::print(v);
}