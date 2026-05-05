module;

export module fan.reflection;

import std;
import fan.print;
import fan.types.color;

#if defined(FAN_REFLECTION)
  #include "gcc_private.h"
#endif

export namespace fan::refl {
  template <typename T>
  consteval auto members() {
    return std::define_static_array(std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked()));
  }

  template <typename T>
  consteval std::size_t member_count() {
    return members<T>().size();
  }

  template <typename T, typename F>
  constexpr void for_each_member(F&& func) {
    template for (constexpr auto m : members<T>()) {
      func.template operator()<m>();
    }
  }

  template <typename T, typename F>
  consteval auto member_at(std::size_t i, F&& func) {
    std::size_t cur = 0;
    template for (constexpr auto m : members<T>()) {
      if (cur++ == i) return func.template operator()<m>();
    }
  }

  template <typename T, typename F>
  constexpr void member_visit(std::size_t i, F&& func) {
    std::size_t cur = 0;
    for_each_member<T>([&]<std::meta::info m> {
      if (cur++ == i) func.template operator()<m>();
    });
  }

  template <typename T>
  constexpr auto& at_index(auto&& obj, std::size_t i) {
    using obj_t  = std::remove_reference_t<decltype(obj)>;
    using cast_t = std::conditional_t<std::is_const_v<obj_t>, const T, T>;
    using M0     = decltype(std::declval<T>().*(&[:members<T>()[0]:]));
    void* ptr    = nullptr;
    member_visit<T>(i, [&]<std::meta::info m> { ptr = std::addressof(static_cast<cast_t&>(obj).*(&[:m:])); });
    return *static_cast<std::remove_reference_t<M0>*>(ptr);
  }

  template <typename T, typename F>
  constexpr void iterate_members(auto&& obj, F&& func) {
    for_each_member<T>([&]<std::meta::info m> { func(std::meta::identifier_of(m), obj.[:m:]); });
  }

  template <typename T, typename F>
  constexpr void apply_at(auto&& obj, std::size_t i, F&& func) {
    member_visit<T>(i, [&]<std::meta::info m> { func(obj.[:m:]); });
  }

  template <std::meta::info M>
  consteval std::meta::info annotation_type() {
    template for (constexpr auto ann : std::define_static_array(std::meta::annotations_of(M))) {
      return std::meta::remove_cv(std::meta::type_of(ann));
    }
    return std::meta::info{};
  }

  template <typename TAttr, std::meta::info M>
  consteval TAttr annotation() {
    template for (constexpr auto ann : std::define_static_array(std::meta::annotations_of(M))) {
      if constexpr (std::meta::remove_cv(std::meta::type_of(ann)) == ^^TAttr) { return std::meta::extract<TAttr>(ann); }
    }
    return TAttr{};
  }

  template <typename T>
  consteval std::string_view member_name(std::size_t i) {
    return member_at<T>(i, []<std::meta::info m> { return std::meta::identifier_of(m); });
  }

  template <typename T>
  consteval std::string_view type_name() {
    return std::meta::identifier_of(^^T);
  }

  consteval std::string_view type_name(std::meta::info t) {
    return std::meta::identifier_of(t);
  }

  template <typename T>
  consteval std::size_t offset_of(std::size_t i) {
    return member_at<T>(i, []<std::meta::info m> { return std::meta::offset_of(m); });
  }

  template <typename T>
  consteval std::size_t index_of(std::string_view name) {
    std::size_t cur = 0;
    template for (constexpr auto m : members<T>()) {
      if (std::meta::identifier_of(m) == name) return cur;
      ++cur;
    }
    return ~std::size_t{0};
  }
}  // namespace fan::refl

export namespace fan {
  template <typename T>
  std::string format_reflect(const T& obj, int indent_level = 0) {
    if constexpr (requires { std::declval<std::ostream&>() << obj; }) {
      std::ostringstream val;
      val << obj;
      std::string t{std::meta::display_string_of(^^T)};
      if (t.starts_with("member_t {aka ")) t = t.substr(14, t.size() - 15);
      return fan::paint(fan::colors::white, val.str()) + fan::paint(fan::colors::gray, " [") +
             fan::paint(fan::colors::teal, t) + fan::paint(fan::colors::gray, "]");
    }
    else {
      std::string ind(indent_level * 2, ' ');
      std::string out = fan::paint(fan::colors::cyan, std::string{std::meta::display_string_of(^^T)}) +
                        fan::paint(fan::colors::gray, " {\n");
      fan::refl::iterate_members<T>(obj, [&](auto name, auto& value) {
        out += fan::paint(fan::colors::amber, ind + "  ." + std::string{name} + " = ") +
               format_reflect(value, indent_level + 1) + "\n";
      });
      return out + fan::paint(fan::colors::gray, ind + "}");
    }
  }

  template <typename... T>
  void print_reflect(const T&... args) {
    (fan::print_color_raw(fan::colors::white, format_reflect(args) + "\n"), ...);
  }
}  // namespace fan