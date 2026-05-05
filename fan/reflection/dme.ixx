module;

export module fan.types.dme;

import std;
import fan.reflection;

export namespace fan {
  template <typename T>
  struct dme_t {
    consteval auto size() { return fan::refl::member_count<T>(); }
    constexpr auto& operator[](this auto&& self, std::size_t i) { return fan::refl::at_index<T>(self, i); }
    consteval std::size_t operator[](std::string_view name) const { return fan::refl::index_of<T>(name); }
    constexpr auto begin() { return &(*this)[0]; }
    constexpr auto end()   { return &(*this)[0] + size(); }
    constexpr std::string_view name(std::size_t i) const { return fan::refl::member_name<T>(i); }
    void print() { fan::print_reflect(static_cast<T&>(*this)); }
    consteval std::meta::info attr_type(std::size_t i) const {
      return fan::refl::member_at<T>(i, []<std::meta::info m>{ return fan::refl::annotation_type<m>(); });
    }  
    template <auto I>
    consteval auto attr() const {
      constexpr auto m = fan::refl::members<T>()[I];
      return fan::refl::annotation<typename [:fan::refl::annotation_type<m>():], m>();
    }
  };
}