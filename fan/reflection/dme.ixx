module;

export module fan.types.dme;

import std;
import fan.reflection;

export namespace fan {
  template <typename derived_t, typename member_t>
  struct dme_t {
    static consteval std::size_t size()                                  { return fan::refl::member_count<derived_t>(); }
    constexpr decltype(auto) operator[](this auto&& self, std::size_t i) { return fan::refl::at_index<derived_t>(self, i); }
    consteval std::size_t operator[](std::string_view name) const        { return fan::refl::index_of<derived_t>(name); }
    constexpr member_t* begin()                                          { return &(*this)[0]; }
    constexpr member_t* end()                                            { return &(*this)[0] + size(); }
    static constexpr std::string_view name(std::size_t i)                { return fan::refl::member_name<derived_t>(i); }
    void print()                                                         { fan::print_reflect(static_cast<derived_t&>(*this)); }
    static consteval std::meta::info attr_type(std::size_t i)            { return fan::refl::member_annotation_type<derived_t>(i); }

    template <std::size_t I>
    static consteval auto attr()                                         { return fan::refl::member_annotation<I, derived_t>(); }

    struct match_proxy {
      derived_t* self;
      std::size_t i;
      constexpr void operator()(fan::refl::match_t<derived_t> cases) const {
        fan::refl::match_visit<derived_t>(i, std::move(cases));
      }
      template <typename... Fs>
      constexpr void operator()(Fs&&... fs) const {
        fan::refl::match_visit<derived_t>(i, std::forward<Fs>(fs)...);
      }
    };

    constexpr match_proxy match(std::size_t i) { return { static_cast<derived_t*>(this), i }; }
  };
}