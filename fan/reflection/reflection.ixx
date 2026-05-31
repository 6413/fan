module;

#include <fan/utility.h>

export module fan.reflection;

import std;
import fan.utility;
import fan.types.color;
import fan.print;

// types
export namespace fan::refl {
   struct member_info {
    const char* name;
    std::size_t size;
    std::size_t offset;
    const std::type_info* type;
  };
}

// free functions
export namespace fan {
  consteval std::string_view make_name(std::size_t i, char prefix = 'v') {
    char buf[32] = {};
    std::size_t pos = 31;

    if (i == 0) {
      buf[--pos] = '0';
    } else {
      while (i > 0) {
        buf[--pos] = char('0' + (i % 10));
        i /= 10;
      }
    }
    buf[--pos] = prefix;

    return std::define_static_string(std::string_view(&buf[pos]));
  }
}

export namespace fan::refl {
  // fan/reflection/reflection.ixx

template <typename T>
consteval auto members() {
  std::vector<std::meta::info> all_members;
  constexpr auto ctx = std::meta::access_context::unchecked();
  auto collect = [&](auto&& self, std::meta::info type_refl) -> void {
    for (auto m : std::meta::nonstatic_data_members_of(type_refl, ctx)) {
      all_members.push_back(m);
    }
    for (auto base : std::meta::bases_of(type_refl, ctx)) {
      self(self, std::meta::type_of(base));
    }
  };
  collect(collect, ^^T);
  return std::define_static_array(all_members);
}

  template <typename T>
  inline constexpr auto members_v = members<T>();

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
  constexpr void for_each_member_indexed(F&& f) {
    fan::_for<member_count<T>()>([&](auto I) {
      constexpr std::size_t i = decltype(I)::value;
      constexpr auto m = members<T>()[i];
      f.template operator()<i, m>();
    });
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

  template <typename T, typename Member, typename Obj>
  constexpr decltype(auto) at_index(Obj&& obj, std::size_t i) {
    using obj_t = std::remove_reference_t<Obj>;
    constexpr bool is_const = std::is_const_v<obj_t>;
    using ptr_t = std::conditional_t<is_const, const Member*, Member*>;

    ptr_t ptr = nullptr;
    member_visit<T>(i, [&]<std::meta::info m> {
      if constexpr (is_const) ptr = std::addressof(static_cast<const T&>(obj).*(&[:m:]));
      else ptr = std::addressof(static_cast<T&>(obj).*(&[:m:]));
    });

    return *ptr;
  }

  template <typename T, typename F>
  constexpr void iterate_members(auto&& obj, F&& func) {
    for_each_member<T>([&]<std::meta::info m> { func(std::meta::identifier_of(m), obj.[:m:]); });
  }

  template <typename T, typename F>
  constexpr void apply_at(auto&& obj, std::size_t i, F&& func) {
    member_visit<T>(i, [&]<std::meta::info m> { func(obj.[:m:]); });
  }

  consteval std::string_view name_of(std::meta::info t) {
    t = std::meta::dealias(t);
    if (std::meta::has_template_arguments(t))
      return std::meta::identifier_of(std::meta::template_of(t));
    if (std::meta::has_identifier(t))
      return std::meta::identifier_of(t);
    return std::meta::display_string_of(t);
  }

  template <typename T>
  consteval std::string_view name_of() {
    return name_of(^^T);
  }

  template <typename TAttr, std::meta::info M>
  consteval TAttr annotation() {
    template for (constexpr auto ann : std::define_static_array(std::meta::annotations_of(M))) {
      if constexpr (std::meta::remove_cv(std::meta::type_of(ann)) == ^^TAttr) { return std::meta::extract<TAttr>(ann); }
    }
    return TAttr{};
  }
  
  template <std::meta::info M>
  consteval std::meta::info annotation_type() {
    template for (constexpr auto ann : std::define_static_array(std::meta::annotations_of(M))) {
      return std::meta::remove_cv(std::meta::type_of(ann));
    }
    return std::meta::info{};
  }

  template <typename T>
  consteval std::meta::info member_annotation_type(std::size_t i) {
    return member_at<T>(i, []<std::meta::info m>{ return annotation_type<m>(); });
  }

  template <std::size_t I, typename T>
  consteval auto member_annotation() {
    constexpr auto m = members<T>()[I];
    using attr_t = typename [:annotation_type<m>():];
    return annotation<attr_t, m>();
  }


  template <typename T>
  consteval std::string_view member_name(std::size_t i) {
    return member_at<T>(i, []<std::meta::info m> { return name_of(m); });
  }

  template <typename T>
  consteval std::size_t offset_of(std::size_t i) {
    return member_at<T>(i, []<std::meta::info m> { return std::meta::offset_of(m); });
  }

  template <typename T>
  consteval std::size_t index_of(std::string_view name) {
    std::size_t cur = 0;
    template for (constexpr auto m : members<T>()) {
      if (name_of(m) == name) return cur;
      ++cur;
    }
    return ~std::size_t{0};
  }

  consteval std::meta::info dealias(std::meta::info t) {
    return std::meta::dealias(t);
  }
  consteval std::meta::info type_of(std::meta::info t) {
    return std::meta::type_of(t);
  }
  consteval bool is_same_name(std::meta::info a, std::meta::info b) {
    return name_of(a) == name_of(b);
  }
  consteval bool is_same_type(std::meta::info a, std::meta::info b) {
    return std::meta::dealias(a) == std::meta::dealias(b);
  }

  template <typename T>
  consteval auto make_match_type_impl(auto target, std::meta::info case_type) {
    std::vector<std::meta::info> specs;
    for (auto m : std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked())) {
      specs.push_back(std::meta::data_member_spec(
        case_type,
        { .name = std::string(std::meta::identifier_of(m)) }
      ));
    }
    return std::meta::define_aggregate(target, specs);
  }

  template <typename T, typename CaseType = std::function<void()>>
  struct match_type {
    struct type;
    consteval { make_match_type_impl<T>(^^type, ^^CaseType); }
  };

  template <typename T, typename CaseType = std::function<void()>>
  using match_t = match_type<T, CaseType>::type;

  template <typename T>
  constexpr void match_visit(std::size_t i, match_t<T> cases) {
    member_visit<match_t<T>>(i, [&]<std::meta::info m> {
      cases.[:m:]();
    });
  }


  template <typename T, typename CaseType>
  constexpr void match_visit(std::size_t i, typename match_type<T, CaseType>::type cases) {
    member_visit<typename match_type<T, CaseType>::type>(i, [&]<std::meta::info m> {
      cases.[:m:]();
    });
  }

  template <typename T, typename... Fs>
  constexpr void match_visit(std::size_t idx, Fs&&... fs) {
    static_assert(sizeof...(Fs) == member_count<T>(), "wrong number of cases");
    fan::_for<sizeof...(Fs)>([&](auto I) {
      constexpr std::size_t i = decltype(I)::value;
      if (i == idx) std::get<i>(std::forward_as_tuple(fs...))();
    });
  }

  template <class insert_t, class from_t>
  consteval auto dynamic_type() {
    std::vector<std::meta::info> specs;
    for (auto m : std::meta::nonstatic_data_members_of(^^from_t, std::meta::access_context::unchecked())) {
      specs.push_back(std::meta::data_member_spec(std::meta::type_of(m), {.name = fan::refl::name_of(m)}));
    }
    return std::meta::define_aggregate(^^insert_t, specs);
  }
  template <class T>
  struct dynamic {
    struct type;
    consteval {
      dynamic_type<type, T>();
    }
  };
  template <class T>
  using dynamic_t = dynamic<T>::type;

  template <typename TO, typename T, std::size_t N>
  consteval auto gen_types_impl() {
    std::vector<std::meta::info> specs;
    for (int i = 0; i < N; ++i) {
      specs.push_back(std::meta::data_member_spec(
        ^^T,
        std::meta::data_member_options{.name = make_name(i)}
      ));
    }
    return std::meta::define_aggregate(^^TO, specs);
  }

  template <typename T, std::size_t N>
  struct gen_types {
    struct type;
    consteval {
      gen_types_impl<type, T, N>();
    }
  };

  template <typename T, std::size_t N>
  using gen_types_t = gen_types<T, N>::type;

  template <typename... Ts>
  auto make_struct(Ts&&... args) {
    struct result;
    consteval {
      std::vector<std::meta::info> specs;
      std::size_t i = 0;
      ((specs.push_back(std::meta::data_member_spec(^^Ts, {.name = fan::make_name(i++)}))), ...);
      std::meta::define_aggregate(^^result, specs);
    }
    return result{std::forward<Ts>(args)...};
  }

  template <std::meta::reflection_range _Rg = std::initializer_list<std::meta::info>>
  consteval std::meta::info define_aggregate(std::meta::info target, _Rg&& members) {
    return std::meta::define_aggregate(target, std::forward<_Rg>(members));
  }

  template <typename T>
  consteval bool has_member(std::string_view name) {
    template for (constexpr auto m : members<T>()) {
      if (std::meta::identifier_of(m) == name) return true;
    }
    return false;
  }

  template <typename T>
  consteval std::meta::info member_type(std::string_view name) {
    template for (constexpr auto m : members<T>()) {
      if (std::meta::identifier_of(m) == name)
        return std::meta::type_of(m);
    }
    return std::meta::info{};
  }

  template <typename T>
  consteval bool has_same_members(std::meta::info a, std::meta::info b) {
    auto ma = std::meta::nonstatic_data_members_of(a, std::meta::access_context::unchecked());
    auto mb = std::meta::nonstatic_data_members_of(b, std::meta::access_context::unchecked());
    if (ma.size() != mb.size()) return false;
    for (std::size_t i = 0; i < ma.size(); ++i)
      if (std::meta::type_of(ma[i]) != std::meta::type_of(mb[i])) return false;
    return true;
  }

  template <typename To, typename From>
  constexpr To struct_cast(const From& from) {
    To result{};
    template for (constexpr auto m : members<To>()) {
      constexpr auto name = std::meta::identifier_of(m);
      if constexpr (has_member<From>(name)) {
        constexpr auto src = []<std::meta::info fm>(){ return fm; }
          .template operator()<*std::ranges::find_if(
            std::meta::nonstatic_data_members_of(^^From, std::meta::access_context::unchecked()),
            [](auto x){ return std::meta::identifier_of(x) == name; }
          )>();
        result.[:m:] = from.[:src:];
      }
    }
    return result;
  }

  template <typename A, typename B>
  consteval bool is_layout_compatible() {
    auto ma = std::meta::nonstatic_data_members_of(^^A, std::meta::access_context::unchecked());
    auto mb = std::meta::nonstatic_data_members_of(^^B, std::meta::access_context::unchecked());
    if (ma.size() != mb.size()) return false;
    for (std::size_t i = 0; i < ma.size(); ++i)
      if (std::meta::type_of(ma[i]) != std::meta::type_of(mb[i])) return false;
    return true;
  }

  template <typename T>
  constexpr auto to_tuple(T&& obj) {
    using U = std::remove_cvref_t<T>;
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      constexpr auto ms = std::define_static_array(
        std::meta::nonstatic_data_members_of(^^U, std::meta::access_context::unchecked())
      );
      return std::make_tuple(obj.[:ms[Is]:]...);
    }(std::make_index_sequence<members<U>().size()>{});
  }

  template <typename T>
  constexpr auto to_tie(T& obj) {
    using U = std::remove_cvref_t<T>;
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      constexpr auto ms = std::define_static_array(
        std::meta::nonstatic_data_members_of(^^U, std::meta::access_context::unchecked())
      );
      return std::tie(obj.[:ms[Is]:]...);
    }(std::make_index_sequence<members<U>().size()>{});
  }

  template <typename T>
  constexpr bool equal(const T& a, const T& b) {
    bool result = true;
    template for (constexpr auto m : members<T>()) {
      if (a.[:m:] != b.[:m:]) { result = false; break; }
    }
    return result;
  }

  template <typename T>
  constexpr std::size_t hash(const T& obj) {
    std::size_t seed = 0;
    template for (constexpr auto m : members<T>()) {
      seed ^= std::hash<std::remove_cvref_t<decltype(obj.[:m:])>>{}(obj.[:m:])
              + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }

  template <typename T>
  constexpr T zero() {
    T result{};
    template for (constexpr auto m : members<T>()) {
      result.[:m:] = {};
    }
    return result;
  }

  template <typename TAttr, std::meta::info M>
  consteval bool has_annotation() {
    template for (constexpr auto ann : std::define_static_array(std::meta::annotations_of(M))) {
      if constexpr (std::meta::remove_cv(std::meta::type_of(ann)) == ^^TAttr) return true;
    }
    return false;
  }

  // iterate only members that have a specific annotation
  template <typename T, typename TAttr, typename F>
  constexpr void for_each_annotated(F&& func) {
    template for (constexpr auto m : members<T>()) {
      if constexpr (has_annotation<TAttr, m>()) {
        func.template operator()<m>();
      }
    }
  }

  template <std::meta::info M>
  consteval auto member_ptr_of() {
    using C = typename [:std::meta::parent_of(M):];
    using T = typename [:std::meta::type_of(M):];

    return std::meta::substitute(
      ^^T C::*,
      {}
    );
  }

  template <typename T>
  consteval auto member_ptrs_of() {
    constexpr auto ms = members<T>();

    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return std::array{ member_ptr_of<ms[Is]>()... };
    }(std::make_index_sequence<ms.size()>{});
  }

  template <typename T>
  consteval auto runtime_table() {
    constexpr auto ms = members<T>();
    constexpr std::size_t N = ms.size();

    std::array<fan::refl::member_info, N> table{};

    fan::_for<N>([&](auto I) {
      constexpr std::size_t Is = decltype(I)::value;

      table[Is] = fan::refl::member_info{
        .name   = std::meta::identifier_of(ms[Is]).data(),
        .size   = std::meta::size_of(ms[Is]),
        .offset = std::meta::offset_of(ms[Is]).bytes,
        .type   = &typeid(std::meta::type_of(ms[Is]))
      };
    });

    return table;
  }
}  // namespace fan::refl


namespace fan::detail {
  template <typename T>
  std::string format_reflect(const T& obj, int indent_level = 0) {
    using U = std::remove_cvref_t<T>;

    if constexpr (requires { std::declval<std::ostream&>() << obj; }) {
      std::ostringstream val;
      val << obj;
      static constexpr auto raw_t_name = fan::refl::name_of<U>();
      std::string t{raw_t_name};
      if (t.starts_with("member_t {aka ")) t = t.substr(14, t.size() - 15);

      return fan::paint(fan::colors::white, val.str()) + 
             fan::paint(fan::colors::gray, " [") +
             fan::paint(fan::colors::teal, t) + 
             fan::paint(fan::colors::gray, "]");
    } 
    else {
      std::string ind(indent_level * 2, ' ');
      static constexpr auto type_name = fan::refl::name_of<U>();
      
      std::string out = fan::paint(fan::colors::cyan, std::string{type_name}) +
                        fan::paint(fan::colors::gray, " {\n");

      static constexpr auto members = fan::refl::members<U>();

      template for (constexpr auto m : members) {
        static constexpr auto m_name = fan::refl::name_of(m);
        out += fan::paint(fan::colors::amber, ind + "  ." + std::string{m_name} + " = ");
        out += format_reflect(obj.[:m:], indent_level + 1) + "\n";

        static constexpr auto anns = std::define_static_array(std::meta::annotations_of(m));
        if constexpr (anns.size() > 0) {
          out += ind + "  " + fan::paint(fan::colors::gray, "[[");
          
          std::size_t j = 0;
          template for (constexpr auto ann : anns) {
            using ann_t = typename [:std::meta::type_of(ann):];
            // Recursion
            out += format_reflect(std::meta::extract<ann_t>(ann), indent_level + 1);
            if (j++ < anns.size() - 1) out += ", ";
          }
          
          out += fan::paint(fan::colors::gray, "]]\n");
        }
      }
      return out + fan::paint(fan::colors::gray, ind + "}");
    }
  }
}

export namespace fan {

  template <typename... Args>
  consteval auto data_member_spec(Args&&... args) {
    return std::meta::data_member_spec(
      std::forward<Args>(args)...
    );
  }

  template <typename T>
  consteval auto reflect_constant(T&& value) {
    return std::meta::reflect_constant(
      std::forward<T>(value)
    );
  }

  using data_member_options = std::meta::data_member_options;
  using info = std::meta::info;

  template <typename... T>
  void print(const T&... args) {
    (fan::print_color_raw(fan::colors::white, fan::detail::format_reflect(args) + "\n"), ...);
  }
}