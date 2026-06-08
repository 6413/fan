module;

export module fan.types.dme;

import std;
import fan.reflection;
import fan.print;
import fan.utility;
import fan.types.compile_time_string;

namespace fan {
  template <typename member_t, auto... Params>
  struct storage_gen {
    struct type;
    consteval {
      auto target = ^^type;
      constexpr auto params = std::make_tuple(Params...);
      std::vector<std::meta::info> specs;

      fan::_for<sizeof...(Params), 2>([&](auto j) {
        constexpr auto idx = decltype(j)::value;
        using annotation_t = [:std::get<idx>(params):];
        constexpr auto name_obj = std::get<idx + 1>(params);

        std::vector<std::meta::info> member_annotations;
        
        if constexpr (!std::is_void_v<annotation_t>) {
          member_annotations.push_back(std::meta::reflect_constant(annotation_t{}));
        }

        specs.push_back(
          std::meta::data_member_spec(
            ^^member_t,
            std::meta::data_member_options{
              .name = std::string(name_obj.data)
            }
          )
        );
      
        //specs.push_back(std::meta::data_member_spec(^^member_t, {
        //  .name = std::string(name_obj.data),
        //  .annotations = member_annotations
        //}));
      });

      std::meta::define_aggregate(target, specs);
    }
  };
}

export namespace fan {

  template <std::size_t N>
  using dme_index_t = fan::smallest_index_t<N>;

  struct required_case {
    std::function<void()> fn;
    template <typename F>
    required_case(F&& f) : fn(std::forward<F>(f)) {}
    void operator()() const { fn(); }
  private:
    struct missing_match_case_all_members_must_have_a_handler {};
    required_case(missing_match_case_all_members_must_have_a_handler) {}
  };

  template <typename derived_t, typename member_t>
  struct dme_t {

        struct id_t {
      using underlying_t = dme_index_t<fan::refl::member_count<derived_t>() + 1>;
      static constexpr underlying_t invalid_value = static_cast<underlying_t>(-1);

      constexpr id_t() : id(invalid_value) {}
      explicit constexpr id_t(underlying_t id_) : id(id_) {}

      static constexpr id_t invalid() { return id_t(invalid_value); }
      constexpr bool is_valid() const { return id != invalid_value; }
      constexpr std::ptrdiff_t gint() const {
        return is_valid() ? static_cast<std::ptrdiff_t>(id) : -1;
      }
      constexpr underlying_t raw() const { return id; }

    private:
      underlying_t id;
    };

    static consteval std::size_t size()                                  { return fan::refl::member_count<derived_t>(); }
    constexpr decltype(auto) operator[](this auto&& self, std::size_t i) { return fan::refl::at_index<derived_t, member_t>(self, i); }
    consteval id_t operator[](std::string_view name) const               { return id_t(static_cast<typename id_t::underlying_t>(fan::refl::index_of<derived_t>(name))); }
    constexpr member_t* begin()                                          { return &(*this)[0]; }
    constexpr member_t* end()                                            { return &(*this)[0] + size(); }
    static constexpr std::string_view name(std::size_t i) {
      static constexpr auto tbl = []<std::size_t... Is>(std::index_sequence<Is...>) consteval {
        return std::array<std::string_view, size()>{fan::refl::member_name<derived_t>(Is)...};
      }(std::make_index_sequence<size()>{});

      if (i >= size()) {
        return {};
      }
      return tbl[i];
    }
    void print()                                                         { fan::print(static_cast<derived_t&>(*this)); }
    static consteval std::meta::info attr_type(std::size_t i)            { return fan::refl::member_annotation_type<derived_t>(i); }

    template <std::size_t I>
    static consteval auto attr()                                         { return fan::refl::member_annotation<I, derived_t>(); }

    struct match_proxy {
      derived_t* self;
      std::size_t i;

      constexpr void operator()(fan::refl::match_t<derived_t, required_case> cases) const {
        fan::refl::match_visit<derived_t, required_case>(i, std::move(cases));
      }
      template <typename... Fs>
      constexpr void operator()(Fs&&... fs) const {
        static_assert(fan::refl::member_count<derived_t>() == sizeof...(Fs),
          "number of cases must match number of members");
        fan::refl::match_visit<derived_t>(i, std::forward<Fs>(fs)...);
      }
    };

    template <std::size_t I>
    struct member_type {
      using annotation = [:fan::refl::member_annotation_type<derived_t>(I):];
      using value      = member_t;
    };

    template <fan::fixed_string Name>
    struct named_member_type {
      static constexpr std::size_t I = fan::refl::index_of<derived_t>(Name.data);
      using annotation = [:fan::refl::member_annotation_type<derived_t>(I):];
      using value      = member_t;
    };

    template <typename Base, std::size_t I>
    struct ann_wrapper : Base {
      static constexpr std::size_t index = I;
      static constexpr std::string_view member_name = fan::refl::member_name<derived_t>(I);
      ann_wrapper() : Base{} {
        if constexpr (requires(Base& a) { a.init(); }) { Base::init(); }
      }
      ~ann_wrapper() {
        if constexpr (requires(Base& a) { a.destroy(); }) { Base::destroy(); }
      }
    };

    template <fan::fixed_string Name>
    using ann = ann_wrapper<typename named_member_type<Name>::annotation, named_member_type<Name>::I>;

    template <std::size_t I>
    using ann_i = ann_wrapper<typename member_type<I>::annotation, I>;

    constexpr match_proxy match(std::size_t i) { return { static_cast<derived_t*>(this), i }; }
    constexpr match_proxy match(id_t i) { return { static_cast<derived_t*>(this), static_cast<std::size_t>(i.raw()) }; }

    static consteval auto members() { return fan::refl::members<derived_t>(); }

    static consteval auto member_ptrs() {
      constexpr auto ms       = members();
      constexpr std::size_t N = ms.size();

      return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::array{std::meta::substitute(^^std::add_pointer_t, {std::meta::type_of(ms[Is])})...};
      }(std::make_index_sequence<N>{});
    }

    static consteval auto runtime_table() {
      return fan::refl::runtime_table<derived_t>();
    }

        template <typename F>
    static void visit(id_t idx, F&& f) {
      if (!idx.is_valid()) {
        return;
      }

      fan::refl::for_each_member_indexed<derived_t>([&]<std::size_t i, std::meta::info m>() {
        if (i == static_cast<std::size_t>(idx.gint())) {
          constexpr auto ann_type = fan::refl::member_annotation_type<derived_t>(i);
          if constexpr (ann_type != std::meta::info{}) {
            ann_i<i> wrapped;
            f(wrapped);
          }
        }
      });
    }
  };

  template <typename member_t, typename derived_t, auto... Params>
  struct dme_builder {
    using base_t = typename fan::storage_gen<member_t, Params...>::type;

    struct type : base_t, fan::dme_t<type, member_t> {
      using base_t::base_t;
    };
  };
}