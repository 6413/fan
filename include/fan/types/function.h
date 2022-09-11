#pragma once

namespace fan {

  template <typename T>
  struct function_t;

  template <typename ret_type, typename ...args_t>
  struct function_t<ret_type(args_t...)> {

    using return_type = ret_type;
    typedef ret_type(*func_type)(args_t&&...);

    //function_t() {
    //  auto l = [](args_t...)-> ret_type { return {}; };
    //  data = new func_t<decltype(l) >>(l);
    //};

    function_t() : func(0) {};

    template <typename T>
    function_t(T lambda) : func(new func_t<T>(lambda)) {}

    //template <typename T>
    //function_t(T&& lambda) : func(new func_t<T>(lambda)) {}

    function_t(const function_t& f) {
      func = f.func;
    }
    function_t(function_t&& f) {
      func = std::move(f.func);
      f.func = 0;
    }

    template <typename T2>
    function_t& operator=(T2 f) {
      func = new func_t<T2>(f);

      return *this;
    };

    function_t& operator=(const function_t& f) = default;
    function_t& operator=(function_t&& f) = default;

    //template <typename T2>
    //function_t& operator=(T2&& f) {
    //  delete func;
    //  func = new func_t<T2>(f);
    //  return *this;
    //};

    ~function_t() { delete func; }

    struct func_base_t {
      func_base_t() = default;
      virtual return_type operator()(args_t... args) = 0;
      virtual ~func_base_t() = default;
    };

    template <typename T>
    struct func_t : func_base_t {
      func_t(const T& o) : f(o) {}
      ~func_t() override = default;
      return_type operator()(args_t... args) override {
        return f(args...);
      }
      T f;
    };

    return_type operator()(args_t... args) {
      return (*func)(args...);
    }

    func_base_t* func;
  };
}