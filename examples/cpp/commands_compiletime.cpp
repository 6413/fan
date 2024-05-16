#include <fan/pch.h>

#include <ranges>

template <typename T>
struct function_traits
{
  template <typename R, typename ... As>
  static std::tuple<As...> pro_args(std::function<R(As...)>);

  using arguments = decltype(pro_args(std::function{ std::declval<T>() }));
};

template <typename T>
struct wrap_t {
  using types = T;
};

struct command_t {
  void* func;
  std::function<void()> args;
};

std::unordered_map<std::string, command_t> func_table;

struct command_errors_e {
  enum {
    success,
    function_not_found,
    invalid_args
  };
};


void register_command(const fan::string& cmd, auto func) {
  //function_traits<decltype(l)>::arg<0>::type
  command_t command;
  command.func = func;
  iterate_lambda_args([&](const std::string& arg) {
    //command.args.push_back(arg);
    }, func);

  func_table[cmd] = command;
}

template <typename T>
struct single_command_t {
  single_command_t(const fan::string& name, T func) {
    this->name = name;
    this->func = func;
  }
  fan::string name;
  T func;
};


template <typename T>
single_command_t<T> make_command(const fan::string& name, T func) {
  single_command_t<T> c(name, func);
  return c;
}

#define auto_inside_struct(varname, func) \
  using varname##func_t = decltype(func); \
  decltype(make_command(STRINGIFY(varname), varname##func_t())) varname = make_command(STRINGIFY(varname), varname##func_t());

struct commands_t {
  auto_inside_struct(add, [](int x) {
    fan::print("a", x);
    });
  auto_inside_struct(del, [](double x) {
    fan::print("d", x);
    });
  auto_inside_struct(echo, [](const std::string& str) {
    fan::print(str);
    });
  auto_inside_struct(help, [](const std::string& str) {
    if (str.empty()) {
      fan::print("help\necho");
    }
    else {
      if (str == "noclip") {
        fan::print("toggles noclip");
      }
      else if (str == "help") {
        fan::print("helps help");
      }
    }
    });
};

template<class T> T transform_arg(std::string const& s);
template<> double transform_arg(std::string const& s) { return atof(s.c_str()); }
template<> int transform_arg(std::string const& s) { return atoi(s.c_str()); }
//template<> std::string transform_arg(std::string const& s) { return s; }
template<> const std::string& transform_arg(std::string const& s) { return s; }

template <typename... Args, std::size_t... Is>
auto create_tuple_impl(std::index_sequence<Is...>, const std::vector<std::string>& arguments) {
  return std::make_tuple(transform_arg<Args>(arguments[Is])...);
}

template <typename... Args>
auto create_tuple(const std::vector<std::string>& arguments) {
  return create_tuple_impl<Args...>(std::index_sequence_for<Args...>{}, arguments);
}

fan::mp_t<commands_t> comm;

template <typename Tuple>
struct get_variant;

template <typename... Ts>
struct get_variant<std::tuple<Ts...>>
{
  using type = std::variant<std::monostate, std::remove_reference_t<Ts> ...>;
};

std::vector<std::string> split(std::string str, std::string token) {
  std::vector<std::string>result;
  while (str.size()) {
    int index = str.find(token);
    if (index != std::string::npos) {
      result.push_back(str.substr(0, index));
      str = str.substr(index + token.size());
      if (str.size() == 0)result.push_back(str);
    }
    else {
      result.push_back(str);
      str = "";
    }
  }
  return result;
}

struct final_commands_t {
  using tuple_t = decltype(comm.get_tuple());
  using var_t = typename get_variant<tuple_t>::type;

  std::unordered_map<std::string, var_t> map;
  final_commands_t() {
    comm.iterate([&]<std::size_t i, typename T>(T & f) {
      map[f.name] = f;
    });
  }
}final_commands;

int call_command(const fan::string& cmd) {
  std::size_t arg0_off = cmd.find(" ");
  if (arg0_off == std::string::npos) {
    arg0_off = cmd.size();
  }
  fan::string arg0 = cmd.substr(0, arg0_off);
  auto found = final_commands.map.find(arg0);
  if (found == final_commands.map.end()) {
    return command_errors_e::function_not_found;
  }
  fan::string rest;
  if (arg0_off + 2 > cmd.size()) {
    rest = "";
  }
  else {
    rest = cmd.substr(arg0_off + 1);
  }
  std::visit([&]<typename T>(T & x) {
    if constexpr (!std::is_same_v<T, std::monostate>) {
      auto line = split(rest, " ");
      typename function_traits<decltype(x.func)>::arguments* tupl;
      [&] <typename... Ts>(std::tuple<Ts...>)
      {
        if (line.empty()) {
          std::apply(x.func, std::tuple<Ts...>{{}});
        }
        else {
          std::apply(x.func, create_tuple<Ts...>(line));
        }
      }(*tupl);
    }
  }, found->second);
  return command_errors_e::success;
}

int main() {

  call_command("add 3.4");
  call_command("del 4.5");
  call_command("echo hey");
  call_command("help");
  call_command("help help");
  call_command("help noclip");
}