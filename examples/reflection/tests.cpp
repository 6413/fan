#include <fan/utility.h>
import fan;
import std;
import fan.reflection;
import fan.types.dme;
import fan.rjson;


constexpr char alphabet[] = "abcdefghijklmnopqrstuvwxyz";

template <std::size_t N, std::size_t Seed>
consteval auto make_shuffled_indices() {
  std::array<std::size_t, N> indices;
  for (std::size_t i = 0; i < N; ++i) indices[i] = i;

  std::size_t state = Seed ^ (N * 2654435761ULL);
  for (std::size_t i = N - 1; i > 0; --i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    std::size_t j = state % (i + 1);
    auto tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
  return indices;
}

template <std::size_t N, std::size_t Seed>
consteval auto make_random_chars() {
    constexpr std::string_view charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::size_t state = Seed ^ (N * 2654435761ULL);

    std::array<char, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      result[i] = charset[state % charset.size()];
    }
    return result;
}

template <std::size_t N, std::size_t Seed>
consteval auto make_unique_name(std::size_t i) {
  constexpr auto chars = make_random_chars<N, Seed>();

  constexpr std::string_view letters =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

  char buf[16]{};
  std::size_t pos = 0;

  // FIRST CHARACTER → must be letter or '_'
  buf[pos++] = letters[i % letters.size()];

  // rest can be alphanumeric
  for (std::size_t j = 0; j < 2; ++j) {
    buf[pos++] = chars[(i + j) % chars.size()];
  }

  // append index (guarantees uniqueness)
  std::size_t x = i;
  char digits[10];
  std::size_t d = 0;

  do {
    digits[d++] = char('0' + (x % 10));
    x /= 10;
  } while (x);

  while (d--) {
    buf[pos++] = digits[d];
  }

  return std::define_static_string(std::string_view(buf, pos));
}

template <typename TO, typename T, std::size_t N, std::size_t Seed = 12345>
consteval auto gen_types_impl() {
  return []<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::meta::define_aggregate(^^TO, {
      std::meta::data_member_spec(^^T, {
        .name = make_unique_name<N, Seed>(Is)
      })...
    });
  }(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N, std::size_t Seed = 12345>
struct gen_types {
  struct type;
  consteval {
    gen_types_impl<type, T, N, Seed>();
  }
};

template <typename T, std::size_t N, std::size_t Seed = 12345>
using gen_types_t = gen_types<T, N, Seed>::type;

consteval std::size_t time_seed() {
  std::string_view t = __TIME__; // "HH:MM:SS"
  return (t[0]-'0')*100000 + (t[1]-'0')*10000 +
         (t[3]-'0')*1000  + (t[4]-'0')*100  +
         (t[6]-'0')*10    + (t[7]-'0');
}

using v = gen_types_t<int, 5, time_seed() ^ (__COUNTER__ * 2654435761ULL)>;


struct MyTypeBase;  // incomplete - declared but not defined

consteval std::meta::info make_struct(
  std::meta::info target,
  std::vector<std::pair<std::string_view, std::meta::info>> fields
) {
  std::vector<std::meta::info> members;
  for (auto& [name, type] : fields)
    members.push_back(std::meta::data_member_spec(type, {.name = std::string(name)}));
  return std::meta::define_aggregate(target, members);
}

struct MyType;
consteval { make_struct(^^MyType, {{"x", ^^int}, {"y", ^^float}}); }

using Player = fan::json_schema_t<R"({"x":"float","y":"float","hp":"int"})">;


static constexpr char schema_data[] = {
  #embed "types_to_parse.json"
  , '\0'
};
static constexpr std::string_view entity_schema{schema_data, sizeof(schema_data) - 1};

struct entity_t;
consteval { fan::json_schema_to_struct(^^entity_t, entity_schema); }

int main() {
  Player p{.x = 1.0f, .y = 2.0f, .hp = 100};
  fan::print(p);

  {
    MyType s{.x = 1, .y = 2.0f};
    fan::print(s);
  }

  entity_t e{.health = 100, .speed = 5.5f, .armor = 30, .alive = true};
  fan::print(e);

  auto s = fan::to_rjson_string(e, 2);
  fan::print(s);

  auto e2 = fan::from_rjson_string<entity_t>(s);
  fan::print(e2);
}