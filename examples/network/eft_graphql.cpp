#include <coroutine>
#include <string>
#include <vector>
#include <expected>
#undef min
#undef max
import fan;

fan::event::task_value_resume_t<std::expected<void, std::string>> get_all_weapons() {
  fan::json weapons_query = {
      { "query", R"(
            query GetWeapons {
                items(lang: en, gameMode: regular) {
                    id
                    name
                    shortName
                    basePrice
                    avg24hPrice
                    weight
                    iconLink
                    types
                    properties {
                        __typename
                        ...on ItemPropertiesWeapon {
                            caliber
                            effectiveDistance
                            ergonomics
                            fireModes
                            fireRate
                            recoilVertical
                            recoilHorizontal
                            sightingRange
                        }
                    }
                }
            }
        )" }
  };

  auto result = co_await fan::network::http::post(
    "https://api.tarkov.dev/graphql",
    weapons_query.dump(),
    { { "Content-Type", "application/json" } },
    { .verify_ssl = 1, .timeout_seconds = 30 }
  );

  if (!result) {
    co_return std::unexpected(result.error());
  }

  if (result->status_code != 200) {
    co_return std::unexpected("HTTP error: " + std::to_string(result->status_code));
  }

  auto json_response = fan::json::parse(result->body);

  if (!json_response.contains("data") || !json_response["data"].contains("items")) {
    co_return std::unexpected("Invalid response structure");
  }

  auto items = json_response["data"]["items"];
  std::vector<fan::json> weapons;

  for (const auto& item : items) {
    if (item.contains("properties") &&
      item["properties"].contains("__typename") &&
      item["properties"]["__typename"].get<std::string>() == "ItemPropertiesWeapon") {
      weapons.push_back(item);
    }
  }

  for (size_t i = 0; i < std::min<size_t>(10, weapons.size()); ++i) {
    const auto& weapon = weapons[i];
    fan::print_color(fan::colors::white, std::to_string(i + 1) + ".", weapon["name"].get<std::string>());
    fan::print_color(fan::colors::white, "   Short:", weapon.value("shortName", "unknown"));
    fan::print_color(fan::colors::green, "   Base Price:", std::to_string(weapon.value("basePrice", 0)), "₽");

    int avg_price = 0;
    if (weapon.contains("avg24hPrice") && !weapon["avg24hPrice"].is_null()) {
      avg_price = weapon["avg24hPrice"].get<int>();
    }
    fan::print_color(fan::colors::yellow, "   24h Avg Price:", avg_price > 0 ? std::to_string(avg_price) + "₽" : "N/A");

    if (weapon.contains("properties") && weapon["properties"].contains("caliber")) {
      const auto& props = weapon["properties"];
      fan::print_color(fan::colors::white, "   Caliber:", props.value("caliber", "unknown"));
      fan::print_color(fan::colors::white, "   Fire Rate:", std::to_string(props.value("fireRate", 0)), "RPM");
      fan::print_color(fan::colors::white, "   Ergonomics:", std::to_string(props.value("ergonomics", 0)));
      fan::print_color(fan::colors::white, "   Recoil V/H:",
        std::to_string(props.value("recoilVertical", 0)) + "/" +
        std::to_string(props.value("recoilHorizontal", 0)));
    }

    fan::print_color(fan::colors::white, "   Weight:", std::to_string(weapon.value("weight", 0.0)), "kg");
    fan::print_color(fan::colors::white, "");
  }

  co_return{};
}

fan::event::task_t main_task() {
  auto weapons_result = co_await get_all_weapons();
  if (!weapons_result) {
    fan::print_color(fan::colors::red, "Error:", weapons_result.error());
  }
}

int main() {
  main_task(); // launch coroutine
  fan::event::loop(); // run event loop
  return 0;
}
