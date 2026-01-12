#define gameplay fan::graphics::gameplay

namespace items {
  struct id_e {
    uint32_t val;
    id_e(uint32_t new_val) : val(new_val) {}
    operator uint32_t& () {
      return val;
    }
    enum : uint32_t {
      health_potion,
      mana_potion,
      iron_shield
    };
  };

  void init() {
    auto& r = gameplay::items::get_registry();

    r.register_item({
      .id = id_e::health_potion,
      .name = "Health Potion",
      .icon = "gui/health_potion.webp",
      .max_stack = 20,
      .description = "Restores 20 HP",
      .effects = {
        {.type = id_e::health_potion, .value = 20 }
      }
    });

    r.register_item({
      .id = id_e::mana_potion,
      .name = "Mana Potion",
      .icon = "gui/mana_potion.webp",
      .max_stack = 20,
      .description = "Restores 50 MP",
      .effects = {
        {.type = id_e::mana_potion, .value = 50 }
      }
    });
    r.register_item({
      .id = id_e::iron_shield,
      .name = "Iron Shield",
      .icon = "images/icon_iron_shield.webp",
      .max_stack = 20,
      .description = "Iron Shield"
    });
  }
}

#undef gameplay