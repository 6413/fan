module;

#include <string>
#include <vector>
#include <unordered_map>

export module fan.graphics.gameplay.items;

import fan.print;
import fan.graphics;
import fan.graphics.gui.base;
import fan.graphics.gameplay.types;

using namespace fan::graphics;

export namespace fan::graphics::gameplay::items {

  struct effect_t {
    uint32_t type;
    int value;
  };

  struct item_definition_t {
    uint32_t id;
    std::string name;
    fan::graphics::image_t icon;
    uint32_t max_stack;
    std::string description;
    std::vector<effect_t> effects;
  };

  struct registry_t {
    void register_item(const item_definition_t& def) {
      item_definition_t copy = def;
      definitions[def.id] = def;
    }

    const item_definition_t* get_definition(uint32_t id) const {
      auto it = definitions.find(id);
      return it != definitions.end() ? &it->second : nullptr;
    }

    gameplay::item_t create_item(uint32_t id) const {
      auto* def = get_definition(id);
      if (!def) {
        fan::throw_error("Item not found in registry");
      }

      gameplay::item_t item;
      item.id = def->id;
      item.icon = def->icon;
      item.max_stack = def->max_stack;
      item.stack_size = 1;
      return item;
    }

  private:
    std::unordered_map<uint32_t, item_definition_t> definitions;
  };

  registry_t& get_registry() {
    static registry_t r;
    return r;
  }

  gameplay::item_t create(uint32_t id) {
    return get_registry().create_item(id);
  }
}