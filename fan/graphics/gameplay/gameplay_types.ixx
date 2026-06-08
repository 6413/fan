module;

export module fan.graphics.gameplay.types;

#if defined (FAN_WINDOW)

import std;

import fan.types.color;

import fan.graphics.common_context;

export namespace fan::graphics::gameplay {

  struct item_t {
    std::uint32_t id;
    fan::graphics::image_t icon;
    std::uint32_t stack_size = 1;
    std::uint32_t max_stack = 99;
    void* user_data = nullptr;
  };

  struct gui_theme_t {
    fan::color panel_bg;
    fan::color panel_border;
    fan::color panel_corner_accent;
    fan::color slot_bg;
    fan::color slot_bg_hover;
    fan::color slot_border;
    fan::color selected_border_color;
  };

  struct item_slot_t {
    bool is_empty() const {
      return !id.has_value();
    }

    bool can_add(std::uint32_t new_id, std::uint32_t max_stack, std::uint32_t amount = 1) const {
      if (amount == 0) {
        return false;
      }
      if (is_empty()) {
        return amount <= max_stack;
      }
      if (*id != new_id) {
        return false;
      }
      if (*stack_size + amount > max_stack) {
        return false;
      }
      return true;
    }

    bool add(std::uint32_t new_id, std::uint32_t max_stack, std::uint32_t amount = 1) {
      if (!can_add(new_id, max_stack, amount)) {
        return false;
      }
      if (is_empty()) {
        id = new_id;
        stack_size = amount;
        return true;
      }
      *stack_size += amount;
      return true;
    }

    bool remove(std::uint32_t amount = 1) {
      if (is_empty()) {
        return false;
      }
      if (amount == 0) {
        return false;
      }
      if (amount > *stack_size) {
        return false;
      }
      if (amount == *stack_size) {
        id.reset();
        stack_size.reset();
        return true;
      }
      *stack_size -= amount;
      return true;
    }

    std::optional<std::uint32_t> id;
    std::optional<std::uint32_t> stack_size;
  };
}

#endif