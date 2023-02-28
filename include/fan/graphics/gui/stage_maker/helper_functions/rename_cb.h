#ifndef return_value
  #define return_value
#endif

if (!pile->loco.vfi.get_focus_text().is_invalid()) {

  pile->loco.menu_maker_button.set_selected(instances[stage_t::stage_options].menu_id, nullptr);
  return return_value;
}

auto selected_id = pile->loco.menu_maker_button.get_selected(instances[stage_t::stage_instance].menu_id);

loco_t::text_box_t::properties_t tp;
tp.matrices = &matrices;
tp.viewport = &viewport;
tp.theme = &theme;
tp.position = pile->loco.button.get(
  selected_id,
  &loco_t::button_t::vi_t::position
);
tp.position.z += 5;
tp.size = pile->loco.button.get(
  selected_id,
  &loco_t::button_t::vi_t::size
);
tp.text = pile->loco.button.get_text(
  selected_id
);
tp.font_size = pile->loco.button.get_text_instance(
  selected_id
).font_size;
tp.keyboard_cb = [this, selected_id](const loco_t::keyboard_data_t& d) -> int {
  if (d.keyboard_state != fan::keyboard_state::press) {

    return 0;
  }
  switch (d.key) {

    case fan::key_enter: {
      const auto& new_name = pile->loco.text_box.get_text(d.cid);
      if (!does_stage_exist(new_name)) {
        do {
          const auto& old_name = get_selected_name(instances[stage_t::stage_instance].menu_id);

          #if defined(fan_platform_windows)
            static std::regex windows_filename_regex(R"([^\\/:*?"<>|\r\n]+(?!\\)$)");
          #elif defined(fan_platform_unix)
            static std::regex unix_filename_regex("[^/]+$");
          #endif

          if (new_name.empty() || 
            #if defined(fan_platform_windows)
              !std::regex_match(new_name, windows_filename_regex)
            #elif defined(fan_platform_unix)
              !std::regex_match(new_name, unix_filename_regex)
            #endif
            ) {
            fan::print("invalid stage name");
            break;
          }

          if (fan::io::file::rename(get_file_fullpath(old_name), get_file_fullpath(new_name)) ||
              fan::io::file::rename(get_file_fullpath_runtime(old_name), get_file_fullpath_runtime(new_name))) {

            fan::print_format("failed to rename file from:{} - to:{}", old_name, new_name);
            break;
          }
          if (old_name == new_name) {
            break;
          }
          std::regex rg(fan::format(R"(\b{0}(_t)?\b)", old_name));
          stage_h_str = std::regex_replace(stage_h_str, rg, new_name + R"($1)");
          write_stage();
          set_selected_name(instances[stage_t::stage_instance].menu_id, new_name);
        } while (0);
      }
      else {
        fan::print("stage name already exists");
      }
      pile->loco.text_box.erase(&rename_cid);
      return 1;
    }
  }
  return 0;
};

pile->loco.text_box.push_back(&rename_cid, tp);
pile->loco.menu_maker_button.set_selected(instances[stage_t::stage_options].menu_id, nullptr);
pile->loco.text_box.set_focus(&rename_cid);

#undef return_value