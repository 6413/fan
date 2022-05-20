pile->builder_draw_types.trc.erase(
            &pile->window,
            &pile->context,
            pile->editor_draw_types.selected_draw_type_index
);
for (uint32_t i = 0; i < pile->editor_draw_types.depth_map.size(); i++) {
  if (pile->editor_draw_types.depth_map[i].builder_draw_type == pile->editor_draw_types.selected_draw_type &&
      pile->editor_draw_types.depth_map[i].builder_draw_type_index == pile->editor_draw_types.selected_draw_type_index) {
    pile->editor_draw_types.depth_map.erase(i);
    for (uint32_t j = i; j < pile->editor_draw_types.depth_map.size(); j++) {
      pile->editor_draw_types.depth_map[j].builder_draw_type_index--;
    }
    pile->editor_draw_types.close_build_properties(pile);
    break;
  }
}