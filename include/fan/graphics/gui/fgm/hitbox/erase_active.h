case builder_draw_type_t::hitbox: {
  pile->builder.hitbox.erase(
    &pile->context,
    pile->editor.selected_type_index
  );
  pile->editor.editor_erase_active(pile);
  break;
}