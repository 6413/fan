case builder_draw_type_t::sprite: {
  pile->builder.sprite.erase(
             &pile->context,
             pile->editor.selected_type_index
  );
  pile->editor.editor_erase_active(pile);
  break;
}