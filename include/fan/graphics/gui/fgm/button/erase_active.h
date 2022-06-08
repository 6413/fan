case builder_draw_type_t::button: {
  pile->builder.sprite.erase(
             &pile->context,
             pile->editor.selected_type_index
  );
  pile->editor.editor_erase_active(pile);
  break;
}