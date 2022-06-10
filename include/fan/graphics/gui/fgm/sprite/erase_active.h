case builder_draw_type_t::sprite: {
  pile->builder.sprite.erase(
             &pile->context,
             pile->editor.selected_type_index
  );
  pile->editor.editor_erase_active(pile);
  pile->editor.sprite_image_names.erase(pile->editor.sprite_image_names.begin() + pile->editor.selected_type_index);
  break;
}