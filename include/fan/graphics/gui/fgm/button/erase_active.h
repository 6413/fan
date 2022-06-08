case builder_draw_type_t::button: {
  pile->builder.button.erase(
              &pile->window,
             &pile->context,
             pile->editor.selected_type_index
  );
  pile->editor.editor_erase_active(pile);
  break;
}