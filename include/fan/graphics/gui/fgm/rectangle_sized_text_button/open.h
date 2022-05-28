rtbs.open(&pile->window, &pile->context);
rtbs.enable_draw(&pile->window, &pile->context);
rtbs.bind_matrices(&pile->context, &pile->editor.gui_matrices);

rtbs_id_counter = 0;