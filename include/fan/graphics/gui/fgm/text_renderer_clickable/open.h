trc.open(&pile->window, &pile->context);
trc.enable_draw(&pile->context);
trc.bind_matrices(&pile->context, &pile->editor.gui_matrices);