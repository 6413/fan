auto resize_cb() {
  if (line.instances.empty()) {
    return;
  }
  fan::vec2 window_size = pile->loco.get_window()->get_size();
	fan::vec2 viewport_size = translate_viewport_position(fan::vec2(1, 1));
	viewport[viewport_area::global].set(
		pile->loco.get_context(),
		0,
		viewport_size,
		window_size
	);
	fan::vec2 ratio = viewport_size / viewport_size.max();
	matrices[viewport_area::global].set_ortho(
		&pile->loco,
		fan::vec2(-1, 1) * ratio.x,
		fan::vec2(-1, 1) * ratio.y
	);

	fan::vec2 viewport_position = translate_viewport_position(editor_position - editor_size);
	viewport_size = translate_viewport_position(editor_size + fan::vec2(-properties_line_position.x / 2 - 0.1));
	ratio = viewport_size / viewport_size.max();
	viewport[viewport_area::editor].set(
		pile->loco.get_context(),
		viewport_position,
		viewport_size,
		pile->loco.get_window()->get_size()
	);
	matrices[viewport_area::editor].set_ortho(
		&pile->loco,
		fan::vec2(editor_viewport[0], editor_viewport[1]) * ratio.x,
    fan::vec2(editor_viewport[2], editor_viewport[3]) * ratio.y
	);

	viewport_position = translate_viewport_position(fan::vec2(properties_line_position.x, -1));
	viewport_size = translate_viewport_position(fan::vec2(1, line_y_offset_between_types_and_properties)) - viewport_position;
	viewport[viewport_area::types].set(
		pile->loco.get_context(),
		viewport_position,
		viewport_size,
		pile->loco.get_window()->get_size()
	);

	ratio = viewport_size / viewport_size.max();
	matrices[viewport_area::types].set_ortho(
		&pile->loco,
		fan::vec2(-1, 1) * ratio.x,
		fan::vec2(-1, 1) * ratio.y
	);

	viewport_position.y += translate_viewport_position(fan::vec2(0, line_y_offset_between_types_and_properties)).y;
	viewport[viewport_area::properties].set(
		pile->loco.get_context(),
		viewport_position,
		viewport_size,
		pile->loco.get_window()->get_size()
	);

	ratio = viewport_size / viewport_size.max();
	matrices[viewport_area::properties].set_ortho(
		&pile->loco,
		fan::vec2(-1, 1) * ratio.x,
		fan::vec2(-1, 1) * ratio.y
	);

	fan::vec3 src, dst;
  src.z = line_z_depth;
  dst.z = line_z_depth;

	src = fan::vec3(editor_position - editor_size, src.z);
	dst.x = editor_position.x + editor_size.x;
	dst.y = src.y;

	pile->loco.line.set_line(
		&line.instances[0]->cid,
		src,
		dst
	);

	src = dst;
	dst.y = editor_position.y + editor_size.y;

	pile->loco.line.set_line(
		&line.instances[1]->cid,
		src,
		dst
	);

	src = dst;
	dst.x = editor_position.x - editor_size.x;

	pile->loco.line.set_line(
		&line.instances[2]->cid,
		src,
		dst
	);

	src = dst;
	dst.y = editor_position.y - editor_size.y;

	pile->loco.line.set_line(
		&line.instances[3]->cid,
		src,
		dst
	);

	src = fan::vec3(translate_to_global(
		viewport[viewport_area::types].get_position()
	), src.z);
	dst.x = src.x;
	dst.y = matrices[viewport_area::global].coordinates.down;

	pile->loco.line.set_line(
		&line.instances[4]->cid,
		src,
		dst
	);
	src = fan::vec3(translate_to_global(
		viewport[viewport_area::types].get_position() +
		fan::vec2(0, viewport[viewport_area::types].get_size().y)
	), src.z);
	dst = fan::vec3(translate_to_global(
		viewport[viewport_area::types].get_position() +
		viewport[viewport_area::types].get_size()
	), dst.z);

	pile->loco.line.set_line(
		&line.instances[5]->cid,
		src,
		dst
	);
};