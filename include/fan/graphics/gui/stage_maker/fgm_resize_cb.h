auto resize_cb() {
	pile_t* pile = OFFSETLESS(OFFSETLESS(this, stage_maker_t, fgm), pile_t, stage_maker);

  if (pile->stage_maker.fgm.line.instance.empty()) {
    return;
  }
  fan::vec2 window_size = get_loco()->get_window()->get_size();
	fan::vec2 viewport_size = pile->stage_maker.fgm.translate_viewport_position(fan::vec2(1, 1));
	pile->stage_maker.fgm.viewport[viewport_area::global].set(
		pile->loco.get_context(),
		0,
		viewport_size,
		window_size
	);
	fan::vec2 ratio = viewport_size / viewport_size.max();
	pile->stage_maker.fgm.matrices[viewport_area::global].set_ortho(
		&pile->loco,
		fan::vec2(-1, 1) * ratio.x,
		fan::vec2(-1, 1) * ratio.y
	);

	fan::vec2 viewport_position = pile->stage_maker.fgm.translate_viewport_position(pile->stage_maker.fgm.editor_position - pile->stage_maker.fgm.editor_size);
	viewport_size = pile->stage_maker.fgm.translate_viewport_position(pile->stage_maker.fgm.editor_size + fan::vec2(-pile->stage_maker.fgm.properties_line_position.x / 2 - 0.1));
	pile->stage_maker.fgm.viewport[viewport_area::editor].set(
		pile->loco.get_context(),
		viewport_position,
		viewport_size,
		pile->loco.get_window()->get_size()
	);
	ratio = viewport_size / viewport_size.max();
	pile->stage_maker.fgm.matrices[viewport_area::editor].set_ortho(
		&pile->loco,
		fan::vec2(-1, 1) * ratio.x,
		fan::vec2(-1, 1) * ratio.y
	);

	viewport_position = pile->stage_maker.fgm.translate_viewport_position(fan::vec2(pile->stage_maker.fgm.properties_line_position.x, -1));
	viewport_size = pile->stage_maker.fgm.translate_viewport_position(fan::vec2(1, pile->stage_maker.fgm.line_y_offset_between_types_and_properties)) - viewport_position;
	pile->stage_maker.fgm.viewport[viewport_area::types].set(
		pile->loco.get_context(),
		viewport_position,
		viewport_size,
		pile->loco.get_window()->get_size()
	);

	ratio = viewport_size / viewport_size.max();
	pile->stage_maker.fgm.matrices[viewport_area::types].set_ortho(
		&pile->loco,
		fan::vec2(-1, 1) * ratio.x,
		fan::vec2(-1, 1) * ratio.y
	);

	viewport_position.y += pile->stage_maker.fgm.translate_viewport_position(fan::vec2(0, pile->stage_maker.fgm.line_y_offset_between_types_and_properties)).y;
	pile->stage_maker.fgm.viewport[viewport_area::properties].set(
		pile->loco.get_context(),
		viewport_position,
		viewport_size,
		pile->loco.get_window()->get_size()
	);

	ratio = viewport_size / viewport_size.max();
	pile->stage_maker.fgm.matrices[viewport_area::properties].set_ortho(
		&pile->loco,
		fan::vec2(-1, 1) * ratio.x,
		fan::vec2(-1, 1) * ratio.y
	);

	fan::vec3 src, dst;

	src = pile->stage_maker.fgm.editor_position - pile->stage_maker.fgm.editor_size;
	dst.x = pile->stage_maker.fgm.editor_position.x + pile->stage_maker.fgm.editor_size.x;
	dst.y = src.y;

	src.z = dst.z = line_z_depth;
	pile->loco.line.set_line(
		&pile->stage_maker.fgm.line.instance[0]->cid,
		src,
		dst
	);

	src = dst;
	dst.y = pile->stage_maker.fgm.editor_position.y + pile->stage_maker.fgm.editor_size.y;

	src.z = dst.z = line_z_depth;
	pile->loco.line.set_line(
		&pile->stage_maker.fgm.line.instance[1]->cid,
		src,
		dst
	);

	src = dst;
	dst.x = pile->stage_maker.fgm.editor_position.x - pile->stage_maker.fgm.editor_size.x;

	src.z = dst.z = line_z_depth;
	pile->loco.line.set_line(
		&pile->stage_maker.fgm.line.instance[2]->cid,
		src,
		dst
	);

	src = dst;
	dst.y = pile->stage_maker.fgm.editor_position.y - pile->stage_maker.fgm.editor_size.y;

	src.z = dst.z = line_z_depth;
	pile->loco.line.set_line(
		&pile->stage_maker.fgm.line.instance[3]->cid,
		src,
		dst
	);

	src = pile->stage_maker.fgm.translate_to_global(
		pile->stage_maker.fgm.viewport[viewport_area::types].get_position()
	);
	dst.x = src.x;
	dst.y = pile->stage_maker.fgm.matrices[viewport_area::global].coordinates.bottom;
	src.z = dst.z = line_z_depth;
	pile->loco.line.set_line(
		&pile->stage_maker.fgm.line.instance[4]->cid,
		src,
		dst
	);
	src = pile->stage_maker.fgm.translate_to_global(
		pile->stage_maker.fgm.viewport[viewport_area::types].get_position() +
		fan::vec2(0, pile->stage_maker.fgm.viewport[viewport_area::types].get_size().y)
	);
	dst = pile->stage_maker.fgm.translate_to_global(
		pile->stage_maker.fgm.viewport[viewport_area::types].get_position() +
		pile->stage_maker.fgm.viewport[viewport_area::types].get_size()
	);
	src.z = dst.z = line_z_depth;
	pile->loco.line.set_line(
		&pile->stage_maker.fgm.line.instance[5]->cid,
		src,
		dst
	);
};