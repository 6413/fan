depth_map.open();
move_offset = 0;
properties_camera = 0;

pile->editor.selected_type = fan::uninitialized;

builder_viewport_size = constants::builder_viewport_size;
origin_shapes = fan::vec2(builder_viewport_size.x, 0);
origin_properties = fan::vec2(builder_viewport_size.x, builder_viewport_size.y / 2);

flags = 0;

fan::vec2 window_size = pile->window.get_size();

fan::graphics::viewport_t::properties_t vp;

vp.size = fan::vec2(window_size.x, window_size.y) - origin_properties;
vp.position = fan::vec2(origin_properties.x, 0);

properties_viewport.open(&pile->context);
properties_viewport.set(&pile->context, vp);
properties_viewport.enable(&pile->context);

properties_button.open(&pile->window, &pile->context);
properties_button.enable_draw(&pile->window, &pile->context);
properties_button.bind_matrices(&pile->context, &gui_properties_matrices);
properties_button.set_viewport_collision_offset(origin_properties);

properties_button_text.open(&pile->context);
properties_button_text.enable_draw(&pile->context);
properties_button_text.bind_matrices(&pile->context, &gui_properties_matrices);

vp.position = 0;
vp.size = fan::vec2(window_size.x, window_size.y);
builder_viewport.open(&pile->context);
builder_viewport.set(&pile->context, vp);
builder_viewport.enable(&pile->context);

resize_rectangles.open(&pile->window, &pile->context);
resize_rectangles.enable_draw(&pile->window, &pile->context);
resize_rectangles.bind_matrices(&pile->context, &gui_matrices);

outline.open(&pile->context);
decltype(outline)::properties_t line_p;
line_p.color = fan::colors::white;

line_p.src = fan::vec2(constants::builder_viewport_size.x, 0);
line_p.dst = constants::builder_viewport_size;
outline.push_back(&pile->context, line_p);
outline.bind_matrices(&pile->context, &gui_matrices);

line_p.src = fan::vec2(0, constants::builder_viewport_size.y);
line_p.dst = constants::builder_viewport_size;
outline.push_back(&pile->context, line_p);

line_p.src = fan::vec2(constants::builder_viewport_size.x, constants::builder_viewport_size.y / 2);
line_p.dst = fan::vec2(pile->window.get_size().x, constants::builder_viewport_size.y / 2);

outline.push_back(&pile->context, line_p);

outline.enable_draw(&pile->context);

// builder_viewport top right
fan::vec2 origin = fan::vec2(constants::builder_viewport_size.x, 0);

builder_types.open(&pile->window, &pile->context);
builder_types.enable_draw(&pile->window, &pile->context);

decltype(builder_types)::properties_t builder_types_p;
builder_types_p.font_size = constants::gui_size;
builder_types_p.size = fan::vec2(constants::gui_size * 4, constants::gui_size);
builder_types_p.position = origin + fan::vec2(50 + builder_types_p.size.x / 2, 50);
builder_types_p.text = "sprite";
builder_types_p.theme = fan_2d::graphics::gui::themes::gray();
builder_types.push_back(&pile->window, &pile->context, builder_types_p);
builder_types_p.position.y += 50;
builder_types_p.text = "text";
builder_types.push_back(&pile->window, &pile->context, builder_types_p);

builder_types_p.position.x -= 30;
builder_types_p.position.y += 170;
builder_types_p.size.x /= 2;
builder_types_p.size.y /= 1.2;
builder_types_p.text = "export";
builder_types.push_back(&pile->window, &pile->context, builder_types_p);

builder_types.bind_matrices(&pile->context, &gui_matrices);