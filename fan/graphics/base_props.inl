fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;
uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
uint32_t vertex_count = 6;
bool blending = true;