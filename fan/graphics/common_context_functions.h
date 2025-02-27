/*shader*/
.shader_create = [](context_t& context) { 
  context_shader_nr_t d;
  d.context_renderer = context_get.shader_create();
  return d; 
}, 
.shader_get = [](context_t& context, context_shader_nr_t nr) { 
  context_shader_t d;
  d.context_renderer = &context_get.shader_get(nr.context_renderer);
  return d;
}, 
.shader_erase = [](context_t& context, context_shader_nr_t nr) { 
  context_get.shader_erase(nr.context_renderer); 
}, 
.shader_use = [](context_t& context, context_shader_nr_t nr) { 
  context_get.shader_use(nr.context_renderer); 
}, 
.shader_set_vertex = [](context_t& context, context_shader_nr_t nr, const std::string& vertex_code) { 
  context_get.shader_set_vertex(nr.context_renderer, vertex_code); 
}, 
.shader_set_fragment = [](context_t& context, context_shader_nr_t nr, const std::string& fragment_code) { 
  context_get.shader_set_fragment(nr.context_renderer, fragment_code); 
}, 
.shader_compile = [](context_t& context, context_shader_nr_t nr) { 
  return context_get.shader_compile(nr.context_renderer); 
}, 
 /*image*/
.image_create = [](context_t& context) { 
  context_image_nr_t d;
  d.context_renderer = context_get.image_create();
  return d; 
}, 
.image_get = [](context_t& context, context_image_nr_t nr) {
  context_image_t d;
  d.context_renderer = context_get.image_get(nr.context_renderer);
  return d; 
}, 
.image_get_handle = [](context_t& context, context_image_nr_t nr) { 
  return (uint64_t)context_get.image_get_handle(nr.context_renderer); 
}, 
.image_erase = [](context_t& context, context_image_nr_t nr) { 
  context_get.image_erase(nr.context_renderer); 
}, 
.image_bind = [](context_t& context, context_image_nr_t nr) { 
  context_get.image_bind(nr.context_renderer); 
}, 
.image_unbind = [](context_t& context, context_image_nr_t nr) { 
  context_get.image_unbind(nr.context_renderer); 
}, 
.image_set_settings = [](context_t& context, const image_load_properties_t& settings) { 
  context_get.image_set_settings(settings); 
}, 
.image_load_info = [](context_t& context, const fan::image::image_info_t& image_info) { 
  context_image_nr_t d;
  d.context_renderer = context_get.image_load(image_info);
  return d; 
}, 
.image_load_info_props = [](context_t& context, const fan::image::image_info_t& image_info, const image_load_properties_t& p) { 
  context_image_nr_t d;
  d.context_renderer = context_get.image_load(image_info, p);
  return d; 
}, 
.image_load_path = [](context_t& context, const fan::string& path) { 
  context_image_nr_t d;
  d.context_renderer = context_get.image_load(path);
  return d;
}, 
.image_load_path_props = [](context_t& context, const fan::string& path, const image_load_properties_t& p) { 
  context_image_nr_t d;
  d.context_renderer = context_get.image_load(path, p);
  return d; 
}, 
.image_load_colors = [](context_t& context, fan::color* colors, const fan::vec2ui& size_) { 
  context_image_nr_t d;
  d.context_renderer = context_get.image_load(colors, size_);
  return d; 
}, 
.image_load_colors_props = [](context_t& context, fan::color* colors, const fan::vec2ui& size_, const image_load_properties_t& p) { 
  context_image_nr_t d;
  d.context_renderer = context_get.image_load(colors, size_, p);
  return d;
}, 
.image_unload = [](context_t& context, context_image_nr_t nr) { 
  context_get.image_unload(nr.context_renderer); 
}, 
.create_missing_texture = [](context_t& context) { 
  context_image_nr_t d;
  d.context_renderer = context_get.create_missing_texture();
  return d; 
}, 
.create_transparent_texture = [](context_t& context) { 
  context_image_nr_t d;
  d.context_renderer = context_get.create_transparent_texture();
  return d;
}, 
.image_reload_pixels = [](context_t& context, context_image_nr_t nr, const fan::image::image_info_t& image_info) { 
  return context_get.image_reload_pixels(nr.context_renderer, image_info); 
}, 
.image_reload_pixels_props = [](context_t& context, context_image_nr_t nr, const fan::image::image_info_t& image_info, const image_load_properties_t& p) { 
  return context_get.image_reload_pixels(nr.context_renderer, image_info, p); 
}, 
.image_create_color = [](context_t& context, const fan::color& color) { 
  context_image_nr_t d;
  d.context_renderer = context_get.image_create(color);
  return d;
}, 
.image_create_color_props = [](context_t& context, const fan::color& color, const image_load_properties_t& p) { 
  context_image_nr_t d;
  d.context_renderer = context_get.image_create(color, p);
  return d;
},
/*camera*/
.camera_create = [](context_t& context) { 
  context_camera_nr_t d;
  d.context_renderer = context_get.camera_create();
  return d; 
},
.camera_get = [](context_t& context, context_camera_nr_t nr) {
  context_camera_t d;
  d.context_renderer = &context_get.camera_get(nr.context_renderer);
  return d; 
},
.camera_erase = [](context_t& context, context_camera_nr_t nr) { 
  context_get.camera_erase(nr.context_renderer); 
},
.camera_open = [](context_t& context, const fan::vec2& x, const fan::vec2& y) {
  context_camera_nr_t d;
  d.context_renderer = context_get.camera_open(x, y);
  return d; 
},
.camera_get_position = [](context_t& context, context_camera_nr_t nr) { 
  return context_get.camera_get_position(nr.context_renderer); 
},
.camera_set_position = [](context_t& context, context_camera_nr_t nr, const fan::vec3& cp) { 
  context_get.camera_set_position(nr.context_renderer, cp); 
},
.camera_get_size = [](context_t& context, context_camera_nr_t nr) { 
  return context_get.camera_get_size(nr.context_renderer); 
},
.camera_set_ortho = [](context_t& context, context_camera_nr_t nr, fan::vec2 x, fan::vec2 y) { 
  context_get.camera_set_ortho(nr.context_renderer, x, y); 
},
.camera_set_perspective = [](context_t& context, context_camera_nr_t nr, f32_t fov, const fan::vec2& window_size) { 
  context_get.camera_set_perspective(nr.context_renderer, fov, window_size); 
},
.camera_rotate = [](context_t& context, context_camera_nr_t nr, const fan::vec2& offset) { 
  context_get.camera_rotate(nr.context_renderer, offset); 
},
/*viewport*/
.viewport_create = [](context_t& context) { 
  context_viewport_nr_t d;
  d.context_renderer = context_get.viewport_create();
  return d; 
},
.viewport_get = [](context_t& context, context_viewport_nr_t nr) { 
  context_viewport_t d;
  d.context_renderer = context_get.viewport_get(nr.context_renderer);
  return d; 
},
.viewport_erase = [](context_t& context, context_viewport_nr_t nr) { 
  context_get.viewport_erase(nr.context_renderer); 
},
.viewport_get_position = [](context_t& context, context_viewport_nr_t nr) { 
  return context_get.viewport_get_position(nr.context_renderer); 
},
.viewport_get_size = [](context_t& context, context_viewport_nr_t nr) { 
  return context_get.viewport_get_size(nr.context_renderer); 
},
.viewport_set = [](context_t& context, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) { 
  context_get.viewport_set(viewport_position_, viewport_size_, window_size); 
},
.viewport_set_nr = [](context_t& context, context_viewport_nr_t nr, const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) { 
  context_get.viewport_set(nr.context_renderer, viewport_position_, viewport_size_, window_size); 
},
.viewport_zero = [](context_t& context, context_viewport_nr_t nr) { 
  context_get.viewport_zero(nr.context_renderer); 
},
.viewport_inside = [](context_t& context, context_viewport_nr_t nr, const fan::vec2& position) { 
  return context_get.viewport_inside(nr.context_renderer, position); 
},
.viewport_inside_wir = [](context_t& context, context_viewport_nr_t nr, const fan::vec2& position) { 
  return context_get.viewport_inside_wir(nr.context_renderer, position); 
},

#undef context_renderer
#undef context_get