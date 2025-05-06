#!/usr/bin/env python3

shapes = [
    "sprite",
    "text",
    "hitbox",
    "line",
    "mark",
    "rectangle",
    "light",
    "unlit_sprite",
    "circle",
    "capsule",
    "polygon",
    "grid",
    "vfi",
    "particles",
    "universal_image_renderer",
    "gradient",
    "light_end",
    "shader_shape",
    "rectangle3d",
    "line3d",
]

functions = [
    #{"name": "push_back", "return_type": "loco_t::shape_t", "args": ["void* properties"], "member": ""},
    
    {"name": "get_position", "return_type": "fan::vec3", "args": ["loco_t::shape_t* shape"], "member": "position"},
    {"name": "set_position2", "return_type": "void", "args": ["loco_t::shape_t* shape", "const fan::vec2& position"], "member": "position"},
    {"name": "set_position3", "return_type": "void", "args": ["loco_t::shape_t* shape", "const fan::vec3& position"], "member": "position"},
    
    {"name": "get_size", "return_type": "fan::vec2", "args": ["loco_t::shape_t* shape"], "member": "size"},
    {"name": "get_size3", "return_type": "fan::vec3", "args": ["loco_t::shape_t* shape"], "member": "size"},
    {"name": "set_size", "return_type": "void", "args": ["loco_t::shape_t* shape", "const fan::vec2& size"], "member": "size"},
    {"name": "set_size3", "return_type": "void", "args": ["loco_t::shape_t* shape", "const fan::vec3& size"], "member": "size"},
    
    {"name": "get_rotation_point", "return_type": "fan::vec2", "args": ["loco_t::shape_t* shape"], "member": "rotation_point"},
    {"name": "set_rotation_point", "return_type": "void", "args": ["loco_t::shape_t* shape", "const fan::vec2& point"], "member": "rotation_point"},
    
    {"name": "get_color", "return_type": "fan::color", "args": ["loco_t::shape_t* shape"], "member": "color"},
    {"name": "set_color", "return_type": "void", "args": ["loco_t::shape_t* shape", "const fan::color& color"], "member": "color"},
    
    {"name": "get_angle", "return_type": "fan::vec3", "args": ["loco_t::shape_t* shape"], "member": "angle"},
    {"name": "set_angle", "return_type": "void", "args": ["loco_t::shape_t* shape", "const fan::vec3& angle"], "member": "angle"},
    
    {"name": "get_tc_position", "return_type": "fan::vec2", "args": ["loco_t::shape_t* shape"], "member": "tc_position"},
    {"name": "set_tc_position", "return_type": "void", "args": ["loco_t::shape_t* shape", "const fan::vec2& position"], "member": "tc_position"},
    
    {"name": "get_tc_size", "return_type": "fan::vec2", "args": ["loco_t::shape_t* shape"], "member": "tc_size"},
    {"name": "set_tc_size", "return_type": "void", "args": ["loco_t::shape_t* shape", "const fan::vec2& size"], "member": "tc_size"},
    
    {"name": "load_tp", "return_type": "bool", "args": ["loco_t::shape_t* shape", "loco_t::texturepack_t::ti_t* tp"], "member": ""},
    
    {"name": "get_grid_size", "return_type": "fan::vec2", "args": ["loco_t::shape_t* shape"], "member": "grid_size"},
    {"name": "set_grid_size", "return_type": "void", "args": ["loco_t::shape_t* shape", "const fan::vec2& size"], "member": "grid_size"},
    
    {"name": "get_camera", "return_type": "loco_t::camera_t", "args": ["loco_t::shape_t* shape"], "member": "camera"},
    {"name": "set_camera", "return_type": "void", "args": ["loco_t::shape_t* shape", "loco_t::camera_t camera"], "member": "camera"},
    
    {"name": "get_viewport", "return_type": "loco_t::viewport_t", "args": ["loco_t::shape_t* shape"], "member": "viewport"},
    {"name": "set_viewport", "return_type": "void", "args": ["loco_t::shape_t* shape", "loco_t::viewport_t viewport"], "member": "viewport"},
    
    {"name": "get_image", "return_type": "loco_t::image_t", "args": ["loco_t::shape_t* shape"], "member": "image"},
    {"name": "set_image", "return_type": "void", "args": ["loco_t::shape_t* shape", "loco_t::image_t image"], "member": "image"},
    
    {"name": "get_image_data", "return_type": "fan::graphics::image_data_t&", "args": ["loco_t::shape_t* shape"], "member": "image_data"},
    
    {"name": "get_parallax_factor", "return_type": "f32_t", "args": ["loco_t::shape_t* shape"], "member": "parallax_factor"},
    {"name": "set_parallax_factor", "return_type": "void", "args": ["loco_t::shape_t* shape", "f32_t factor"], "member": "parallax_factor"},
    
    #{"name": "get_rotation_vector", "return_type": "fan::vec3", "args": ["loco_t::shape_t* shape"], "member": "rotation_vector"},
    
    {"name": "get_flags", "return_type": "uint32_t", "args": ["loco_t::shape_t* shape"], "member": "flags"},
    {"name": "set_flags", "return_type": "void", "args": ["loco_t::shape_t* shape", "uint32_t flags"], "member": "flags"},
    
    {"name": "get_radius", "return_type": "f32_t", "args": ["loco_t::shape_t* shape"], "member": "radius"},
    {"name": "get_src", "return_type": "fan::vec3", "args": ["loco_t::shape_t* shape"], "member": "src"},
    {"name": "get_dst", "return_type": "fan::vec3", "args": ["loco_t::shape_t* shape"], "member": "dst"},
    {"name": "get_outline_size", "return_type": "f32_t", "args": ["loco_t::shape_t* shape"], "member": "outline_size"},
    {"name": "get_outline_color", "return_type": "fan::color", "args": ["loco_t::shape_t* shape"], "member": "outline_color"},
    
    #{"name": "reload", "return_type": "void", "args": ["loco_t::shape_t* shape", "uint8_t format", "void** image_data", "const fan::vec2& size", "uint32_t filter"], "member": ""},
    
    {"name": "draw", "return_type": "void", "args": ["uint8_t draw_range"], "member": ""},
    
    {"name": "set_line", "return_type": "void", "args": ["loco_t::shape_t* shape", "const fan::vec3& src", "const fan::vec3& dst"], "member": ""},
    {"name": "set_line3", "return_type": "void", "args": ["loco_t::shape_t* shape", "const fan::vec3& src", "const fan::vec3& dst"], "member": ""},
]

exclusions = {
    "get_grid_size": ["sprite", "text", "line", "rectangle", "light", "unlit_sprite", 
                      "circle", "capsule", "polygon", "particles", "universal_image_renderer", "gradient", 
                      "shader_shape", "rectangle3d", "line3d"],
    "set_grid_size": ["sprite", "text", "line", "rectangle", "light", "unlit_sprite", 
                      "circle", "capsule", "polygon", "particles", "universal_image_renderer", "gradient", 
                      "shader_shape", "rectangle3d", "line3d"],
    
    "get_radius": ["sprite", "text", "line", "rectangle", "light", "unlit_sprite", 
                   "polygon", "grid", "vfi", "particles", "universal_image_renderer", "gradient", 
                   "shader_shape", "rectangle3d", "line3d"],
    
    "set_line": ["sprite", "text", "rectangle", "light", "unlit_sprite", 
                 "circle", "capsule", "polygon", "grid", "vfi", "particles", 
                 "universal_image_renderer", "gradient", "shader_shape", "rectangle3d"],
    "set_line3": ["sprite", "text", "line", "rectangle", "light", "unlit_sprite", 
                  "circle", "capsule", "polygon", "grid", "vfi", "particles", 
                  "universal_image_renderer", "gradient", "shader_shape"],
    
    "set_position3": ["text", "polygon", "grid", "vfi", 
                      "particles", "universal_image_renderer"],
    "get_size3": ["sprite", "text", "line", "rectangle", "light", 
                 "unlit_sprite", "circle", "capsule", "polygon", "grid", "vfi", 
                 "particles", "universal_image_renderer", "gradient", "shader_shape", "line3d"],
    "set_size3": ["sprite", "text", "line", "rectangle", "light", 
                 "unlit_sprite", "circle", "capsule", "polygon", "grid", "vfi", 
                 "particles", "universal_image_renderer", "gradient", "shader_shape", "line3d"],
    "get_size" : ["text", "polygon", "vfi", "line", "line3d"],
    
    "get_tc_position": ["line", "light", "circle", "capsule", 
                        "line3d", "rectangle3d", "rectangle", "grid", "polygon", "gradient"],
    "set_tc_position": ["line", "light", "circle", "capsule", 
                        "line3d", "rectangle3d", "rectangle", "grid", "polygon", "gradient"],
    "get_tc_size": ["line", "light", "circle", "capsule", 
                    "line3d", "rectangle3d", "rectangle", "grid", "polygon", "gradient"],
    "set_tc_size": ["line", "light", "circle", "capsule", 
                    "line3d", "rectangle3d", "rectangle", "grid", "polygon", "gradient"],
    
    "get_image": ["line", "light", "circle", "capsule", "line3d"],
    "set_image": ["line", "light", "circle", "capsule", "line3d"],
    "get_image_data": ["line", "light", "circle", "capsule", "line3d"],
    
    "get_outline_size": ["text", "light", "vfi", "particles", "gradient"],
    "get_outline_color": ["text", "light", "vfi", "particles", "gradient"],

    "get_src": ["sprite", "text", "rectangle", "light", "unlit_sprite", 
                "circle", "capsule", "polygon", "grid", "vfi", "particles", 
                "universal_image_renderer", "gradient", "shader_shape", "rectangle3d"],
    "get_dst": ["sprite", "text", "rectangle", "light", "unlit_sprite", 
                "circle", "capsule", "polygon", "grid", "vfi", "particles", 
                "universal_image_renderer", "gradient", "shader_shape", "rectangle3d"],
    "get_position" : ["text", "vfi"],
    "set_position2" : ["text", "vfi", "particles", "line3d", "rectangle3d"],
    "set_size": ["line", "line3d", "polygon"],
    "get_rotation_point" : ["line", "line3d", "polygon", "universal_image_renderer", "rectangle3d"],
    "set_rotation_point" : ["line", "line3d", "polygon", "universal_image_renderer", "rectangle3d"],
    "get_color" : ["polygon", "universal_image_renderer"],
    "set_color" : ["polygon", "universal_image_renderer"],
    "get_angle" : ["line", "line3d", "polygon", "universal_image_renderer"],
    "set_angle" : ["line", "line3d", "polygon", "universal_image_renderer"],
    
    "get_parallax_factor" : ["line", "line3d", "text", "rectangle", "light", "unlit_sprite", 
                "circle", "capsule", "polygon", "grid", "vfi", "particles", 
                "universal_image_renderer", "gradient", "shader_shape", "rectangle3d"],
    "set_parallax_factor" : ["line", "line3d", "text", "rectangle", "light", "unlit_sprite", 
                "circle", "capsule", "polygon", "grid", "vfi", "particles", 
                "universal_image_renderer", "gradient", "shader_shape", "rectangle3d"],
    
    "get_flags": ["line", "line3d", "text", "rectangle", "light", "unlit_sprite", 
                "polygon", "grid", "vfi", "particles", 
                "universal_image_renderer", "gradient", "shader_shape", "rectangle3d"],
    "set_flags": ["line", "line3d", "text", "rectangle", "light", "unlit_sprite", 
                "polygon", "grid", "vfi", "particles", 
                "universal_image_renderer", "gradient", "shader_shape", "rectangle3d"],
                
    "get_outline_color": ["line", "line3d","sprite", "text", "rectangle", "light", "unlit_sprite", 
        "circle", "capsule", "polygon", "grid", "vfi", "particles", 
        "universal_image_renderer", "gradient", "shader_shape", "rectangle3d"],
    "set_outline_color": ["line", "line3d","sprite", "text", "rectangle", "light", "unlit_sprite", 
        "circle", "capsule", "polygon", "grid", "vfi", "particles", 
        "universal_image_renderer", "gradient", "shader_shape", "rectangle3d"],
    "get_outline_size": ["line", "line3d","sprite", "text", "rectangle", "light", "unlit_sprite", 
        "circle", "capsule", "polygon", "grid", "vfi", "particles", 
        "universal_image_renderer", "gradient", "shader_shape", "rectangle3d"],
    "set_outline_size": ["line", "line3d","sprite", "text", "rectangle", "light", "unlit_sprite", 
        "circle", "capsule", "polygon", "grid", "vfi", "particles", 
        "universal_image_renderer", "gradient", "shader_shape", "rectangle3d"],
        
    "get_image" : ["line", "line3d", "text", "rectangle", "light", "unlit_sprite", 
        "polygon", "grid", "vfi", "particles", 
        "gradient", "shader_shape", "rectangle3d"],
    "set_image" : ["line", "line3d", "text", "rectangle", "light", "unlit_sprite", 
        "polygon", "grid", "vfi", 
        "gradient", "shader_shape", "rectangle3d"],
    "push_back": ["hitbox", "mark", "light_end"]
        
    #"get_rotation_vector" : ["line", "line3d", "text", "rectangle", "light", "unlit_sprite", 
     #   "polygon", "grid", "vfi", "particles", 
      #  "gradient", "shader_shape", "rectangle3d", "capsule", "circle", "universal_image_renderer", "sprite"], # deprecated
}

for func in functions:
    name = func["name"]
    if name not in exclusions:
        exclusions[name] = []
    exclusions[name].extend(["light_end", "hitbox", "mark", "text", "vfi", "particles"])

custom_implementations = {

    ("set_line", "line"): """  auto data = reinterpret_cast<loco_t::line_t::vi_t*>(shape->GetRenderData(gloco->shaper));
  data->src = fan::vec3(src.x, src.y, 0);
  data->dst = fan::vec3(dst.x, dst.y, 0);
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::line3d_t::vi_t::src),
      sizeof(loco_t::line3d_t::vi_t::src)
    );
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::line3d_t::vi_t::dst),
      sizeof(loco_t::line3d_t::vi_t::dst)
    );
  }
""",
    ("get_position", "line"): """  return reinterpret_cast<loco_t::line_t::vi_t*>(shape->GetRenderData(gloco->shaper))->src;
""",
    ("set_position2", "line"): """  reinterpret_cast<loco_t::line_t::vi_t*>(shape->GetRenderData(gloco->shaper))->src = position;
""",

("set_line", "line3d"): """  auto data = reinterpret_cast<loco_t::line3d_t::vi_t*>(shape->GetRenderData(gloco->shaper));
  data->src = fan::vec3(src.x, src.y, 0);
  data->dst = fan::vec3(dst.x, dst.y, 0);
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::line3d_t::vi_t::src),
      sizeof(loco_t::line3d_t::vi_t::src)
    );
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::line3d_t::vi_t::dst),
      sizeof(loco_t::line3d_t::vi_t::dst)
    );
  }
""",
    ("get_position", "line3d"): """  return reinterpret_cast<loco_t::line3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->src;
""",
    ("set_position3", "line3d"): """  reinterpret_cast<loco_t::line3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->src = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::line3d_t::vi_t::src),
      sizeof(loco_t::line3d_t::vi_t::src)
    );
  }
""",

    ("get_position", "polygon"): """  auto ri = (loco_t::polygon_t::ri_t*)shape->GetData(gloco->shaper);
  fan::vec3 position = 0;
  fan::opengl::core::get_glbuffer(
   gloco->context.gl,
   &position,
   ri->vbo.m_buffer,
   sizeof(position),
   sizeof(loco_t::polygon_vertex_t) * 0 + fan::member_offset(&loco_t::polygon_vertex_t::offset),
   ri->vbo.m_target
  );
  return position;
""",
    ("set_position2", "polygon"): """  auto ri = (loco_t::polygon_t::ri_t*)shape->GetData(gloco->shaper);
  ri->vao.bind(gloco->context.gl);
  ri->vbo.bind(gloco->context.gl);
  uint32_t vertex_count = ri->buffer_size / sizeof(loco_t::polygon_vertex_t);
  for (uint32_t i = 0; i < vertex_count; ++i) {
    fan::opengl::core::edit_glbuffer(
     gloco->context.gl, 
     ri->vbo.m_buffer, 
     &position, 
     sizeof(loco_t::polygon_vertex_t) * i + fan::member_offset(&loco_t::polygon_vertex_t::offset),
     sizeof(position),
     ri->vbo.m_target
    );
  }
""",
("set_angle", "polygon"): """  auto ri = (loco_t::polygon_t::ri_t*)shape->GetData(gloco->shaper);
	ri->vao.bind(gloco->context.gl);
	ri->vbo.bind(gloco->context.gl);
	uint32_t vertex_count = ri->buffer_size / sizeof(loco_t::polygon_vertex_t);
	for (uint32_t i = 0; i < vertex_count; ++i) {
      fan::opengl::core::edit_glbuffer(
       gloco->context.gl, 
       ri->vbo.m_buffer, 
       &angle, 
       sizeof(loco_t::polygon_vertex_t) * i + fan::member_offset(&loco_t::polygon_vertex_t::angle),
       sizeof(angle),
       ri->vbo.m_target
      );
	}
""",
    ("get_size", "circle"): """  return reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->radius;
""",
    ("get_size", "capsule"): """  return reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->radius;
""",
    ("get_position", "particles"): """  return reinterpret_cast<loco_t::particles_t::ri_t*>(shape->GetData(gloco->shaper))->position;
""",
    ("get_size", "particles"): """  return reinterpret_cast<loco_t::particles_t::ri_t*>(shape->GetData(gloco->shaper))->size;
""",
    ("set_size", "circle"): """  reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetData(gloco->shaper))->radius = size.x;
if (gloco->window.renderer == loco_t::renderer_t::opengl) {
  auto& data = gloco->shaper.ShapeList[*shape];
  gloco->shaper.ElementIsPartiallyEdited(
    data.sti,
    data.blid,
    data.ElementIndex,
    fan::member_offset(&loco_t::circle_t::vi_t::radius),
    sizeof(loco_t::circle_t::vi_t::radius)
  );
}
""",
    ("set_size", "capsule"): """  reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetData(gloco->shaper))->radius = size.x;
if (gloco->window.renderer == loco_t::renderer_t::opengl) {
  auto& data = gloco->shaper.ShapeList[*shape];
  gloco->shaper.ElementIsPartiallyEdited(
    data.sti,
    data.blid,
    data.ElementIndex,
    fan::member_offset(&loco_t::capsule_t::vi_t::radius),
    sizeof(loco_t::capsule_t::vi_t::radius)
  );
}
""",
    ("set_position3", "line"): """  reinterpret_cast<loco_t::line_t::vi_t*>(shape->GetData(gloco->shaper))->src = position;
if (gloco->window.renderer == loco_t::renderer_t::opengl) {
  auto& data = gloco->shaper.ShapeList[*shape];
  gloco->shaper.ElementIsPartiallyEdited(
    data.sti,
    data.blid,
    data.ElementIndex,
    fan::member_offset(&loco_t::line_t::vi_t::src),
    sizeof(loco_t::line_t::vi_t::src)
  );
}
""",
    ("reload", "universal_image_renderer"): """    loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)shape->GetData(gloco->shaper);
  if (format != ri.format) {
    auto sti = gloco->shaper.ShapeList[*shape].sti;
    uint8_t* KeyPack = gloco->shaper.GetKeys(*shape);
    loco_t::image_t vi_image = shaper_get_key_safe(loco_t::image_t, texture_t, image);


    auto shader = gloco->shaper.GetShader(sti);
    gloco->shader_set_vertex(
      shader,
      loco_t::read_shader("shaders/opengl/2D/objects/pixel_format_renderer.vs")
    );
    {
      std::string fs;
      switch (format) {
      case fan::pixel_format::yuv420p: {
        fs = loco_t::read_shader("shaders/opengl/2D/objects/yuv420p.fs");
        break;
      }
      case fan::pixel_format::nv12: {
        fs = loco_t::read_shader("shaders/opengl/2D/objects/nv12.fs");
        break;
      }
      default: {
        fan::throw_error("unimplemented format");
      }
      }
      gloco->shader_set_fragment(shader, fs);
      gloco->shader_compile(shader);
    }

    uint8_t image_count_old = fan::pixel_format::get_texture_amount(ri.format);
    uint8_t image_count_new = fan::pixel_format::get_texture_amount(format);
    if (image_count_new < image_count_old) {
      // -1 ? 
      for (uint32_t i = image_count_old - 1; i > image_count_new; --i) {
        if (i == 0) {
          gloco->image_erase(vi_image);
        }
        else {
          gloco->image_erase(ri.images_rest[i - 1]);
        }
      }
    }
    else if (image_count_new > image_count_old) {
      loco_t::image_t images[4];
      for (uint32_t i = image_count_old; i < image_count_new; ++i) {
        images[i] = gloco->image_create();
      }
      shape->set_image(images[0]);
      std::copy(&images[1], &images[0] + ri.images_rest.size(), ri.images_rest.data());
    }
  }

  auto vi_image = shape->get_image();

  uint8_t image_count_new = fan::pixel_format::get_texture_amount(format);
  for (uint32_t i = 0; i < image_count_new; i++) {
    fan::image::image_info_t image_info;
    image_info.data = image_data[i];
    image_info.size = fan::pixel_format::get_image_sizes(format, image_size)[i];
    auto lp = fan::pixel_format::get_image_properties<loco_t::image_load_properties_t>(format)[i];
    lp.min_filter = filter;
    lp.mag_filter = filter;
    if (i == 0) {
      gloco->image_reload(
        vi_image,
        image_info,
        lp
      );
    }
    else {
      gloco->image_reload(
        ri.images_rest[i - 1],
        image_info,
        lp
      );
    }
  }
  ri.format = format;
""",
}

for shape in shapes:
    custom_implementations[("get_camera", shape)] = """	return loco_t::get_camera(shape);
"""
    custom_implementations[("set_camera", shape)] = """	loco_t::set_camera(shape, camera);
"""
    
    custom_implementations[("get_viewport", shape)] = """	return loco_t::get_viewport(shape);
"""
    custom_implementations[("set_viewport", shape)] = """	loco_t::set_viewport(shape, viewport);
"""
    
    custom_implementations[("get_image", shape)] = """	return loco_t::get_image(shape);
"""
    custom_implementations[("set_image", shape)] = """	loco_t::set_image(shape, image);
"""
    custom_implementations[("get_image_data", shape)] = """	return gloco->image_list[shape->get_image()];
"""


def extract_param_name(arg_str):
    parts = arg_str.strip().split()
    if len(parts) >= 2:
        param = parts[-1]
        if param.endswith('&'):
            param = param[:-1]
        return param
    return ""

def generate_function_impl(func, shape):
    func_name = f"{func['name']}_{shape}"
    
    if func["name"] in exclusions and shape in exclusions[func["name"]]:
        # args_str = ", ".join(func["args"])
        # signature = f"static {func['return_type']} {func_name}({args_str})"
        
        # body = "{\n"
        # if func["return_type"] != "void":
            # if func["return_type"] == "bool":
                # body += "  return false;\n"
            # elif func["return_type"] == "f32_t":
                # body += "  return 0.0f;\n"
            # elif func["return_type"].startswith("fan::"):
                # body += f"  return {func['return_type']}{{}};\n"
            # elif func["return_type"].endswith("&"):
                # body += "  static auto dummy = {};\n  return dummy;\n"
            # else:
                # body += f"  return {func['return_type']}{{}};\n"
        # body += "  // Function excluded for this shape\n"
        # body += "}\n"
        # return signature + " " + body
        return ""
    
    custom_key = (func["name"], shape)
    
    args_str = ", ".join(func["args"])
    signature = f"static {func['return_type']} {func_name}({args_str})"
    
    body = "{\n"
    
    if custom_key in custom_implementations:
        # Use custom implementation
        body += custom_implementations[custom_key]
        if func["return_type"] != "void" and not any(return_line.strip().startswith("return ") for return_line in custom_implementations[custom_key].split('\n')):
            # Add default return if custom implementation doesn't have one
            if func["return_type"] == "bool":
                body += "  return false;\n"
            elif func["return_type"] == "f32_t":
                body += "  return 0.0f;\n"
            elif func["return_type"].startswith("fan::"):
                body += f"  return {func['return_type']}{{}};\n"
            elif func["return_type"].endswith("&"):
                body += "  static auto dummy = {};\n  return dummy;\n"
            else:
                body += f"  return {func['return_type']}{{}};\n"
    elif func["name"].startswith("get_") and func["member"]:
        body += f"  return reinterpret_cast<loco_t::{shape}_t::vi_t*>(shape->GetRenderData(gloco->shaper))->{func['member']};\n"
    elif func["name"].startswith("set_") and func["member"]:
        # Extract the parameter name (last argument)
        param_name = extract_param_name(func["args"][-1])
        if param_name:
            if func["name"] == "set_position3":
                body += f"  loco_t::set_position(shape, {param_name});\n  reinterpret_cast<loco_t::{shape}_t::vi_t*>(shape->GetRenderData(gloco->shaper))->{func['member']} = {param_name};\n"
            else:
                body += f"  reinterpret_cast<loco_t::{shape}_t::vi_t*>(shape->GetRenderData(gloco->shaper))->{func['member']} = {param_name};\n"
            
            body += """  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
"""
            body += f"      fan::member_offset(&loco_t::{shape}_t::vi_t::{func['member']}),\n"
            body += f"      sizeof(loco_t::{shape}_t::vi_t::{func['member']})\n    );\n  "
            body += "}\n"
        else:
            body += f"  // WARNING: Could not extract parameter name\n"
    else:
        if func["return_type"] != "void":
            if func["return_type"] == "bool":
                body += "  return false;\n"
            elif func["return_type"] == "f32_t":
                body += "  return 0.0f;\n"
            elif func["return_type"].startswith("fan::"):
                body += f"  return {func['return_type']}{{}};\n"
            elif func["return_type"].endswith("&"):
                body += "  static auto dummy = {};\n  return dummy;\n"
            else:
                body += f"  return {func['return_type']}{{}};\n"
    
    body += "}\n"
    return signature + " " + body

def generate_function_array(func):
    array_name = f"{func['name']}_functions"
    array_type = f"{func['name']}_cb"
    
    array_str = f"inline static loco_t::{array_type} {array_name}[] = {{\n"
    for shape in shapes:
        if func["name"] in exclusions and shape in exclusions[func["name"]]:
            array_str += f"  nullptr,\n"
        else:
            array_str += f"  &{func['name']}_{shape},\n"
    array_str += "};\n\n"
    return array_str

def generate_function_getter():
    func_str = "loco_t::functions_t get_shape_functions(uint16_t type) {\n"
    func_str += "  uint16_t index = type;\n"
    func_str += "  loco_t::functions_t funcs{};\n\n"
    
    for func in functions:
        func_str += f"  funcs.{func['name']} = {func['name']}_functions[index];\n"
        
    func_str += f"  funcs.push_back = push_back_functions[index];\n"
    
    func_str += "\n  return funcs;\n}\n"
    return func_str

def generate_position():
    return """
inline static void set_position(loco_t::shape_t* shape, const fan::vec3& position) {
   // alloc can be avoided inside switch
  auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
  uint8_t* KeyPack = new uint8_t[KeyPackSize];
  gloco->shaper.WriteKeys(*shape, KeyPack);
  auto sti = shape->get_shape_type();
  switch (sti) {       
  case loco_t::shape_type_t::light: {
    break;
  }
  // common
  case loco_t::shape_type_t::capsule:
  case loco_t::shape_type_t::gradient:
  case loco_t::shape_type_t::grid:
  case loco_t::shape_type_t::circle:
  case loco_t::shape_type_t::rectangle:
  case loco_t::shape_type_t::rectangle3d:
  case loco_t::shape_type_t::line: {
    shaper_get_key_safe(depth_t, common_t, depth) = position.z;
    break;
  }
                                  // texture
  case loco_t::shape_type_t::particles:
  case loco_t::shape_type_t::universal_image_renderer:
  case loco_t::shape_type_t::unlit_sprite:
  case loco_t::shape_type_t::sprite: {
    shaper_get_key_safe(depth_t, texture_t, depth) = position.z;
    break;
  }
  default: {
    fan::throw_error("unimplemented");
  }
  }


  auto _vi = shape->GetRenderData(gloco->shaper);
  auto vlen = gloco->shaper.GetRenderDataSize(sti);
  uint8_t* vi = new uint8_t[vlen];
  std::memcpy(vi, _vi, vlen);

  auto _ri = shape->GetData(gloco->shaper);
  auto rlen = gloco->shaper.GetDataSize(sti);
  uint8_t* ri = new uint8_t[rlen];
  std::memcpy(ri, _ri, rlen);

  shape->remove();
  *shape = gloco->shaper.add(
    sti,
    KeyPack,
    KeyPackSize,
    vi,
    ri
  );
#if defined(debug_shape_t)
  fan::print("+", shape->NRI);
#endif
  delete[] KeyPack;
  delete[] vi;
  delete[] ri;
}
"""

def generate_camera():
    return """
inline static loco_t::camera_t get_camera(loco_t::shape_t* shape) {
  auto sti = shape->get_shape_type();

  // alloc can be avoided inside switch
  uint8_t* KeyPack = gloco->shaper.GetKeys(*shape);
  
  switch (sti) {
    // light
  case loco_t::shape_type_t::light: {
    return shaper_get_key_safe(camera_t, light_t, camera);
  }
                                  // common
  case loco_t::shape_type_t::capsule:
  case loco_t::shape_type_t::gradient:
  case loco_t::shape_type_t::grid:
  case loco_t::shape_type_t::circle:
  case loco_t::shape_type_t::rectangle:
  case loco_t::shape_type_t::line: {
    return shaper_get_key_safe(camera_t, common_t, camera);
  }
                                  // texture
  case loco_t::shape_type_t::particles:
  case loco_t::shape_type_t::universal_image_renderer:
  case loco_t::shape_type_t::unlit_sprite:
  case loco_t::shape_type_t::sprite: {
    return shaper_get_key_safe(camera_t, texture_t, camera);
  }
  default: {
    fan::throw_error("unimplemented");
  }
  }
  return loco_t::camera_t();
}

inline static void set_camera(loco_t::shape_t* shape, loco_t::camera_t camera) {
   // alloc can be avoided inside switch
  auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
  uint8_t* KeyPack = new uint8_t[KeyPackSize];
  gloco->shaper.WriteKeys(*shape, KeyPack);
  auto sti = shape->get_shape_type();
  switch(sti) {
  // light
  case loco_t::shape_type_t::light: {
    shaper_get_key_safe(camera_t, light_t, camera) = camera;
    break;
  }
  // common
  case loco_t::shape_type_t::capsule:
  case loco_t::shape_type_t::gradient:
  case loco_t::shape_type_t::grid:
  case loco_t::shape_type_t::circle:
  case loco_t::shape_type_t::rectangle:
  case loco_t::shape_type_t::rectangle3d:
  case loco_t::shape_type_t::line: {
    shaper_get_key_safe(camera_t, common_t, camera) = camera;
    break;
  }
  // texture
  case loco_t::shape_type_t::particles:
  case loco_t::shape_type_t::universal_image_renderer:
  case loco_t::shape_type_t::unlit_sprite:
  case loco_t::shape_type_t::sprite: {
    shaper_get_key_safe(camera_t, texture_t, camera) = camera;
    break;
  }
  default: {
    fan::throw_error("unimplemented");
  }
  }
  
  auto _vi = shape->GetRenderData(gloco->shaper);
  auto vlen = gloco->shaper.GetRenderDataSize(sti);
  uint8_t* vi = new uint8_t[vlen];
  std::memcpy(vi, _vi, vlen);
  
  auto _ri = shape->GetData(gloco->shaper);
  auto rlen = gloco->shaper.GetDataSize(sti);
  uint8_t* ri = new uint8_t[rlen];
  std::memcpy(ri, _ri, rlen);
  
  shape->remove();
  *shape = gloco->shaper.add(
  sti,
  KeyPack,
  KeyPackSize,
  vi,
  ri
  );
  #if defined(debug_shape_t)
  fan::print("+", shape->NRI);
  #endif
  delete[] KeyPack;
  delete[] vi;
  delete[] ri;
}
"""

def generate_viewport():
    return """inline static loco_t::viewport_t get_viewport(loco_t::shape_t* shape) {
  uint8_t* KeyPack = gloco->shaper.GetKeys(*shape);

  auto sti = shape->get_shape_type();

  switch(sti) {
    // light
    case loco_t::shape_type_t::light: {
      return shaper_get_key_safe(viewport_t, light_t, viewport);
    }
    // common
    case loco_t::shape_type_t::capsule:
    case loco_t::shape_type_t::gradient:
    case loco_t::shape_type_t::grid:
    case loco_t::shape_type_t::circle:
    case loco_t::shape_type_t::rectangle:
    case loco_t::shape_type_t::line: {
      return shaper_get_key_safe(viewport_t, common_t, viewport);
    }
    // texture
    case loco_t::shape_type_t::particles:
    case loco_t::shape_type_t::universal_image_renderer:
    case loco_t::shape_type_t::unlit_sprite:
    case loco_t::shape_type_t::sprite: {
      return shaper_get_key_safe(viewport_t, texture_t, viewport);
    }
    default: {
      fan::throw_error("unimplemented");
    }
  }
  __unreachable();
}

inline static void set_viewport(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
  auto sti = shape->get_shape_type();

  // alloc can be avoided inside switch
  auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
  uint8_t* KeyPack = new uint8_t[KeyPackSize];
  gloco->shaper.WriteKeys(*shape, KeyPack);
  switch (sti) {
    // light
  case loco_t::shape_type_t::light: {
    shaper_get_key_safe(viewport_t, light_t, viewport) = viewport;
    break;
  }
                                  // common
  case loco_t::shape_type_t::capsule:
  case loco_t::shape_type_t::gradient:
  case loco_t::shape_type_t::grid:
  case loco_t::shape_type_t::circle:
  case loco_t::shape_type_t::rectangle:
  case loco_t::shape_type_t::line: {
    shaper_get_key_safe(viewport_t, common_t, viewport) = viewport;
    break;
  }
                                 // texture
  case loco_t::shape_type_t::particles:
  case loco_t::shape_type_t::universal_image_renderer:
  case loco_t::shape_type_t::unlit_sprite:
  case loco_t::shape_type_t::sprite: {
    shaper_get_key_safe(viewport_t, texture_t, viewport) = viewport;
    break;
  }
  default: {
    fan::throw_error("unimplemented");
  }
  }

  auto _vi = shape->GetRenderData(gloco->shaper);
  auto vlen = gloco->shaper.GetRenderDataSize(sti);
  uint8_t* vi = new uint8_t[vlen];
  std::memcpy(vi, _vi, vlen);

  auto _ri = shape->GetData(gloco->shaper);
  auto rlen = gloco->shaper.GetDataSize(sti);
  uint8_t* ri = new uint8_t[rlen];
  std::memcpy(ri, _ri, rlen);

  shape->remove();
  *shape = gloco->shaper.add(
    sti,
    KeyPack,
    KeyPackSize,
    vi,
    ri
  );
#if defined(debug_shape_t)
  fan::print("+", shape->NRI);
#endif
  delete[] KeyPack;
  delete[] vi;
  delete[] ri;
}
"""

def generate_image():
    return """inline static loco_t::image_t get_image(loco_t::shape_t* shape) {
  auto sti = gloco->shaper.ShapeList[*shape].sti;
  uint8_t* KeyPack = gloco->shaper.GetKeys(*shape);
  switch (sti) {
  // texture
  case loco_t::shape_type_t::particles:
  case loco_t::shape_type_t::universal_image_renderer:
  case loco_t::shape_type_t::unlit_sprite:
  case loco_t::shape_type_t::sprite: {
    return shaper_get_key_safe(image_t, texture_t, image);
  }
  default: {
    fan::throw_error("unimplemented");
  }
  }
  return loco_t::image_t();
}

inline static void set_image(loco_t::shape_t* shape, loco_t::image_t image) {
         
  auto sti = gloco->shaper.ShapeList[*shape].sti;

  // alloc can be avoided inside switch
  auto KeyPackSize = gloco->shaper.GetKeysSize(*shape);
  uint8_t* KeyPack = new uint8_t[KeyPackSize];
  gloco->shaper.WriteKeys(*shape, KeyPack);
  switch (sti) {
  // texture
  case loco_t::shape_type_t::particles:
  case loco_t::shape_type_t::universal_image_renderer:
  case loco_t::shape_type_t::unlit_sprite:
  case loco_t::shape_type_t::sprite: 
  case loco_t::shape_type_t::shader_shape:
  {
    shaper_get_key_safe(image_t, texture_t, image) = image;
    break;
  }
  default: {
    fan::throw_error("unimplemented");
  }
  }
            
  auto _vi = shape->GetRenderData(gloco->shaper);
  auto vlen = gloco->shaper.GetRenderDataSize(sti);
  uint8_t* vi = new uint8_t[vlen];
  std::memcpy(vi, _vi, vlen);

  auto _ri = shape->GetData(gloco->shaper);
  auto rlen = gloco->shaper.GetDataSize(sti);
  uint8_t* ri = new uint8_t[rlen];
  std::memcpy(ri, _ri, rlen);

  shape->remove();
  *shape = gloco->shaper.add(
    sti,
    KeyPack,
    KeyPackSize,
    vi,
    ri
  );
#if defined(debug_shape_t)
  fan::print("+", shape->NRI);
#endif
  delete[] KeyPack;
  delete[] vi;
  delete[] ri;
}
"""

def generate_push_backs():
    ret_str = ""
    for shape in shapes:
        if shape == "mark" or shape == "hitbox" or shape == "light_end":
            continue
        ret_str += f"static loco_t::shape_t push_back_{shape}(void* properties)"
        ret_str += " {\n"
        ret_str += f"  return gloco->{shape}.push_back(*reinterpret_cast<loco_t::{shape}_t::properties_t*>(properties));"
        ret_str += "\n}\n\n"
    return ret_str

def generate_code():
    code = "#pragma once\n\n //auto-generated\n\n"
    
    code += generate_push_backs()
    
    code += generate_position()
    code += generate_camera()
    code += generate_viewport()
    code += generate_image()
    
    code += "// function implementations\n"
    for func in functions:
        for shape in shapes:
            generated_fuction = generate_function_impl(func, shape)
            if generated_fuction != "": # excluded
                code += generated_fuction + "\n"
    
    code += "// function pointer arrays\n"
    for func in functions:
        code += generate_function_array(func)
    
    code += generate_function_array({"name": "push_back"})
    code += "// function table generator\n"
    code += generate_function_getter()
    
    return code

if __name__ == "__main__":
    code = generate_code()
    
    with open("shape_functions_generated.h", "w+") as f:
        f.write(code)
    
    print("Generated shape function implementations in shape_functions_generated.h")