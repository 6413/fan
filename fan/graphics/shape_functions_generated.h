#pragma once

 //auto-generated

static loco_t::shape_t push_back_sprite(void* properties) {
  return gloco->sprite.push_back(*reinterpret_cast<loco_t::sprite_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_text(void* properties) {
  return gloco->text.push_back(*reinterpret_cast<loco_t::text_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_line(void* properties) {
  return gloco->line.push_back(*reinterpret_cast<loco_t::line_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_rectangle(void* properties) {
  return gloco->rectangle.push_back(*reinterpret_cast<loco_t::rectangle_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_light(void* properties) {
  return gloco->light.push_back(*reinterpret_cast<loco_t::light_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_unlit_sprite(void* properties) {
  return gloco->unlit_sprite.push_back(*reinterpret_cast<loco_t::unlit_sprite_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_circle(void* properties) {
  return gloco->circle.push_back(*reinterpret_cast<loco_t::circle_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_capsule(void* properties) {
  return gloco->capsule.push_back(*reinterpret_cast<loco_t::capsule_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_polygon(void* properties) {
  return gloco->polygon.push_back(*reinterpret_cast<loco_t::polygon_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_grid(void* properties) {
  return gloco->grid.push_back(*reinterpret_cast<loco_t::grid_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_vfi(void* properties) {
  return gloco->vfi.push_back(*reinterpret_cast<loco_t::vfi_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_particles(void* properties) {
  return gloco->particles.push_back(*reinterpret_cast<loco_t::particles_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_universal_image_renderer(void* properties) {
  return gloco->universal_image_renderer.push_back(*reinterpret_cast<loco_t::universal_image_renderer_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_gradient(void* properties) {
  return gloco->gradient.push_back(*reinterpret_cast<loco_t::gradient_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_shader_shape(void* properties) {
  return gloco->shader_shape.push_back(*reinterpret_cast<loco_t::shader_shape_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_rectangle3d(void* properties) {
  return gloco->rectangle3d.push_back(*reinterpret_cast<loco_t::rectangle3d_t::properties_t*>(properties));
}

static loco_t::shape_t push_back_line3d(void* properties) {
  return gloco->line3d.push_back(*reinterpret_cast<loco_t::line3d_t::properties_t*>(properties));
}


static loco_t::camera_t get_camera(loco_t::shape_t* shape) {
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

static void set_camera(loco_t::shape_t* shape, loco_t::camera_t camera) {
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
static loco_t::viewport_t get_viewport(loco_t::shape_t* shape) {
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

static void set_viewport(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
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
loco_t::image_t get_image(loco_t::shape_t* shape) {
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

static void set_image(loco_t::shape_t* shape, loco_t::image_t image) {
         
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
// function implementations
static fan::vec3 get_position_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position;
}

static fan::vec3 get_position_line(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::line_t::vi_t*>(shape->GetRenderData(gloco->shaper))->src;
}

static fan::vec3 get_position_rectangle(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::rectangle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position;
}

static fan::vec3 get_position_light(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::light_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position;
}

static fan::vec3 get_position_unlit_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position;
}

static fan::vec3 get_position_circle(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position;
}

static fan::vec3 get_position_capsule(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position;
}

static fan::vec3 get_position_polygon(loco_t::shape_t* shape) {
  auto ri = (loco_t::polygon_t::ri_t*)shape->GetData(gloco->shaper);
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
}

static fan::vec3 get_position_grid(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::grid_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position;
}

static fan::vec3 get_position_universal_image_renderer(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::universal_image_renderer_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position;
}

static fan::vec3 get_position_gradient(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::gradient_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position;
}

static fan::vec3 get_position_shader_shape(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position;
}

static fan::vec3 get_position_rectangle3d(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::rectangle3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position;
}

static fan::vec3 get_position_line3d(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::line3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->src;
}

static void set_position2_sprite(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::sprite_t::vi_t::position),
      sizeof(loco_t::sprite_t::vi_t::position)
    );
  }
}

static void set_position2_line(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::line_t::vi_t*>(shape->GetRenderData(gloco->shaper))->src = position;
}

static void set_position2_rectangle(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::rectangle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::rectangle_t::vi_t::position),
      sizeof(loco_t::rectangle_t::vi_t::position)
    );
  }
}

static void set_position2_light(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::light_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::light_t::vi_t::position),
      sizeof(loco_t::light_t::vi_t::position)
    );
  }
}

static void set_position2_unlit_sprite(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::unlit_sprite_t::vi_t::position),
      sizeof(loco_t::unlit_sprite_t::vi_t::position)
    );
  }
}

static void set_position2_circle(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::circle_t::vi_t::position),
      sizeof(loco_t::circle_t::vi_t::position)
    );
  }
}

static void set_position2_capsule(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::capsule_t::vi_t::position),
      sizeof(loco_t::capsule_t::vi_t::position)
    );
  }
}

static void set_position2_polygon(loco_t::shape_t* shape, const fan::vec2& position) {
  auto ri = (loco_t::polygon_t::ri_t*)shape->GetData(gloco->shaper);
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
}

static void set_position2_grid(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::grid_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::grid_t::vi_t::position),
      sizeof(loco_t::grid_t::vi_t::position)
    );
  }
}

static void set_position2_universal_image_renderer(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::universal_image_renderer_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::universal_image_renderer_t::vi_t::position),
      sizeof(loco_t::universal_image_renderer_t::vi_t::position)
    );
  }
}

static void set_position2_gradient(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::gradient_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::gradient_t::vi_t::position),
      sizeof(loco_t::gradient_t::vi_t::position)
    );
  }
}

static void set_position2_shader_shape(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::shader_shape_t::vi_t::position),
      sizeof(loco_t::shader_shape_t::vi_t::position)
    );
  }
}

static void set_position3_sprite(loco_t::shape_t* shape, const fan::vec3& position) {
  reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::sprite_t::vi_t::position),
      sizeof(loco_t::sprite_t::vi_t::position)
    );
  }
}

static void set_position3_line(loco_t::shape_t* shape, const fan::vec3& position) {
  reinterpret_cast<loco_t::line_t::vi_t*>(shape->GetData(gloco->shaper))->src = position;
}

static void set_position3_rectangle(loco_t::shape_t* shape, const fan::vec3& position) {
  reinterpret_cast<loco_t::rectangle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::rectangle_t::vi_t::position),
      sizeof(loco_t::rectangle_t::vi_t::position)
    );
  }
}

static void set_position3_light(loco_t::shape_t* shape, const fan::vec3& position) {
  reinterpret_cast<loco_t::light_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::light_t::vi_t::position),
      sizeof(loco_t::light_t::vi_t::position)
    );
  }
}

static void set_position3_unlit_sprite(loco_t::shape_t* shape, const fan::vec3& position) {
  reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::unlit_sprite_t::vi_t::position),
      sizeof(loco_t::unlit_sprite_t::vi_t::position)
    );
  }
}

static void set_position3_circle(loco_t::shape_t* shape, const fan::vec3& position) {
  reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::circle_t::vi_t::position),
      sizeof(loco_t::circle_t::vi_t::position)
    );
  }
}

static void set_position3_capsule(loco_t::shape_t* shape, const fan::vec3& position) {
  reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::capsule_t::vi_t::position),
      sizeof(loco_t::capsule_t::vi_t::position)
    );
  }
}

static void set_position3_gradient(loco_t::shape_t* shape, const fan::vec3& position) {
  reinterpret_cast<loco_t::gradient_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::gradient_t::vi_t::position),
      sizeof(loco_t::gradient_t::vi_t::position)
    );
  }
}

static void set_position3_shader_shape(loco_t::shape_t* shape, const fan::vec3& position) {
  reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::shader_shape_t::vi_t::position),
      sizeof(loco_t::shader_shape_t::vi_t::position)
    );
  }
}

static void set_position3_rectangle3d(loco_t::shape_t* shape, const fan::vec3& position) {
  reinterpret_cast<loco_t::rectangle3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::rectangle3d_t::vi_t::position),
      sizeof(loco_t::rectangle3d_t::vi_t::position)
    );
  }
}

static void set_position3_line3d(loco_t::shape_t* shape, const fan::vec3& position) {
  reinterpret_cast<loco_t::line3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->src = position;
}

static fan::vec2 get_size_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size;
}

static fan::vec2 get_size_rectangle(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::rectangle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size;
}

static fan::vec2 get_size_light(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::light_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size;
}

static fan::vec2 get_size_unlit_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size;
}

static fan::vec2 get_size_circle(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->radius;
}

static fan::vec2 get_size_capsule(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->radius;
}

static fan::vec2 get_size_grid(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::grid_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size;
}

static fan::vec2 get_size_universal_image_renderer(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::universal_image_renderer_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size;
}

static fan::vec2 get_size_gradient(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::gradient_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size;
}

static fan::vec2 get_size_shader_shape(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size;
}

static fan::vec2 get_size_rectangle3d(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::rectangle3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size;
}

static fan::vec3 get_size3_rectangle3d(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::rectangle3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size;
}

static void set_size_sprite(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::sprite_t::vi_t::size),
      sizeof(loco_t::sprite_t::vi_t::size)
    );
  }
}

static void set_size_rectangle(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::rectangle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::rectangle_t::vi_t::size),
      sizeof(loco_t::rectangle_t::vi_t::size)
    );
  }
}

static void set_size_light(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::light_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::light_t::vi_t::size),
      sizeof(loco_t::light_t::vi_t::size)
    );
  }
}

static void set_size_unlit_sprite(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::unlit_sprite_t::vi_t::size),
      sizeof(loco_t::unlit_sprite_t::vi_t::size)
    );
  }
}

static void set_size_circle(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetData(gloco->shaper))->radius = size.x;
}

static void set_size_capsule(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetData(gloco->shaper))->radius = size.x;
}

static void set_size_grid(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::grid_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::grid_t::vi_t::size),
      sizeof(loco_t::grid_t::vi_t::size)
    );
  }
}

static void set_size_universal_image_renderer(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::universal_image_renderer_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::universal_image_renderer_t::vi_t::size),
      sizeof(loco_t::universal_image_renderer_t::vi_t::size)
    );
  }
}

static void set_size_gradient(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::gradient_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::gradient_t::vi_t::size),
      sizeof(loco_t::gradient_t::vi_t::size)
    );
  }
}

static void set_size_shader_shape(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::shader_shape_t::vi_t::size),
      sizeof(loco_t::shader_shape_t::vi_t::size)
    );
  }
}

static void set_size_rectangle3d(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::rectangle3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::rectangle3d_t::vi_t::size),
      sizeof(loco_t::rectangle3d_t::vi_t::size)
    );
  }
}

static void set_size3_rectangle3d(loco_t::shape_t* shape, const fan::vec3& size) {
  reinterpret_cast<loco_t::rectangle3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::rectangle3d_t::vi_t::size),
      sizeof(loco_t::rectangle3d_t::vi_t::size)
    );
  }
}

static fan::vec2 get_rotation_point_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point;
}

static fan::vec2 get_rotation_point_rectangle(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::rectangle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point;
}

static fan::vec2 get_rotation_point_light(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::light_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point;
}

static fan::vec2 get_rotation_point_unlit_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point;
}

static fan::vec2 get_rotation_point_circle(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point;
}

static fan::vec2 get_rotation_point_capsule(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point;
}

static fan::vec2 get_rotation_point_grid(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::grid_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point;
}

static fan::vec2 get_rotation_point_gradient(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::gradient_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point;
}

static fan::vec2 get_rotation_point_shader_shape(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point;
}

static void set_rotation_point_sprite(loco_t::shape_t* shape, const fan::vec2& point) {
  reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point = point;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::sprite_t::vi_t::rotation_point),
      sizeof(loco_t::sprite_t::vi_t::rotation_point)
    );
  }
}

static void set_rotation_point_rectangle(loco_t::shape_t* shape, const fan::vec2& point) {
  reinterpret_cast<loco_t::rectangle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point = point;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::rectangle_t::vi_t::rotation_point),
      sizeof(loco_t::rectangle_t::vi_t::rotation_point)
    );
  }
}

static void set_rotation_point_light(loco_t::shape_t* shape, const fan::vec2& point) {
  reinterpret_cast<loco_t::light_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point = point;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::light_t::vi_t::rotation_point),
      sizeof(loco_t::light_t::vi_t::rotation_point)
    );
  }
}

static void set_rotation_point_unlit_sprite(loco_t::shape_t* shape, const fan::vec2& point) {
  reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point = point;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::unlit_sprite_t::vi_t::rotation_point),
      sizeof(loco_t::unlit_sprite_t::vi_t::rotation_point)
    );
  }
}

static void set_rotation_point_circle(loco_t::shape_t* shape, const fan::vec2& point) {
  reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point = point;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::circle_t::vi_t::rotation_point),
      sizeof(loco_t::circle_t::vi_t::rotation_point)
    );
  }
}

static void set_rotation_point_capsule(loco_t::shape_t* shape, const fan::vec2& point) {
  reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point = point;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::capsule_t::vi_t::rotation_point),
      sizeof(loco_t::capsule_t::vi_t::rotation_point)
    );
  }
}

static void set_rotation_point_grid(loco_t::shape_t* shape, const fan::vec2& point) {
  reinterpret_cast<loco_t::grid_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point = point;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::grid_t::vi_t::rotation_point),
      sizeof(loco_t::grid_t::vi_t::rotation_point)
    );
  }
}

static void set_rotation_point_gradient(loco_t::shape_t* shape, const fan::vec2& point) {
  reinterpret_cast<loco_t::gradient_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point = point;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::gradient_t::vi_t::rotation_point),
      sizeof(loco_t::gradient_t::vi_t::rotation_point)
    );
  }
}

static void set_rotation_point_shader_shape(loco_t::shape_t* shape, const fan::vec2& point) {
  reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->rotation_point = point;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::shader_shape_t::vi_t::rotation_point),
      sizeof(loco_t::shader_shape_t::vi_t::rotation_point)
    );
  }
}

static fan::color get_color_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color;
}

static fan::color get_color_line(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::line_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color;
}

static fan::color get_color_rectangle(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::rectangle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color;
}

static fan::color get_color_light(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::light_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color;
}

static fan::color get_color_unlit_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color;
}

static fan::color get_color_circle(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color;
}

static fan::color get_color_capsule(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color;
}

static fan::color get_color_grid(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::grid_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color;
}

static fan::color get_color_gradient(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::gradient_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color[0];
}

static fan::color get_color_shader_shape(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color;
}

static fan::color get_color_rectangle3d(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::rectangle3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color;
}

static fan::color get_color_line3d(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::line3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color;
}

static void set_color_sprite(loco_t::shape_t* shape, const fan::color& color) {
  reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color = color;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::sprite_t::vi_t::color),
      sizeof(loco_t::sprite_t::vi_t::color)
    );
  }
}

static void set_color_line(loco_t::shape_t* shape, const fan::color& color) {
  reinterpret_cast<loco_t::line_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color = color;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::line_t::vi_t::color),
      sizeof(loco_t::line_t::vi_t::color)
    );
  }
}

static void set_color_rectangle(loco_t::shape_t* shape, const fan::color& color) {
  reinterpret_cast<loco_t::rectangle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color = color;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::rectangle_t::vi_t::color),
      sizeof(loco_t::rectangle_t::vi_t::color)
    );
  }
}

static void set_color_light(loco_t::shape_t* shape, const fan::color& color) {
  reinterpret_cast<loco_t::light_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color = color;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::light_t::vi_t::color),
      sizeof(loco_t::light_t::vi_t::color)
    );
  }
}

static void set_color_unlit_sprite(loco_t::shape_t* shape, const fan::color& color) {
  reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color = color;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::unlit_sprite_t::vi_t::color),
      sizeof(loco_t::unlit_sprite_t::vi_t::color)
    );
  }
}

static void set_color_circle(loco_t::shape_t* shape, const fan::color& color) {
  reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color = color;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::circle_t::vi_t::color),
      sizeof(loco_t::circle_t::vi_t::color)
    );
  }
}

static void set_color_capsule(loco_t::shape_t* shape, const fan::color& color) {
  reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color = color;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::capsule_t::vi_t::color),
      sizeof(loco_t::capsule_t::vi_t::color)
    );
  }
}

static void set_color_grid(loco_t::shape_t* shape, const fan::color& color) {
  reinterpret_cast<loco_t::grid_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color = color;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::grid_t::vi_t::color),
      sizeof(loco_t::grid_t::vi_t::color)
    );
  }
}

static void set_color_gradient(loco_t::shape_t* shape, const fan::color& color) {
  reinterpret_cast<loco_t::gradient_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color[0] = color;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::gradient_t::vi_t::color),
      sizeof(loco_t::gradient_t::vi_t::color)
    );
  }
}

static void set_color_shader_shape(loco_t::shape_t* shape, const fan::color& color) {
  reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color = color;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::shader_shape_t::vi_t::color),
      sizeof(loco_t::shader_shape_t::vi_t::color)
    );
  }
}

static void set_color_rectangle3d(loco_t::shape_t* shape, const fan::color& color) {
  reinterpret_cast<loco_t::rectangle3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color = color;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::rectangle3d_t::vi_t::color),
      sizeof(loco_t::rectangle3d_t::vi_t::color)
    );
  }
}

static void set_color_line3d(loco_t::shape_t* shape, const fan::color& color) {
  reinterpret_cast<loco_t::line3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->color = color;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::line3d_t::vi_t::color),
      sizeof(loco_t::line3d_t::vi_t::color)
    );
  }
}

static fan::vec3 get_angle_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle;
}

static fan::vec3 get_angle_rectangle(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::rectangle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle;
}

static fan::vec3 get_angle_light(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::light_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle;
}

static fan::vec3 get_angle_unlit_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle;
}

static fan::vec3 get_angle_circle(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle;
}

static fan::vec3 get_angle_capsule(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle;
}

static fan::vec3 get_angle_grid(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::grid_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle;
}

static fan::vec3 get_angle_gradient(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::gradient_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle;
}

static fan::vec3 get_angle_shader_shape(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle;
}

static fan::vec3 get_angle_rectangle3d(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::rectangle3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle;
}

static void set_angle_sprite(loco_t::shape_t* shape, const fan::vec3& angle) {
  reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle = angle;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::sprite_t::vi_t::angle),
      sizeof(loco_t::sprite_t::vi_t::angle)
    );
  }
}

static void set_angle_rectangle(loco_t::shape_t* shape, const fan::vec3& angle) {
  reinterpret_cast<loco_t::rectangle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle = angle;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::rectangle_t::vi_t::angle),
      sizeof(loco_t::rectangle_t::vi_t::angle)
    );
  }
}

static void set_angle_light(loco_t::shape_t* shape, const fan::vec3& angle) {
  reinterpret_cast<loco_t::light_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle = angle;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::light_t::vi_t::angle),
      sizeof(loco_t::light_t::vi_t::angle)
    );
  }
}

static void set_angle_unlit_sprite(loco_t::shape_t* shape, const fan::vec3& angle) {
  reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle = angle;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::unlit_sprite_t::vi_t::angle),
      sizeof(loco_t::unlit_sprite_t::vi_t::angle)
    );
  }
}

static void set_angle_circle(loco_t::shape_t* shape, const fan::vec3& angle) {
  reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle = angle;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::circle_t::vi_t::angle),
      sizeof(loco_t::circle_t::vi_t::angle)
    );
  }
}

static void set_angle_capsule(loco_t::shape_t* shape, const fan::vec3& angle) {
  reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle = angle;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::capsule_t::vi_t::angle),
      sizeof(loco_t::capsule_t::vi_t::angle)
    );
  }
}

static void set_angle_grid(loco_t::shape_t* shape, const fan::vec3& angle) {
  reinterpret_cast<loco_t::grid_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle = angle;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::grid_t::vi_t::angle),
      sizeof(loco_t::grid_t::vi_t::angle)
    );
  }
}

static void set_angle_gradient(loco_t::shape_t* shape, const fan::vec3& angle) {
  reinterpret_cast<loco_t::gradient_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle = angle;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::gradient_t::vi_t::angle),
      sizeof(loco_t::gradient_t::vi_t::angle)
    );
  }
}

static void set_angle_shader_shape(loco_t::shape_t* shape, const fan::vec3& angle) {
  reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle = angle;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::shader_shape_t::vi_t::angle),
      sizeof(loco_t::shader_shape_t::vi_t::angle)
    );
  }
}

static void set_angle_rectangle3d(loco_t::shape_t* shape, const fan::vec3& angle) {
  reinterpret_cast<loco_t::rectangle3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->angle = angle;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::rectangle3d_t::vi_t::angle),
      sizeof(loco_t::rectangle3d_t::vi_t::angle)
    );
  }
}

static fan::vec2 get_tc_position_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_position;
}

static fan::vec2 get_tc_position_unlit_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_position;
}

static fan::vec2 get_tc_position_universal_image_renderer(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::universal_image_renderer_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_position;
}

static fan::vec2 get_tc_position_shader_shape(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_position;
}

static void set_tc_position_sprite(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::sprite_t::vi_t::tc_position),
      sizeof(loco_t::sprite_t::vi_t::tc_position)
    );
  }
}

static void set_tc_position_unlit_sprite(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::unlit_sprite_t::vi_t::tc_position),
      sizeof(loco_t::unlit_sprite_t::vi_t::tc_position)
    );
  }
}

static void set_tc_position_universal_image_renderer(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::universal_image_renderer_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::universal_image_renderer_t::vi_t::tc_position),
      sizeof(loco_t::universal_image_renderer_t::vi_t::tc_position)
    );
  }
}

static void set_tc_position_shader_shape(loco_t::shape_t* shape, const fan::vec2& position) {
  reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_position = position;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::shader_shape_t::vi_t::tc_position),
      sizeof(loco_t::shader_shape_t::vi_t::tc_position)
    );
  }
}

static fan::vec2 get_tc_size_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_size;
}

static fan::vec2 get_tc_size_unlit_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_size;
}

static fan::vec2 get_tc_size_universal_image_renderer(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::universal_image_renderer_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_size;
}

static fan::vec2 get_tc_size_shader_shape(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_size;
}

static void set_tc_size_sprite(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::sprite_t::vi_t::tc_size),
      sizeof(loco_t::sprite_t::vi_t::tc_size)
    );
  }
}

static void set_tc_size_unlit_sprite(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::unlit_sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::unlit_sprite_t::vi_t::tc_size),
      sizeof(loco_t::unlit_sprite_t::vi_t::tc_size)
    );
  }
}

static void set_tc_size_universal_image_renderer(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::universal_image_renderer_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::universal_image_renderer_t::vi_t::tc_size),
      sizeof(loco_t::universal_image_renderer_t::vi_t::tc_size)
    );
  }
}

static void set_tc_size_shader_shape(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::shader_shape_t::vi_t*>(shape->GetRenderData(gloco->shaper))->tc_size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::shader_shape_t::vi_t::tc_size),
      sizeof(loco_t::shader_shape_t::vi_t::tc_size)
    );
  }
}

static bool load_tp_sprite(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_line(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_rectangle(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_light(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_unlit_sprite(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_circle(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_capsule(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_polygon(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_grid(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_universal_image_renderer(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_gradient(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_shader_shape(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_rectangle3d(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static bool load_tp_line3d(loco_t::shape_t* shape, loco_t::texturepack_t::ti_t* tp) {
  return false;
}

static fan::vec2 get_grid_size_grid(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::grid_t::vi_t*>(shape->GetRenderData(gloco->shaper))->grid_size;
}

static void set_grid_size_grid(loco_t::shape_t* shape, const fan::vec2& size) {
  reinterpret_cast<loco_t::grid_t::vi_t*>(shape->GetRenderData(gloco->shaper))->grid_size = size;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::grid_t::vi_t::grid_size),
      sizeof(loco_t::grid_t::vi_t::grid_size)
    );
  }
}

static loco_t::camera_t get_camera_sprite(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_line(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_rectangle(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_light(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_unlit_sprite(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_circle(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_capsule(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_polygon(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_grid(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_universal_image_renderer(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_gradient(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_shader_shape(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_rectangle3d(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static loco_t::camera_t get_camera_line3d(loco_t::shape_t* shape) {
	return get_camera(shape);
}

static void set_camera_sprite(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_line(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_rectangle(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_light(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_unlit_sprite(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_circle(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_capsule(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_polygon(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_grid(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_universal_image_renderer(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_gradient(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_shader_shape(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_rectangle3d(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static void set_camera_line3d(loco_t::shape_t* shape, loco_t::camera_t camera) {
	set_camera(shape, camera);
}

static loco_t::viewport_t get_viewport_sprite(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_line(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_rectangle(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_light(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_unlit_sprite(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_circle(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_capsule(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_polygon(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_grid(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_universal_image_renderer(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_gradient(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_shader_shape(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_rectangle3d(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static loco_t::viewport_t get_viewport_line3d(loco_t::shape_t* shape) {
	return get_viewport(shape);
}

static void set_viewport_sprite(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_line(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_rectangle(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_light(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_unlit_sprite(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_circle(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_capsule(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_polygon(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_grid(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_universal_image_renderer(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_gradient(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_shader_shape(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_rectangle3d(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static void set_viewport_line3d(loco_t::shape_t* shape, loco_t::viewport_t viewport) {
	set_viewport(shape, viewport);
}

static loco_t::image_t get_image_sprite(loco_t::shape_t* shape) {
	return get_image(shape);
}

static loco_t::image_t get_image_circle(loco_t::shape_t* shape) {
	return get_image(shape);
}

static loco_t::image_t get_image_capsule(loco_t::shape_t* shape) {
	return get_image(shape);
}

static loco_t::image_t get_image_universal_image_renderer(loco_t::shape_t* shape) {
	return get_image(shape);
}

static void set_image_sprite(loco_t::shape_t* shape, loco_t::image_t image) {
	set_image(shape, image);
}

static void set_image_circle(loco_t::shape_t* shape, loco_t::image_t image) {
	set_image(shape, image);
}

static void set_image_capsule(loco_t::shape_t* shape, loco_t::image_t image) {
	set_image(shape, image);
}

static void set_image_universal_image_renderer(loco_t::shape_t* shape, loco_t::image_t image) {
	set_image(shape, image);
}

static fan::graphics::image_data_t& get_image_data_sprite(loco_t::shape_t* shape) {
	return gloco->image_list[shape->get_image()];
}

static fan::graphics::image_data_t& get_image_data_rectangle(loco_t::shape_t* shape) {
	return gloco->image_list[shape->get_image()];
}

static fan::graphics::image_data_t& get_image_data_unlit_sprite(loco_t::shape_t* shape) {
	return gloco->image_list[shape->get_image()];
}

static fan::graphics::image_data_t& get_image_data_polygon(loco_t::shape_t* shape) {
	return gloco->image_list[shape->get_image()];
}

static fan::graphics::image_data_t& get_image_data_grid(loco_t::shape_t* shape) {
	return gloco->image_list[shape->get_image()];
}

static fan::graphics::image_data_t& get_image_data_universal_image_renderer(loco_t::shape_t* shape) {
	return gloco->image_list[shape->get_image()];
}

static fan::graphics::image_data_t& get_image_data_gradient(loco_t::shape_t* shape) {
	return gloco->image_list[shape->get_image()];
}

static fan::graphics::image_data_t& get_image_data_shader_shape(loco_t::shape_t* shape) {
	return gloco->image_list[shape->get_image()];
}

static fan::graphics::image_data_t& get_image_data_rectangle3d(loco_t::shape_t* shape) {
	return gloco->image_list[shape->get_image()];
}

static f32_t get_parallax_factor_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->parallax_factor;
}

static void set_parallax_factor_sprite(loco_t::shape_t* shape, f32_t factor) {
  reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->parallax_factor = factor;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::sprite_t::vi_t::parallax_factor),
      sizeof(loco_t::sprite_t::vi_t::parallax_factor)
    );
  }
}

static uint32_t get_flags_sprite(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->flags;
}

static uint32_t get_flags_circle(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->flags;
}

static uint32_t get_flags_capsule(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->flags;
}

static void set_flags_sprite(loco_t::shape_t* shape, uint32_t flags) {
  reinterpret_cast<loco_t::sprite_t::vi_t*>(shape->GetRenderData(gloco->shaper))->flags = flags;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::sprite_t::vi_t::flags),
      sizeof(loco_t::sprite_t::vi_t::flags)
    );
  }
}

static void set_flags_circle(loco_t::shape_t* shape, uint32_t flags) {
  reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->flags = flags;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::circle_t::vi_t::flags),
      sizeof(loco_t::circle_t::vi_t::flags)
    );
  }
}

static void set_flags_capsule(loco_t::shape_t* shape, uint32_t flags) {
  reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->flags = flags;
  if (gloco->window.renderer == loco_t::renderer_t::opengl) {
    auto& data = gloco->shaper.ShapeList[*shape];
    gloco->shaper.ElementIsPartiallyEdited(
      data.sti,
      data.blid,
      data.ElementIndex,
      fan::member_offset(&loco_t::capsule_t::vi_t::flags),
      sizeof(loco_t::capsule_t::vi_t::flags)
    );
  }
}

static f32_t get_radius_circle(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::circle_t::vi_t*>(shape->GetRenderData(gloco->shaper))->radius;
}

static f32_t get_radius_capsule(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::capsule_t::vi_t*>(shape->GetRenderData(gloco->shaper))->radius;
}

static fan::vec3 get_src_line(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::line_t::vi_t*>(shape->GetRenderData(gloco->shaper))->src;
}

static fan::vec3 get_src_line3d(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::line3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->src;
}

static fan::vec3 get_dst_line(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::line_t::vi_t*>(shape->GetRenderData(gloco->shaper))->dst;
}

static fan::vec3 get_dst_line3d(loco_t::shape_t* shape) {
  return reinterpret_cast<loco_t::line3d_t::vi_t*>(shape->GetRenderData(gloco->shaper))->dst;
}

static void reload_sprite(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_line(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_rectangle(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_light(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_unlit_sprite(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_circle(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_capsule(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_polygon(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_grid(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_universal_image_renderer(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_gradient(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_shader_shape(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_rectangle3d(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void reload_line3d(loco_t::shape_t* shape, uint8_t format, void** image_data, const fan::vec2& size, uint32_t filter) {
}

static void draw_sprite(uint8_t draw_range) {
}

static void draw_line(uint8_t draw_range) {
}

static void draw_rectangle(uint8_t draw_range) {
}

static void draw_light(uint8_t draw_range) {
}

static void draw_unlit_sprite(uint8_t draw_range) {
}

static void draw_circle(uint8_t draw_range) {
}

static void draw_capsule(uint8_t draw_range) {
}

static void draw_polygon(uint8_t draw_range) {
}

static void draw_grid(uint8_t draw_range) {
}

static void draw_universal_image_renderer(uint8_t draw_range) {
}

static void draw_gradient(uint8_t draw_range) {
}

static void draw_shader_shape(uint8_t draw_range) {
}

static void draw_rectangle3d(uint8_t draw_range) {
}

static void draw_line3d(uint8_t draw_range) {
}

static void set_line_line(loco_t::shape_t* shape, const fan::vec2& src, const fan::vec2& dst) {
  auto data = reinterpret_cast<loco_t::line_t::vi_t*>(shape->GetRenderData(gloco->shaper));
  data->src = fan::vec3(src.x, src.y, 0);
  data->dst = fan::vec3(dst.x, dst.y, 0);
}

static void set_line_line3d(loco_t::shape_t* shape, const fan::vec2& src, const fan::vec2& dst) {
auto data = reinterpret_cast<loco_t::line3d_t::vi_t*>(shape->GetRenderData(gloco->shaper));
  data->src = fan::vec3(src.x, src.y, 0);
  data->dst = fan::vec3(dst.x, dst.y, 0);
}

static void set_line3_rectangle3d(loco_t::shape_t* shape, const fan::vec3& src, const fan::vec3& dst) {
}

static void set_line3_line3d(loco_t::shape_t* shape, const fan::vec3& src, const fan::vec3& dst) {
}

// function pointer arrays
static get_position_cb get_position_functions[] = {
  &get_position_sprite,
  nullptr,
  nullptr,
  &get_position_line,
  nullptr,
  &get_position_rectangle,
  &get_position_light,
  &get_position_unlit_sprite,
  &get_position_circle,
  &get_position_capsule,
  &get_position_polygon,
  &get_position_grid,
  nullptr,
  nullptr,
  &get_position_universal_image_renderer,
  &get_position_gradient,
  nullptr,
  &get_position_shader_shape,
  &get_position_rectangle3d,
  &get_position_line3d,
};

static set_position2_cb set_position2_functions[] = {
  &set_position2_sprite,
  nullptr,
  nullptr,
  &set_position2_line,
  nullptr,
  &set_position2_rectangle,
  &set_position2_light,
  &set_position2_unlit_sprite,
  &set_position2_circle,
  &set_position2_capsule,
  &set_position2_polygon,
  &set_position2_grid,
  nullptr,
  nullptr,
  &set_position2_universal_image_renderer,
  &set_position2_gradient,
  nullptr,
  &set_position2_shader_shape,
  nullptr,
  nullptr,
};

static set_position3_cb set_position3_functions[] = {
  &set_position3_sprite,
  nullptr,
  nullptr,
  &set_position3_line,
  nullptr,
  &set_position3_rectangle,
  &set_position3_light,
  &set_position3_unlit_sprite,
  &set_position3_circle,
  &set_position3_capsule,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_position3_gradient,
  nullptr,
  &set_position3_shader_shape,
  &set_position3_rectangle3d,
  &set_position3_line3d,
};

static get_size_cb get_size_functions[] = {
  &get_size_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_size_rectangle,
  &get_size_light,
  &get_size_unlit_sprite,
  &get_size_circle,
  &get_size_capsule,
  nullptr,
  &get_size_grid,
  nullptr,
  nullptr,
  &get_size_universal_image_renderer,
  &get_size_gradient,
  nullptr,
  &get_size_shader_shape,
  &get_size_rectangle3d,
  nullptr,
};

static get_size3_cb get_size3_functions[] = {
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_size3_rectangle3d,
  nullptr,
};

static set_size_cb set_size_functions[] = {
  &set_size_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_size_rectangle,
  &set_size_light,
  &set_size_unlit_sprite,
  &set_size_circle,
  &set_size_capsule,
  nullptr,
  &set_size_grid,
  nullptr,
  nullptr,
  &set_size_universal_image_renderer,
  &set_size_gradient,
  nullptr,
  &set_size_shader_shape,
  &set_size_rectangle3d,
  nullptr,
};

static set_size3_cb set_size3_functions[] = {
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_size3_rectangle3d,
  nullptr,
};

static get_rotation_point_cb get_rotation_point_functions[] = {
  &get_rotation_point_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_rotation_point_rectangle,
  &get_rotation_point_light,
  &get_rotation_point_unlit_sprite,
  &get_rotation_point_circle,
  &get_rotation_point_capsule,
  nullptr,
  &get_rotation_point_grid,
  nullptr,
  nullptr,
  nullptr,
  &get_rotation_point_gradient,
  nullptr,
  &get_rotation_point_shader_shape,
  nullptr,
  nullptr,
};

static set_rotation_point_cb set_rotation_point_functions[] = {
  &set_rotation_point_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_rotation_point_rectangle,
  &set_rotation_point_light,
  &set_rotation_point_unlit_sprite,
  &set_rotation_point_circle,
  &set_rotation_point_capsule,
  nullptr,
  &set_rotation_point_grid,
  nullptr,
  nullptr,
  nullptr,
  &set_rotation_point_gradient,
  nullptr,
  &set_rotation_point_shader_shape,
  nullptr,
  nullptr,
};

static get_color_cb get_color_functions[] = {
  &get_color_sprite,
  nullptr,
  nullptr,
  &get_color_line,
  nullptr,
  &get_color_rectangle,
  &get_color_light,
  &get_color_unlit_sprite,
  &get_color_circle,
  &get_color_capsule,
  nullptr,
  &get_color_grid,
  nullptr,
  nullptr,
  nullptr,
  &get_color_gradient,
  nullptr,
  &get_color_shader_shape,
  &get_color_rectangle3d,
  &get_color_line3d,
};

static set_color_cb set_color_functions[] = {
  &set_color_sprite,
  nullptr,
  nullptr,
  &set_color_line,
  nullptr,
  &set_color_rectangle,
  &set_color_light,
  &set_color_unlit_sprite,
  &set_color_circle,
  &set_color_capsule,
  nullptr,
  &set_color_grid,
  nullptr,
  nullptr,
  nullptr,
  &set_color_gradient,
  nullptr,
  &set_color_shader_shape,
  &set_color_rectangle3d,
  &set_color_line3d,
};

static get_angle_cb get_angle_functions[] = {
  &get_angle_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_angle_rectangle,
  &get_angle_light,
  &get_angle_unlit_sprite,
  &get_angle_circle,
  &get_angle_capsule,
  nullptr,
  &get_angle_grid,
  nullptr,
  nullptr,
  nullptr,
  &get_angle_gradient,
  nullptr,
  &get_angle_shader_shape,
  &get_angle_rectangle3d,
  nullptr,
};

static set_angle_cb set_angle_functions[] = {
  &set_angle_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_angle_rectangle,
  &set_angle_light,
  &set_angle_unlit_sprite,
  &set_angle_circle,
  &set_angle_capsule,
  nullptr,
  &set_angle_grid,
  nullptr,
  nullptr,
  nullptr,
  &set_angle_gradient,
  nullptr,
  &set_angle_shader_shape,
  &set_angle_rectangle3d,
  nullptr,
};

static get_tc_position_cb get_tc_position_functions[] = {
  &get_tc_position_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_tc_position_unlit_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_tc_position_universal_image_renderer,
  nullptr,
  nullptr,
  &get_tc_position_shader_shape,
  nullptr,
  nullptr,
};

static set_tc_position_cb set_tc_position_functions[] = {
  &set_tc_position_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_tc_position_unlit_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_tc_position_universal_image_renderer,
  nullptr,
  nullptr,
  &set_tc_position_shader_shape,
  nullptr,
  nullptr,
};

static get_tc_size_cb get_tc_size_functions[] = {
  &get_tc_size_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_tc_size_unlit_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_tc_size_universal_image_renderer,
  nullptr,
  nullptr,
  &get_tc_size_shader_shape,
  nullptr,
  nullptr,
};

static set_tc_size_cb set_tc_size_functions[] = {
  &set_tc_size_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_tc_size_unlit_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_tc_size_universal_image_renderer,
  nullptr,
  nullptr,
  &set_tc_size_shader_shape,
  nullptr,
  nullptr,
};

static load_tp_cb load_tp_functions[] = {
  &load_tp_sprite,
  nullptr,
  nullptr,
  &load_tp_line,
  nullptr,
  &load_tp_rectangle,
  &load_tp_light,
  &load_tp_unlit_sprite,
  &load_tp_circle,
  &load_tp_capsule,
  &load_tp_polygon,
  &load_tp_grid,
  nullptr,
  nullptr,
  &load_tp_universal_image_renderer,
  &load_tp_gradient,
  nullptr,
  &load_tp_shader_shape,
  &load_tp_rectangle3d,
  &load_tp_line3d,
};

static get_grid_size_cb get_grid_size_functions[] = {
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_grid_size_grid,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

static set_grid_size_cb set_grid_size_functions[] = {
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_grid_size_grid,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

static get_camera_cb get_camera_functions[] = {
  &get_camera_sprite,
  nullptr,
  nullptr,
  &get_camera_line,
  nullptr,
  &get_camera_rectangle,
  &get_camera_light,
  &get_camera_unlit_sprite,
  &get_camera_circle,
  &get_camera_capsule,
  &get_camera_polygon,
  &get_camera_grid,
  nullptr,
  nullptr,
  &get_camera_universal_image_renderer,
  &get_camera_gradient,
  nullptr,
  &get_camera_shader_shape,
  &get_camera_rectangle3d,
  &get_camera_line3d,
};

static set_camera_cb set_camera_functions[] = {
  &set_camera_sprite,
  nullptr,
  nullptr,
  &set_camera_line,
  nullptr,
  &set_camera_rectangle,
  &set_camera_light,
  &set_camera_unlit_sprite,
  &set_camera_circle,
  &set_camera_capsule,
  &set_camera_polygon,
  &set_camera_grid,
  nullptr,
  nullptr,
  &set_camera_universal_image_renderer,
  &set_camera_gradient,
  nullptr,
  &set_camera_shader_shape,
  &set_camera_rectangle3d,
  &set_camera_line3d,
};

static get_viewport_cb get_viewport_functions[] = {
  &get_viewport_sprite,
  nullptr,
  nullptr,
  &get_viewport_line,
  nullptr,
  &get_viewport_rectangle,
  &get_viewport_light,
  &get_viewport_unlit_sprite,
  &get_viewport_circle,
  &get_viewport_capsule,
  &get_viewport_polygon,
  &get_viewport_grid,
  nullptr,
  nullptr,
  &get_viewport_universal_image_renderer,
  &get_viewport_gradient,
  nullptr,
  &get_viewport_shader_shape,
  &get_viewport_rectangle3d,
  &get_viewport_line3d,
};

static set_viewport_cb set_viewport_functions[] = {
  &set_viewport_sprite,
  nullptr,
  nullptr,
  &set_viewport_line,
  nullptr,
  &set_viewport_rectangle,
  &set_viewport_light,
  &set_viewport_unlit_sprite,
  &set_viewport_circle,
  &set_viewport_capsule,
  &set_viewport_polygon,
  &set_viewport_grid,
  nullptr,
  nullptr,
  &set_viewport_universal_image_renderer,
  &set_viewport_gradient,
  nullptr,
  &set_viewport_shader_shape,
  &set_viewport_rectangle3d,
  &set_viewport_line3d,
};

static get_image_cb get_image_functions[] = {
  &get_image_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_image_circle,
  &get_image_capsule,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_image_universal_image_renderer,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

static set_image_cb set_image_functions[] = {
  &set_image_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_image_circle,
  &set_image_capsule,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_image_universal_image_renderer,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

static get_image_data_cb get_image_data_functions[] = {
  &get_image_data_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_image_data_rectangle,
  nullptr,
  &get_image_data_unlit_sprite,
  nullptr,
  nullptr,
  &get_image_data_polygon,
  &get_image_data_grid,
  nullptr,
  nullptr,
  &get_image_data_universal_image_renderer,
  &get_image_data_gradient,
  nullptr,
  &get_image_data_shader_shape,
  &get_image_data_rectangle3d,
  nullptr,
};

static get_parallax_factor_cb get_parallax_factor_functions[] = {
  &get_parallax_factor_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

static set_parallax_factor_cb set_parallax_factor_functions[] = {
  &set_parallax_factor_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

static get_flags_cb get_flags_functions[] = {
  &get_flags_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_flags_circle,
  &get_flags_capsule,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

static set_flags_cb set_flags_functions[] = {
  &set_flags_sprite,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_flags_circle,
  &set_flags_capsule,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

static get_radius_cb get_radius_functions[] = {
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_radius_circle,
  &get_radius_capsule,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

static get_src_cb get_src_functions[] = {
  nullptr,
  nullptr,
  nullptr,
  &get_src_line,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_src_line3d,
};

static get_dst_cb get_dst_functions[] = {
  nullptr,
  nullptr,
  nullptr,
  &get_dst_line,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &get_dst_line3d,
};

static get_outline_size_cb get_outline_size_functions[] = {
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

static get_outline_color_cb get_outline_color_functions[] = {
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
};

static reload_cb reload_functions[] = {
  &reload_sprite,
  nullptr,
  nullptr,
  &reload_line,
  nullptr,
  &reload_rectangle,
  &reload_light,
  &reload_unlit_sprite,
  &reload_circle,
  &reload_capsule,
  &reload_polygon,
  &reload_grid,
  nullptr,
  nullptr,
  &reload_universal_image_renderer,
  &reload_gradient,
  nullptr,
  &reload_shader_shape,
  &reload_rectangle3d,
  &reload_line3d,
};

static draw_cb draw_functions[] = {
  &draw_sprite,
  nullptr,
  nullptr,
  &draw_line,
  nullptr,
  &draw_rectangle,
  &draw_light,
  &draw_unlit_sprite,
  &draw_circle,
  &draw_capsule,
  &draw_polygon,
  &draw_grid,
  nullptr,
  nullptr,
  &draw_universal_image_renderer,
  &draw_gradient,
  nullptr,
  &draw_shader_shape,
  &draw_rectangle3d,
  &draw_line3d,
};

static set_line_cb set_line_functions[] = {
  nullptr,
  nullptr,
  nullptr,
  &set_line_line,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_line_line3d,
};

static set_line3_cb set_line3_functions[] = {
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  &set_line3_rectangle3d,
  &set_line3_line3d,
};

static push_back_cb push_back_functions[] = {
  &push_back_sprite,
  &push_back_text,
  nullptr,
  &push_back_line,
  nullptr,
  &push_back_rectangle,
  &push_back_light,
  &push_back_unlit_sprite,
  &push_back_circle,
  &push_back_capsule,
  &push_back_polygon,
  &push_back_grid,
  &push_back_vfi,
  &push_back_particles,
  &push_back_universal_image_renderer,
  &push_back_gradient,
  nullptr,
  &push_back_shader_shape,
  &push_back_rectangle3d,
  &push_back_line3d,
};

// function table generator
loco_t::functions_t loco_t::get_shape_functions(uint16_t type) {
  uint16_t index = type;
  loco_t::functions_t funcs{};

  funcs.get_position = get_position_functions[index];
  funcs.set_position2 = set_position2_functions[index];
  funcs.set_position3 = set_position3_functions[index];
  funcs.get_size = get_size_functions[index];
  funcs.get_size3 = get_size3_functions[index];
  funcs.set_size = set_size_functions[index];
  funcs.set_size3 = set_size3_functions[index];
  funcs.get_rotation_point = get_rotation_point_functions[index];
  funcs.set_rotation_point = set_rotation_point_functions[index];
  funcs.get_color = get_color_functions[index];
  funcs.set_color = set_color_functions[index];
  funcs.get_angle = get_angle_functions[index];
  funcs.set_angle = set_angle_functions[index];
  funcs.get_tc_position = get_tc_position_functions[index];
  funcs.set_tc_position = set_tc_position_functions[index];
  funcs.get_tc_size = get_tc_size_functions[index];
  funcs.set_tc_size = set_tc_size_functions[index];
  funcs.load_tp = load_tp_functions[index];
  funcs.get_grid_size = get_grid_size_functions[index];
  funcs.set_grid_size = set_grid_size_functions[index];
  funcs.get_camera = get_camera_functions[index];
  funcs.set_camera = set_camera_functions[index];
  funcs.get_viewport = get_viewport_functions[index];
  funcs.set_viewport = set_viewport_functions[index];
  funcs.get_image = get_image_functions[index];
  funcs.set_image = set_image_functions[index];
  funcs.get_image_data = get_image_data_functions[index];
  funcs.get_parallax_factor = get_parallax_factor_functions[index];
  funcs.set_parallax_factor = set_parallax_factor_functions[index];
  funcs.get_flags = get_flags_functions[index];
  funcs.set_flags = set_flags_functions[index];
  funcs.get_radius = get_radius_functions[index];
  funcs.get_src = get_src_functions[index];
  funcs.get_dst = get_dst_functions[index];
  funcs.get_outline_size = get_outline_size_functions[index];
  funcs.get_outline_color = get_outline_color_functions[index];
  funcs.reload = reload_functions[index];
  funcs.draw = draw_functions[index];
  funcs.set_line = set_line_functions[index];
  funcs.set_line3 = set_line3_functions[index];
  funcs.push_back = push_back_functions[index];

  return funcs;
}
