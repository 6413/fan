get_context()->opengl.call(get_context()->opengl.glDisable, fan::opengl::GL_BLEND);

#if defined(loco_light)
light.draw();
#endif

#if defined(loco_light_sun)
light_sun.draw();
#endif

for (auto& i : m_draw_queue_light) {
  i();
}

for (auto it = m_draw_queue.rbegin(); it != m_draw_queue.rend(); ++it) {
  for (auto it2 = it->f.begin(); it2 != it->f.end(); ++it2) {
    (*it2)();
  }
}

m_draw_queue.clear();

for (uint8_t i = 0; i < 2; i++) {
  if (i == 0) {
    sprite.draw_queue_helper.push_back([&]() {
      get_context()->opengl.call(get_context()->opengl.glDisable, fan::opengl::GL_BLEND);
    });

    unlit_sprite.draw_queue_helper.push_back([&]() {
      get_context()->opengl.call(get_context()->opengl.glDisable, fan::opengl::GL_BLEND);
    });
  }
  else {
    sprite.draw_queue_helper.push_back([&]() {
      get_context()->opengl.call(get_context()->opengl.glEnable, fan::opengl::GL_BLEND);
      get_context()->opengl.call(get_context()->opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
    });

    unlit_sprite.draw_queue_helper.push_back([&]() {
      get_context()->opengl.call(get_context()->opengl.glEnable, fan::opengl::GL_BLEND);
      get_context()->opengl.call(get_context()->opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
    });
  }
  #if defined(loco_rectangle)
    rectangle.draw(i);
  #endif
  #if defined(loco_circle)
    circle.draw(i);
  #endif
  #if defined(loco_pixel_format_renderer)
    pixel_format_renderer.draw(i);
  #endif
  #if defined(loco_sprite)
    // can be moved
    sprite.draw(i);
  #endif
  #if defined(loco_unlit_sprite)
    // can be moved
    unlit_sprite.draw(i);
  #endif
  #if defined(loco_line)
    line.draw(i);
  #endif
  #if defined(loco_letter)
    // loco_t::text gets drawn here as well as it uses letter
    letter.draw(i);
  #endif
  #if defined(loco_text_box)
    text_box.draw(i);
  #endif
  #if defined(loco_button)
    button.draw(i);
  #endif
  #if defined(loco_model_3d)
    model.draw(i);
  #endif
}

#if defined(loco_post_process)
  post_process.draw();
#endif

for (auto it = m_draw_queue.rbegin(); it != m_draw_queue.rend(); ++it) {
  for (auto it2 = it->f.begin(); it2 != it->f.end(); ++it2) {
    (*it2)();
  }
}

m_draw_queue.clear();