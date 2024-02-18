// <redraw key> <depth> <shape type> <context key> == block manager
// <redraw key, depth, shape type, context key> == block manager - problem is that depth needs order and other keys doesnt need

/*
* TODO light draw
*/

begin = 0;

for (auto& i : m_draw_queue_light) {
  i();
}

typename loco_bdbt_Key_t<sizeof(redraw_key_t) * 8>::Traverse_t t0;
t0.i(root);
redraw_key_t redraw_key;
while(t0.t(&bdbt, &redraw_key)) {
  if (redraw_key.blending == false) {
    get_context().opengl.call(get_context().opengl.glDisable, fan::opengl::GL_BLEND);
  }
  else {
    get_context().opengl.call(get_context().opengl.glEnable, fan::opengl::GL_BLEND);
    get_context().opengl.call(get_context().opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
  }

  typename loco_bdbt_Key_t<sizeof(uint16_t) * 8, true>::Traverse_t t1;
  t1.i0(t0.Output, redraw_key.blending == false);
  uint16_t depth;
  while (t1.t0(&bdbt, &depth, redraw_key.blending == false)) {
    typename loco_bdbt_Key_t<sizeof(shape_type_t) * 8>::Traverse_t t2;
    t2.i(t1.Output);
    loco_t::shape_type_t shape_type;
    while (t2.t(&bdbt, &shape_type)) {  
      shape_draw(shape_type, redraw_key, t2.Output);
    }
  }
}

m_framebuffer.unbind(get_context());

get_context().opengl.glFlush();
get_context().opengl.glFinish();

for (auto& i : m_post_draw) {
  i();
}

get_context().opengl.glFlush();
get_context().opengl.glFinish();

m_framebuffer.bind(get_context());