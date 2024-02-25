// <redraw key> <depth> <shape type> <context key> == block manager
// <redraw key, depth, shape type, context key> == block manager - problem is that depth needs order and other keys doesnt need

/*
* TODO light draw
*/

begin = 0;

for (auto& i : m_draw_queue_light) {
  i();
}

//#if defined(loco_vulkan)
//shapes.iterate([&]<auto i>(auto & shape) {
//  get_context().bind_draw(
//    shape.m_pipeline,
//    1,
//    &shape.m_ssbo.m_descriptor.m_descriptor_set[get_context().currentFrame]
//  );
//});
//#endif

#if defined(loco_vulkan)
bool bind_array[shapes.size()]{};
#endif

typename loco_bdbt_Key_t<sizeof(redraw_key_t) * 8>::Traverse_t t0;
t0.i(root);
redraw_key_t redraw_key;
while(t0.t(&bdbt, &redraw_key)) {
  if (redraw_key.blending == false) {
    #if defined(loco_opengl)
    get_context().opengl.call(get_context().opengl.glDisable, fan::opengl::GL_BLEND);
    #endif
  }
  else {
    #if defined(loco_opengl)
    get_context().opengl.call(get_context().opengl.glEnable, fan::opengl::GL_BLEND);
    get_context().opengl.call(get_context().opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
    #endif
  }

  typename loco_bdbt_Key_t<sizeof(uint16_t) * 8, true>::Traverse_t t1;
  t1.i0(t0.Output, redraw_key.blending == false);
  uint16_t depth;
  while (t1.t0(&bdbt, &depth, redraw_key.blending == false)) {
    typename loco_bdbt_Key_t<sizeof(shape_type_t) * 8>::Traverse_t t2;
    t2.i(t1.Output);
    loco_t::shape_type_t shape_type;
    while (t2.t(&bdbt, &shape_type)) {
      #if defined(loco_vulkan)
     // if (bind_array[(int)shape_type] == false) {
        //fan::print((int)shape_type);
        shapes.get_value((int)shape_type, [](auto& shape) {
          gloco->get_context().bind_draw(
            shape.m_pipeline,
            1,
            &shape.m_ssbo.m_descriptor.m_descriptor_set[gloco->get_context().currentFrame]
          );
        });
      //  bind_array[(int)shape_type] = true;
     // }
      #endif
      shape_draw(shape_type, redraw_key, t2.Output);
    }
  }

}
//fan::print("(int)shape_type");
//#if defined(loco_vulkan)
//int index = 0;
//for (auto& shape : bind_array) {
//  if (shape == false) {
//    fan::print("AA", index);
//    shapes.get_value((int)index, [](auto& shape) {
//      gloco->get_context().bind_draw(
//        shape.m_pipeline,
//        1,
//        &shape.m_ssbo.m_descriptor.m_descriptor_set[gloco->get_context().currentFrame]
//      );
//    });
//  }
//  ++index;
//}
//#endif

// TODO!
//m_framebuffer.unbind(get_context());
//
//get_context().opengl.glFlush();
//get_context().opengl.glFinish();

for (auto& i : m_post_draw) {
  i();
}

//get_context().opengl.glFlush();
//get_context().opengl.glFinish();

//m_framebuffer.bind(get_context());