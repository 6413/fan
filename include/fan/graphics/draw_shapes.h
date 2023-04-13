// <redraw key> <depth> <shape type> <rest of keys> == block manager

/*
* TODO light draw
*/

for (auto& i : m_draw_queue_light) {
  i();
}

light.draw();

typename loco_bdbt_Key_t<sizeof(redraw_key_t) * 8>::Traverse_t t0;
t0.i(root);
redraw_key_t redraw_key;
while(t0.t(&bdbt, &redraw_key)) {
  if (redraw_key.blending == false) {
    get_context()->opengl.call(get_context()->opengl.glDisable, fan::opengl::GL_BLEND);
  }
  else {
    get_context()->opengl.call(get_context()->opengl.glEnable, fan::opengl::GL_BLEND);
    get_context()->opengl.call(get_context()->opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
  }

  typename loco_bdbt_Key_t<sizeof(uint16_t) * 8, true>::Traverse_t t1;
  t1.i0(t0.Output, redraw_key.blending == false);
  uint16_t depth;
  while (t1.t0(&bdbt, &depth, redraw_key.blending == false)) {
    typename loco_bdbt_Key_t<sizeof(shape_type_t::_t) * 8>::Traverse_t t2;
    t2.i(t1.Output);
    shape_type_t::_t shape_type;
    while (t2.t(&bdbt, &shape_type)) {  
      shape_draw(shape_type, redraw_key, t2.Output);
    }
  }
}

//redraw_t redraw;
//for (uint8_t i = 0; i < 2; i++) {
//  if (i == 0) {
//    get_context()->opengl.call(get_context()->opengl.glDisable, fan::opengl::GL_BLEND);
//  }
//  else {
//    get_context()->opengl.call(get_context()->opengl.glEnable, fan::opengl::GL_BLEND);
//    get_context()->opengl.call(get_context()->opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
//  }
//  redraw.blending ^= 1;
//}
//
//
////redraw_key[depth[shape_type[rest_of_keys[...]]]] = block_manager;
//
//void f() {
//  typename loco_bdbt_Key_t<sizeof(redraw_key_t) * 8>::Traverse_t t0;
//  t0.i(ROOT);
//  redraw_key_t redraw_key;
//  while(t0.t(&bdbt, &redraw_key)) {
//    typename loco_bdbt_Key_t<sizeof(uint16_t) * 8>::Traverse_t t1;
//    t1.i(t0.Output);
//    uint16_t depth;
//    while (t1.t(&bdbt, &depth)) {
//      typename loco_bdbt_Key_t < sizeof(shape_type_t::_t) * 8 > ::Traverse_t t2;
//      t2.i(t1.Output);
//      shape_type_t::_t shape_type;
//      while (t2.t(&bdbt, &shape_type)) {
//        
//        //shape_draw(shape_type);
//      }
//    }
//  }
//}
//
//template <uint32_t depth = 0>
//void traverse_draw(loco_bdbt_NodeReference_t nr, uint32_t draw_mode, auto lambda) {
//  loco_t* loco = get_loco();
//  if constexpr (depth == bm_properties_t::key_t::count + 1) {
//    auto bmn = bm_list.GetNodeByReference(*(shape_bm_NodeReference_t*)&nr);
//    auto bnr = bmn->data.first_block;
//    #ifndef sb_inline_draw
//    draw_queue_helper.push_back([this, loco, draw_mode, bmn, bnr, lambda]() mutable {
//      #endif
//
//      m_current_shader->use(loco->get_context());
//    #if defined(loco_opengl)
//    #if defined (loco_letter)
//    if constexpr (std::is_same<std::remove_pointer_t<decltype(this)>, loco_t::letter_t>::value) {
//      loco->process_block_properties_element<0>(this, &loco->font.image);
//    }
//    #endif
//    #endif
//
//    m_current_shader->set_vec3(loco->get_context(), loco_t::lighting_t::ambient_name, loco->lighting.ambient);
//
//    #if defined(loco_framebuffer)
//    #if defined(sb_is_light)
//    loco->get_context()->opengl.call(loco->get_context()->opengl.glEnable, fan::opengl::GL_BLEND);
//    //loco->get_context()->opengl.call(loco->get_context()->opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
//    loco->get_context()->set_depth_test(false);
//    if constexpr (std::is_same<std::remove_pointer_t<decltype(this)>, sb_shape_name>::value) {
//      loco->get_context()->opengl.call(loco->get_context()->opengl.glBlendFunc, fan::opengl::GL_ONE, fan::opengl::GL_ONE);
//
//      unsigned int attachments[sizeof(loco->color_buffers) / sizeof(loco->color_buffers[0])];
//
//      for (uint8_t i = 0; i < std::size(loco->color_buffers); ++i) {
//        attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
//      }
//
//      loco->get_context()->opengl.call(loco->get_context()->opengl.glDrawBuffers, std::size(attachments), attachments);
//
//    }
//    #endif
//    #endif
//
//    m_current_shader->set_vec2(loco->get_context(), "window_size", loco->get_window()->get_size());
//
//    while (1) {
//      auto node = blocks.GetNodeByReference(bnr);
//      node->data.block.uniform_buffer.bind_buffer_range(
//        loco->get_context(),
//        node->data.block.uniform_buffer.size()
//      );
//
//      node->data.block.uniform_buffer.draw(
//        loco->get_context(),
//        0 * sb_vertex_count,
//        node->data.block.uniform_buffer.size() * sb_vertex_count,
//        draw_mode
//      );
//      if (bnr == bmn->data.last_block) {
//        break;
//      }
//      bnr = node->NextNodeReference;
//    }
//    #if defined(loco_framebuffer)
//    #if defined(sb_is_light)
//    loco->get_context()->opengl.call(loco->get_context()->opengl.glDisable, fan::opengl::GL_BLEND);
//    loco->get_context()->set_depth_test(true);
//    if constexpr (std::is_same<std::remove_pointer_t<decltype(this)>, sb_shape_name>::value) {
//      loco->get_context()->opengl.call(loco->get_context()->opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
//      unsigned int attachments[sizeof(loco->color_buffers) / sizeof(loco->color_buffers[0])];
//
//      for (uint8_t i = 0; i < std::size(loco->color_buffers); ++i) {
//        attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
//      }
//
//      loco->get_context()->opengl.call(loco->get_context()->opengl.glDrawBuffers, 1, attachments);
//    }
//    #endif
//    #endif
//    #ifndef sb_inline_draw
//      });
//    #endif
//
//    #ifndef sb_inline_draw
//    loco->m_draw_queue.insert(loco_t::draw_t{
//      (uint64_t)zdepth,
//      std::vector<fan::function_t<void()>>(draw_queue_helper.begin(), draw_queue_helper.end())
//    });
//
//    draw_queue_helper.clear();
//    #endif
//  }
//  else {
//    //loco_bdbt_Key_t<sizeof(typename instance_properties_t::key_t::get_type<depth>::type) * 8> k;
//    typename loco_bdbt_Key_t<sizeof(typename bm_properties_t::key_t::get_type<depth>::type) * 8>::Traverse_t kt;
//    kt.init(nr);
//    typename bm_properties_t::key_t::get_type<depth>::type o;
//    #if fan_use_uninitialized == 0
//    memset(&o, 0, sizeof(o));
//    #endif
//    while (kt.Traverse(&loco->bdbt, &o)) {
//      // update zdepth here if changes
//      if constexpr (std::is_same_v<decltype(o), uint16_t>) {
//        zdepth = o;
//      }
//      #ifndef sb_inline_draw
//      draw_queue_helper.push_back([this, loco, o, kt, draw_mode]() {
//        #endif
//        m_current_shader->use(loco->get_context());
//      loco->process_block_properties_element(this, o);
//      #ifndef sb_inline_draw
//        });
//      #endif
//      traverse_draw<depth + 1>(kt.Output, draw_mode, lambda);
//    }
//  }
//}
