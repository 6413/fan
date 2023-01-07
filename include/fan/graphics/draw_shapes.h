#if defined(loco_line)
  line.draw();
#endif
#if defined(loco_rectangle)
  rectangle.draw();
#endif
#if defined(loco_circle)
  circle.draw();
#endif
#if defined(loco_yuv420p)
  yuv420p.draw();
#endif
#if defined(loco_sprite)
  // can be moved
  sprite.draw();
#endif
#if defined(loco_letter)
  // loco_t::text gets drawn here as well as it uses letter
  letter.draw();
#endif
#if defined(loco_text_box)
  text_box.draw();
#endif
#if defined(loco_button)
  button.draw();
#endif
#if defined(loco_model_3d)
  model.draw();
#endif
#if defined(loco_post_process)
  post_process.draw();
#endif