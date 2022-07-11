struct loco_t {

  struct properties_t {
    fan_2d::graphics::font_t* font;
  };

#if defined(loco_rectangle)
  fan_2d::graphics::rectangle_t rectangle;
#endif
#if defined(loco_sprite)
  fan_2d::graphics::sprite_t sprite;
#endif
#if defined(loco_letter)
  fan_2d::graphics::letter_t letter;
#endif
#if defined(loco_rectangle_text_button)
  #if !defined(loco_letter)
  #error need to enable loco_letter to use rectangle_text_button
  #undef loco_rectangle_text_button
  #endif
  fan_2d::graphics::gui::rectangle_text_button_t rectangle_text_button;
#endif

  fan_2d::graphics::gui::be_t button_event;
  fan::opengl::matrices_t matrices;

  void open(const properties_t& p) {
#if defined(loco_rectangle)
    rectangle.open(&context);
#endif
#if defined(loco_sprite)
    sprite.open(&context);
#endif
#if defined(loco_letter)
    letter.open(&context, p.font);
#endif
#if defined(loco_rectangle_text_button)
    rectangle_text_button.open(&context);
#endif

    button_event.open();
  }
  void close() {
#if defined(loco_rectangle)
    rectangle.close(&context);
#endif
#if defined(loco_sprite)
    sprite.close(&context);
#endif
#if defined(loco_letter)
    letter.close(&context);
#endif
#if defined(loco_rectangle_text_button)
    rectangle_text_button.close(&context);
#endif
  }

#if defined(loco_rectangle)
  void push_back(fan::opengl::cid_t* cid, const fan_2d::graphics::rectangle_t::properties_t& p) {
    rectangle.push_back(&context, cid, p);
  }
#endif
#if defined(loco_sprite)
  void push_back(fan::opengl::cid_t* cid, const fan_2d::graphics::sprite_t::properties_t& p) {
    sprite.push_back(&context, cid, p);
  }
#endif
#if defined(loco_rectangle_text_button)
  void push_back(fan::opengl::cid_t* cid, const fan_2d::graphics::gui::rectangle_text_button_t::properties_t& p) {
    rectangle_text_button.push_back(&context, &button_event, &letter, cid, p);
  }
#endif

  fan::opengl::context_t context;
};