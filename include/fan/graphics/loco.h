struct loco_t {

  struct properties_t {
    fan::opengl::matrices_t* matrices;
  };

  #if defined(loco_rectangle)
    fan_2d::graphics::rectangle_t rectangle;
  #endif
  #if defined(loco_sprite)
    fan_2d::graphics::sprite_t sprite;
  #endif
  #if defined(loco_letter)
    #if !defined(loco_font)
      #define loco_font "fonts/bitter"
    #endif
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

  uint32_t focus_shape_type;
  uint32_t focus_shape_id;

  #if defined(loco_letter)
    fan_2d::graphics::font_t font;
  #endif

  enum class shape_type_e{
    #if defined(loco_rectangle_text_button)
      rectangle_text_button
    #endif
  };

  void open(const properties_t& p) {
    button_event.open();

    #if defined(loco_letter)
      font.open(&context, loco_font);
    #endif

    #if defined(loco_rectangle)
      rectangle.open(&context);
      rectangle.bind_matrices(&context, p.matrices);
    #endif
    #if defined(loco_sprite)
      sprite.open(&context);
      sprite.bind_matrices(&context, p.matrices);
    #endif
    #if defined(loco_letter)
      letter.open(&context, &font);
      letter.bind_matrices(&context, p.matrices);
    #endif
    #if defined(loco_rectangle_text_button)
      rectangle_text_button.open(&context);
      rectangle_text_button.bind_matrices(&context, p.matrices);
    #endif

    focus_shape_type = fan::uninitialized;
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

  void feed_mouse_move(fan::opengl::context_t* context, const fan::vec2& mouse_position) {
    button_event.feed_mouse_move(context, mouse_position);
  }

  void feed_mouse_input(fan::opengl::context_t* context, uint16_t button, fan::key_state key_state, const fan::vec2& mouse_position) {
    button_event.feed_mouse_input(context, button, key_state, mouse_position);
  }

  fan::opengl::context_t context;
};