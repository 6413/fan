struct loco_t {

  struct properties_t {
    fan::opengl::matrices_t* matrices;
  };

  static constexpr uint32_t max_depths = 100;

  struct {

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

  }element_depth[max_depths];

  using mouse_input_data_t = fan_2d::graphics::gui::be_t::mouse_input_data_t;
  using mouse_move_data_t = fan_2d::graphics::gui::be_t::mouse_move_data_t;

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
    for (uint32_t depth = 0; depth < max_depths; depth++) {

      element_depth[depth].button_event.open();

    #if defined(loco_letter)
      font.open(&context, loco_font);
    #endif

    #if defined(loco_rectangle)
      element_depth[depth].rectangle.open(&context);
      element_depth[depth].rectangle.bind_matrices(&context, p.matrices);
      element_depth[depth].rectangle.enable_draw(&context);
    #endif
    #if defined(loco_sprite)
      element_depth[depth].sprite.open(&context);
      element_depth[depth].sprite.bind_matrices(&context, p.matrices);
      element_depth[depth].sprite.enable_draw(&context);
    #endif
    #if defined(loco_rectangle_text_button)
      element_depth[depth].rectangle_text_button.open(&context);
      element_depth[depth].rectangle_text_button.bind_matrices(&context, p.matrices);
      element_depth[depth].rectangle_text_button.enable_draw(&context);
    #endif
    #if defined(loco_letter)
      element_depth[depth].letter.open(&context, &font);
      element_depth[depth].letter.bind_matrices(&context, p.matrices);
      element_depth[depth].letter.enable_draw(&context);
    #endif

    }

    focus_shape_type = fan::uninitialized;
  }
  void close(const properties_t& p) {
    for (uint32_t depth = 0; depth < max_depths; depth++) {
      #if defined(loco_rectangle)
        element_depth[depth].rectangle.unbind_matrices(&context, p.matrices);
        element_depth[depth].rectangle.close(&context);
      #endif
      #if defined(loco_sprite)
        element_depth[depth].sprite.unbind_matrices(&context, p.matrices);
        element_depth[depth].sprite.close(&context);
      #endif
      #if defined(loco_letter)
        element_depth[depth].letter.unbind_matrices(&context, p.matrices);
        element_depth[depth].letter.close(&context);
      #endif
      #if defined(loco_rectangle_text_button)
        element_depth[depth].rectangle_text_button.unbind_matrices(&context, p.matrices);
        element_depth[depth].rectangle_text_button.close(&context);
      #endif
    }
  }

  #if defined(loco_rectangle)
    void push_back(uint32_t depth, fan::opengl::cid_t* cid, const fan_2d::graphics::rectangle_t::properties_t& p) {
      element_depth[depth].rectangle.push_back(&context, cid, p);
    }
  #endif
  #if defined(loco_sprite)
    void push_back(uint32_t depth, fan::opengl::cid_t* cid, const fan_2d::graphics::sprite_t::properties_t& p) {
      element_depth[depth].sprite.push_back(&context, cid, p);
    }
  #endif
  #if defined(loco_rectangle_text_button)
    void push_back(uint32_t depth, uint32_t input_depth, uint32_t* id, const fan_2d::graphics::gui::rectangle_text_button_t::properties_t& p) {
      *id = element_depth[depth].rectangle_text_button.push_back(&context, &element_depth[input_depth].button_event, &element_depth[depth].letter, p);
    }
  #endif

    uint32_t push_back(uint32_t depth, const fan_2d::graphics::gui::be_t::properties_t& p) {
      return element_depth[depth].button_event.push_back(p);
    }

  /*
  
    when any feed function comes loco will check focus_shape_type. if its  uninitialized loco will query input if input is on something if its related with something. It will assign focus_shape_type to what its supposed to be.
  */

  void feed_mouse_move(fan::opengl::context_t* context, const fan::vec2& mouse_position) {
    for (uint32_t depth = max_depths; depth--; ) {
      element_depth[depth].button_event.feed_mouse_move(context, mouse_position, depth);
    }
  }

  void feed_mouse_input(fan::opengl::context_t* context, uint16_t button, fan::key_state key_state, const fan::vec2& mouse_position) {
    for (uint32_t depth = max_depths; depth--; ) {
      element_depth[depth].button_event.feed_mouse_input(context, button, key_state, mouse_position, depth);
    }
  }

  void feed_keyboard(fan::opengl::context_t* context, uint16_t key, fan::key_state key_state) {
    //button_event.feed_keyboard(context, key, key_state);
  }

  fan::opengl::context_t context;
};