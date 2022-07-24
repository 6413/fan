struct loco_t {

  struct properties_t {
    fan::opengl::matrices_t* matrices;
    fan::opengl::context_t* context;
  };

  static constexpr uint32_t max_depths = 100;

  #if defined(loco_rectangle)
    struct rectangle_t : fan_2d::graphics::rectangle_t {
    private:
      using fan_2d::graphics::rectangle_t::push_back;
    public:
      void push_back(fan::opengl::cid_t* cid, const properties_t& p) {
        loco_t* loco = OFFSETLESS(this, loco_t, rectangle);
        fan_2d::graphics::rectangle_t::push_back(loco->context, cid, p);
      }
    }rectangle;
  #endif
  #if defined(loco_sprite)
    struct sprite_t : fan_2d::graphics::sprite_t {
      void push_back(fan::opengl::cid_t* cid, const properties_t& p) {
        loco_t* loco = OFFSETLESS(this, loco_t, sprite);
        fan_2d::graphics::sprite_t::push_back(loco->context, cid, p);
      }
    private:
      using fan_2d::graphics::sprite_t::push_back;
    }sprite;
  #endif
  #if defined(loco_letter)
  #if !defined(loco_font)
  #define loco_font "fonts/bitter"
  #endif
    struct letter_t : fan_2d::graphics::letter_t {
      void push_back(fan::opengl::cid_t* cid, const properties_t& p) {
        loco_t* loco = OFFSETLESS(this, loco_t, letter);
        fan_2d::graphics::letter_t::push_back(loco->context, cid, p);
      }
    private:
      using fan_2d::graphics::letter_t::push_back;
      }letter;
  #endif
  #if defined(loco_rectangle_text_button)
  #if !defined(loco_letter)
  #error need to enable loco_letter to use rectangle_text_button
  #undef loco_rectangle_text_button
  #endif
    struct button_t : fan_2d::graphics::gui::rectangle_text_button_t {
      uint32_t push_back(uint32_t input_depth, const properties_t& p) {
        loco_t* loco = OFFSETLESS(this, loco_t, button);
        return fan_2d::graphics::gui::rectangle_text_button_t::push_back(loco->context, &loco->element_depth[input_depth].input_hitbox, &loco->letter, p);
      }
    private:
      using fan_2d::graphics::gui::rectangle_text_button_t::push_back;
    }button;
  #endif

  struct {
  fan_2d::graphics::gui::be_t input_hitbox;
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
    context = p.context;

    for (uint32_t depth = 0; depth < max_depths; depth++) {

      element_depth[depth].input_hitbox.open();
    }

    #if defined(loco_letter)
      font.open(context, loco_font);
    #endif

    #if defined(loco_rectangle)
      rectangle.open(context);
      rectangle.bind_matrices(context, p.matrices);
      rectangle.enable_draw(context);
    #endif
    #if defined(loco_sprite)
      sprite.open(context);
      sprite.bind_matrices(context, p.matrices);
      sprite.enable_draw(context);
    #endif
    #if defined(loco_rectangle_text_button)
      button.open(context);
      button.bind_matrices(context, p.matrices);
      button.enable_draw(context);
    #endif
    #if defined(loco_letter)
      letter.open(context, &font);
      letter.bind_matrices(context, p.matrices);
      letter.enable_draw(context);
    #endif

    focus_shape_type = fan::uninitialized;
  }
  void close(const properties_t& p) {
    #if defined(loco_rectangle)
      rectangle.unbind_matrices(context, p.matrices);
      rectangle.close(context);
    #endif
    #if defined(loco_sprite)
      sprite.unbind_matrices(context, p.matrices);
      sprite.close(context);
    #endif
    #if defined(loco_letter)
      letter.unbind_matrices(context, p.matrices);
      letter.close(context);
    #endif
    #if defined(loco_rectangle_text_button)
      button.unbind_matrices(context, p.matrices);
      button.close(context);
    #endif
  }

  uint32_t push_back_input_hitbox(uint32_t depth, const fan_2d::graphics::gui::be_t::properties_t& p) {
    return element_depth[depth].input_hitbox.push_back(p);
  }

  /*
  
    when any feed function comes loco will check focus_shape_type. if its  uninitialized loco will query input if input is on something if its related with something. It will assign focus_shape_type to what its supposed to be.
  */

  void feed_mouse_move(fan::opengl::context_t* context, const fan::vec2& mouse_position) {
    for (uint32_t depth = max_depths; depth--; ) {
       uint32_t r = element_depth[depth].input_hitbox.feed_mouse_move(context, mouse_position, depth);
       if (r == 0) {
         break;
       }
    }
  }

  void feed_mouse_input(fan::opengl::context_t* context, uint16_t button, fan::key_state key_state, const fan::vec2& mouse_position) {
    for (uint32_t depth = max_depths; depth--; ) {
      element_depth[depth].input_hitbox.feed_mouse_input(context, button, key_state, mouse_position, depth);
     /* if (!element_depth[depth].can_we_go_behind()) {
        break;
      }*/
    }
  }

  void feed_keyboard(fan::opengl::context_t* context, uint16_t key, fan::key_state key_state) {
    //input_hitbox.feed_keyboard(context, key, key_state);
  }

  fan::opengl::context_t* context;
};