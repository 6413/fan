#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(graphics/gui/be.h)

#define BDBT_set_prefix loco_bdbt
#define BDBT_set_type_node uint16_t
#define BDBT_set_BitPerNode 2
#define BDBT_set_declare_basic_types 1
#define BDBT_set_declare_rest 1
#define BDBT_set_declare_Key 0
#define BDBT_set_BaseLibrary 1
#include _FAN_PATH(BDBT/BDBT.h)

#define BDBT_set_prefix loco_bdbt
#define BDBT_set_type_node uint16_t
#define BDBT_set_KeySize 0
#define BDBT_set_BitPerNode 2
#define BDBT_set_declare_basic_types 0
#define BDBT_set_declare_rest 0
#define BDBT_set_declare_Key 1
#define BDBT_set_base_prefix loco_bdbt
#define BDBT_set_BaseLibrary 1
#include _FAN_PATH(BDBT/BDBT.h)

#include _FAN_PATH(graphics/opengl/uniform_block.h)

struct loco_t {

  struct properties_t {
  #ifndef loco_window
    fan::window_t* window;
  #endif
  #ifndef loco_context
    fan::context_t* context;
  #endif
  };

  static constexpr uint32_t max_depths = 2;

  fan::window_t* get_window() {
  #ifdef loco_window
    return &window;
  #else
    return window;
  #endif
  }

  fan::opengl::context_t* get_context() {
  #ifdef loco_context
    return &context;
  #else
    return context;
  #endif
  }

  void process_block_properties_element(auto* shape, fan::opengl::matrices_list_NodeReference_t matrices_id) {
    auto node = fan::opengl::matrices_list_GetNodeByReference(&get_context()->matrices_list, matrices_id);
    shape->m_shader.set_matrices(get_context(), node->data.matrices_id);
  }

  void process_block_properties_element(auto* shape, fan::opengl::viewport_list_NodeReference_t viewport_id) {
    auto node = fan::opengl::viewport_list_GetNodeByReference(&get_context()->viewport_list, viewport_id);
    node->data.viewport_id->set_viewport(
      get_context(),
      node->data.viewport_id->get_viewport_position(),
      node->data.viewport_id->get_viewport_size()
    );
  }

  template <uint8_t n>
  void process_block_properties_element(auto* shape, fan::opengl::textureid_t<n> tid) {
    auto node = fan::opengl::image_list_GetNodeByReference(&get_context()->image_list, tid);
    shape->m_shader.set_int(get_context(), tid.name, n);
    get_context()->opengl.call(get_context()->opengl.glActiveTexture, fan::opengl::GL_TEXTURE0 + n);
    get_context()->opengl.call(get_context()->opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, node->data.texture_id);
  }

  loco_bdbt_t bdbt;

  // automatically gets necessary macros for shapes

  #if defined(loco_text_box)
    #if !defined(loco_letter)
      #define loco_letter
    #endif
    #if !defined(loco_text)
      #define loco_text
    #endif
    #if !defined(loco_box)
      #define loco_box
    #endif
  #endif
  #if defined(loco_button)
    #if !defined(loco_letter)
      #define loco_letter
    #endif
    #if !defined(loco_text)
      #define loco_text
    #endif
    #if !defined(loco_box)
      #define loco_box
    #endif
    #if !defined(loco_text_box)
      #define loco_text_box
    #endif
  #endif

  #if defined(loco_rectangle)
    #include _FAN_PATH(graphics/opengl/2D/objects/rectangle.h)
    rectangle_t rectangle;
  #endif
  #if defined(loco_sprite)
    #define sb_sprite_name sprite_t
    #include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
    sprite_t sprite;
  #endif
  #if defined(loco_letter)
    #if !defined(loco_font)
      #define loco_font "fonts/bitter"
    #endif
    #include _FAN_PATH(graphics/opengl/2D/objects/letter_renderer.h)
    letter_t letter;
  #endif
  #if defined(loco_text)
    #include _FAN_PATH(graphics/opengl/2D/objects/text_renderer.h)
    using text_t = text_renderer_t;
    text_t text;
  #endif
  #if defined(loco_box)
    #include _FAN_PATH(graphics/gui/rectangle_box.h)
    using box_t = rectangle_box_t;
    box_t box;
  #endif
  #if defined(loco_text_box)
    #include _FAN_PATH(graphics/gui/rectangle_text_box.h)
    using text_box_t = rectangle_text_box_t;
    text_box_t text_box;
  #endif
  #if defined(loco_button)
    #include _FAN_PATH(graphics/gui/rectangle_text_button.h)
    struct button_t : rectangle_text_button_t {
      void push_back(uint32_t input_depth, fan::opengl::cid_t* cid, properties_t& p) {
        loco_t* loco = OFFSETLESS(this, loco_t, button);
        rectangle_text_button_t::push_back(loco, cid, &loco->element_depth[input_depth].input_hitbox, p);
      }
    private:
      using rectangle_text_button_t::push_back;
    }button;
  #endif
  #if defined(loco_post_process)
    #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/effects/post_process.vs)
    #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/effects/post_process.fs)
    #define sb_sprite_name post_sprite_t
    #include _FAN_PATH(graphics/opengl/2D/effects/post_process.h)
    post_process_t post_process;
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

    #ifdef loco_window
      window.open();
    #else
      window = p.window;
    #endif

    loco_bdbt_open(&bdbt);

    get_window()->add_keys_callback(this, [](fan::window_t* window, uint16_t key, fan::key_state key_state, void* user_ptr) {
      loco_t& loco = *(loco_t*)user_ptr;
      fan::vec2 window_size = window->get_size();
      // not custom ortho friendly - made for -1 1
      loco.feed_mouse_input(loco.get_context(), key, key_state, fan::cast<f32_t>(window->get_mouse_position()) / window_size * 2 - 1);
    });

    get_window()->add_mouse_move_callback(this, [](fan::window_t* window, const fan::vec2i& mouse_position, void* user_ptr) {
      loco_t& loco = *(loco_t*)user_ptr;
      fan::vec2 window_size = window->get_size();
      // not custom ortho friendly - made for -1 1
      loco.feed_mouse_move(loco.get_context(), fan::cast<f32_t>(mouse_position) / window_size * 2 - 1);
    });

    context.open();
    context.bind_to_window(&window);

    for (uint32_t depth = 0; depth < max_depths; depth++) {

      element_depth[depth].input_hitbox.open();
    }

    #if defined(loco_letter)
      font.open(get_context(), loco_font);
    #endif

    #if defined(loco_rectangle)
      rectangle.open(this);
    #endif
    #if defined(loco_sprite)
      sprite.open(this);
    #endif
    #if defined(loco_letter)
      letter.open(this);
    #endif
    #if defined(loco_text)
      text.open(this);
    #endif
    #if defined(loco_box)
      box.open(this);
    #endif
    #if defined(loco_text_box)
      text_box.open(this);
    #endif
    #if defined(loco_button)
      button.open(this);
    #endif
    #if defined(loco_post_process)
      fan::opengl::core::renderbuffer_t::properties_t rp;
      rp.size = get_window()->get_size();
      if (post_process.open(this, rp)) {
        fan::throw_error("failed to initialize frame buffer");
      }
      post_process.start_capture(this);
    #endif

    focus_shape_type = fan::uninitialized;

    m_write_queue.open();
  }
  void close(const properties_t& p) {

    loco_bdbt_close(&bdbt);

    #if defined(loco_rectangle)
      rectangle.close(this);
    #endif
    #if defined(loco_sprite)
      sprite.close(this);
    #endif
    #if defined(loco_letter)
      letter.close(this);
    #endif
    #if defined(loco_text)
      text.close(this);
    #endif
    #if defined(loco_box)
      box.close(this);
    #endif
    #if defined(loco_text_box)
      text_box.close(this);
    #endif
    #if defined(loco_button)
      button.close(this);
    #endif
    #if defined(loco_post_process)
      post_process.close(this);
    #endif

    m_write_queue.close();
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

  uint32_t process_frame() {
    #ifdef loco_window
      uint32_t window_event = get_window()->handle_events();
      if(window_event & fan::window_t::events::close){
        get_window()->close();
        return window_event;
      }
    #endif
      #if fan_renderer == fan_renderer_opengl
      //get_context()->opengl.call(get_context()->opengl.glDepthMask, 0);
     // get_context()->opengl.call(get_context()->opengl.glClearColor, 1, 0, 0, 0);
      get_context()->opengl.call(get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
      #endif

      m_write_queue.process(get_context());

    #if defined(loco_rectangle)
      rectangle.draw(this);
    #endif
    #if defined(loco_sprite)
      // can be moved
      sprite.draw(this);
    #endif
    #if defined(loco_letter)
      // loco_t::text gets drawn here as well as it uses letter
      letter.draw(this);
    #endif
    #if defined(loco_box)
      box.draw(this);
    #endif
    #if defined(loco_post_process)
      post_process.draw(this);
    #endif

    #ifdef loco_window
      get_context()->render(get_window());
      return window_event;
    #else
      return 0;
    #endif
  }

  bool window_open(uint32_t event) {
    return event != fan::window_t::events::close;
  }
  uint32_t get_fps() {
    return get_window()->get_fps();
  }

  void set_vsync(bool flag) {
    get_context()->set_vsync(get_window(), flag);
  }

  fan::opengl::core::uniform_write_queue_t m_write_queue;

protected:

  #ifdef loco_window
    fan::window_t window;
  #else
    fan::window_t* window;
  #endif

  #ifdef loco_context
    fan::opengl::context_t context;
  #else
    fan::opengl::context_t* context;
  #endif

};