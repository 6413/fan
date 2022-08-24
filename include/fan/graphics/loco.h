#include _FAN_PATH(graphics/graphics.h)

struct loco_t;

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

  #define vfi_var_name vfi
  #include _FAN_PATH(graphics/gui/vfi.h)
  using mouse_move_data_t = vfi_t::mouse_move_data_t;
  using mouse_button_data_t = vfi_t::mouse_button_data_t;

  vfi_t vfi_var_name;

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
      node->data.viewport_id->get_viewport_size(),
      get_window()->get_size()
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
  #endif
  #if defined(loco_button)
    #if !defined(loco_letter)
      #define loco_letter
    #endif
    #if !defined(loco_text)
      #define loco_text
    #endif
    #if !defined(loco_text_box)
      #define loco_text_box
    #endif
  #endif

  #if defined(loco_line)
    #define sb_shape_var_name line
    #include _FAN_PATH(graphics/opengl/2D/objects/line.h)
    line_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_rectangle)
    #define sb_shape_var_name rectangle
    #include _FAN_PATH(graphics/opengl/2D/objects/rectangle.h)
    rectangle_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_yuv420p)
    #define sb_shape_var_name yuv420p
    #define sb_sprite_name yuv420p_t
    #include _FAN_PATH(graphics/opengl/2D/objects/yuv420p.h)
    yuv420p_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_sprite)
    #define sb_shape_var_name sprite
    #define sb_sprite_name sprite_t
    #include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
    sprite_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_letter)
    #if !defined(loco_font)
      #define loco_font "fonts/bitter"
    #endif
    #define sb_shape_var_name letter
    #include _FAN_PATH(graphics/opengl/2D/objects/letter_renderer.h)
    letter_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_text)
    #define sb_shape_var_name text
    #include _FAN_PATH(graphics/opengl/2D/objects/text_renderer.h)
    using text_t = text_renderer_t;
    text_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_button)
    #define sb_shape_var_name button
    #include _FAN_PATH(graphics/gui/rectangle_text_button.h)
    button_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_post_process)
    #define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/2D/effects/post_process.vs)
    #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/effects/post_process.fs)
    #define sb_post_process_var_name post_process
    #define sb_sprite_name post_sprite_t
    #include _FAN_PATH(graphics/opengl/2D/effects/post_process.h)
    post_process_t sb_post_process_var_name;
    #undef sb_post_process_var_name
  #endif

  #if defined(loco_letter)
    fan_2d::graphics::font_t font;
  #endif

  enum class shape_type_e{
    #if defined(loco_rectangle_text_button)
      rectangle_text_button
    #endif
  };

  void open(const properties_t& p) {

    vfi_var_name.open();

    #ifdef loco_window
      window.open(fan::vec2(1920, 1080));
    #else
      window = p.window;
    #endif

    loco_bdbt_open(&bdbt);

    get_window()->add_buttons_callback(this, [](fan::window_t* window, uint16_t key, fan::key_state key_state, void* user_ptr) {
      loco_t& loco = *(loco_t*)user_ptr;
      fan::vec2 window_size = window->get_size();
      loco.feed_mouse_button(key, key_state, loco.get_mouse_position());
    });

    get_window()->add_keys_callback(this, [](fan::window_t* window, uint16_t key, fan::key_state key_state, void* user_ptr) {
      loco_t& loco = *(loco_t*)user_ptr;
      loco.feed_keyboard(key, key_state);
    });

    get_window()->add_mouse_move_callback(this, [](fan::window_t* window, const fan::vec2i& mouse_position, void* user_ptr) {
      loco_t& loco = *(loco_t*)user_ptr;
      fan::vec2 window_size = window->get_size();
      loco.feed_mouse_move(loco.get_mouse_position());
    });

    context.open();
    context.bind_to_window(&window);

    #if defined(loco_letter)
      font.open(get_context(), loco_font);
    #endif

    #if defined(loco_line)
      line.open();
    #endif
    #if defined(loco_rectangle)
      rectangle.open();
    #endif
   #if defined(loco_yuv420p)
      yuv420p.open();
    #endif
    #if defined(loco_sprite)
      sprite.open();
    #endif
    #if defined(loco_letter)
      letter.open();
    #endif
    #if defined(loco_text)
      text.open();
    #endif
    #if defined(loco_button)
      button.open();
    #endif
    #if defined(loco_post_process)
      fan::opengl::core::renderbuffer_t::properties_t rp;
      rp.size = get_window()->get_size();
      if (post_process.open(rp)) {
        fan::throw_error("failed to initialize frame buffer");
      }
    #endif

    m_write_queue.open();
  }
  void close() {

    vfi.close();

    loco_bdbt_close(&bdbt);

    #if defined(loco_line)
      line.close();
    #endif
    #if defined(loco_rectangle)
      rectangle.close();
    #endif
    #if defined(loco_yuv420p)
      yuv420p.close();
    #endif
    #if defined(loco_sprite)
      sprite.close();
    #endif
    #if defined(loco_letter)
      letter.close();
    #endif
    #if defined(loco_text)
      text.close();
    #endif
    #if defined(loco_button)
      button.close();
    #endif
    #if defined(loco_post_process)
      post_process.close();
    #endif

    m_write_queue.close();
  }

  vfi_t::shape_id_t push_back_input_hitbox(uint32_t depth, const vfi_t::properties_t& p) {
    return vfi.push_shape(p);
  }
 /* uint32_t push_back_keyboard_event(uint32_t depth, const fan_2d::graphics::gui::ke_t::properties_t& p) {
    return element_depth[depth].keyboard_event.push_back(p);
  }*/

  void feed_mouse_move(const fan::vec2& mouse_position) {
    vfi.feed_mouse_move(this, mouse_position);
   
  }

  void feed_mouse_button(uint16_t button, fan::key_state key_state, const fan::vec2& mouse_position) {
    vfi.feed_mouse_button(this, button, key_state);
  }

  void feed_keyboard(uint16_t key, fan::key_state key_state) {
    /*for (uint32_t depth = max_depths; depth--; ) {
      uint32_t r = element_depth[depth].keyboard_event.feed_keyboard(this, key, key_state, depth);
      if (r == 0) {
        break;
      }
    }*/
  }

  void process_frame(const std::function<void()>& f) {
    #if fan_renderer == fan_renderer_opengl
    // get_context()->opengl.call(get_context()->opengl.glClearColor, 1, 0, 0, 0);
    get_context()->opengl.call(get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
    #endif

    m_write_queue.process(get_context());

    f();
    #if defined(loco_line)
      line.draw();
    #endif
    #if defined(loco_rectangle)
      rectangle.draw();
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
    #if defined(loco_button)
      button.draw();
    #endif
    #if defined(loco_post_process)
      post_process.draw();
    #endif

    #ifdef loco_window
      get_context()->render(get_window());
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

  fan::vec2 transform_matrix(const fan::vec2& position) {
    fan::vec2 window_size = get_window()->get_size();
    // not custom ortho friendly - made for -1 1
    return position / window_size * 2 - 1;
  }

  fan::vec2 get_mouse_position() {
    // not custom ortho friendly - made for -1 1
    //return transform_matrix(get_window()->get_mouse_position());
    return get_window()->get_mouse_position();
  }
  fan::vec2 get_mouse_position(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    fan::vec2 x;
    x.x = (get_mouse_position().x - viewport_position.x - viewport_size.x / 2) / (viewport_size.x / 2);
    x.y = (get_mouse_position().y - viewport_position.y - viewport_size.y / 2) / (viewport_size.y / 2) + (viewport_position.y / viewport_size.y) * 2;
    return x;
  }

  fan::opengl::core::uniform_write_queue_t m_write_queue;

  void loop(const auto& lambda) {
    while (1) {
      uint32_t window_event = get_window()->handle_events();
      if(window_event & fan::window_t::events::close){
        get_window()->close();
        break;
      }
      
      process_frame(lambda);
    }
  }

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