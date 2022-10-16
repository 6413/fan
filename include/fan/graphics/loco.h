#ifndef loco_vulkan
  #define loco_opengl
#endif

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(time/timer.h)

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

#define loco_vfi

#if defined(loco_sprite)
  #include _FAN_PATH(graphics/opengl/texture_pack.h)
#endif

#if defined(loco_text_box)
  #define ETC_WED_set_BaseLibrary 1
  #define ETC_WED_set_Prefix wed
  #include _FAN_PATH(ETC/WED/WED.h)
#endif

#if defined(loco_model_3d)
  extern "C" {
    #define FAST_OBJ_IMPLEMENTATION
    #include _FAN_PATH(graphics/fast_obj/fast_obj.h)
  }
  #include _FAN_PATH(graphics/transform_interpolator.h)
#endif

struct loco_t {

#if defined(loco_vfi)

  #define vfi_var_name vfi
  #include _FAN_PATH(graphics/gui/vfi.h)

  struct mouse_move_data_t : vfi_t::mouse_move_data_t {
    mouse_move_data_t(const vfi_t::mouse_move_data_t& mm) : vfi_t::mouse_move_data_t(mm) {

    }

    fan::graphics::cid_t* cid;
  };
  struct mouse_button_data_t : vfi_t::mouse_button_data_t{
    mouse_button_data_t(const vfi_t::mouse_button_data_t& mm) : vfi_t::mouse_button_data_t(mm) {

    }

    fan::graphics::cid_t* cid;
  };
  struct keyboard_data_t : vfi_t::keyboard_data_t {
    keyboard_data_t(const vfi_t::keyboard_data_t& mm) : vfi_t::keyboard_data_t(mm) {

    }

    fan::graphics::cid_t* cid;
  };

  struct text_data_t : vfi_t::text_data_t {
    text_data_t(const vfi_t::text_data_t& mm) : vfi_t::text_data_t(mm) {

    }

    fan::graphics::cid_t* cid;
  };

  using mouse_move_cb_t = fan::function_t<int(const mouse_move_data_t&)>;
  using mouse_button_cb_t = fan::function_t<int(const mouse_button_data_t&)>;
  using keyboard_cb_t = fan::function_t<int(const keyboard_data_t&)>;
  using text_cb_t = fan::function_t<int(const text_data_t&)>;

  vfi_t vfi_var_name;

#endif

  #ifdef loco_window
    using mouse_buttons_cb_data_t = fan::window_t::mouse_buttons_cb_data_t;
    using keyboard_keys_cb_data_t = fan::window_t::keyboard_keys_cb_data_t;
    using keyboard_key_cb_data_t = fan::window_t::keyboard_key_cb_data_t;
    using text_cb_data_t = fan::window_t::text_cb_data_t;
    using mouse_move_cb_data_t = fan::window_t::mouse_move_cb_data_t;
    using close_cb_data_t = fan::window_t::close_cb_data_t;
    using resize_cb_data_t = fan::window_t::resize_cb_data_t;
    using move_cb_data_t = fan::window_t::move_cb_data_t;
  #endif

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

  fan::graphics::context_t* get_context() {
  #ifdef loco_context
    return &context;
  #else
    return context;
  #endif
  }

  void process_block_properties_element(auto* shape, fan::graphics::matrices_list_NodeReference_t matrices_id) {
    shape->m_shader.set_matrices(get_context(), get_context()->matrices_list[matrices_id].matrices_id);
  }

  void process_block_properties_element(auto* shape, fan::graphics::viewport_list_NodeReference_t viewport_id) {
    auto data = &get_context()->viewport_list[viewport_id];
    data->viewport_id->set(
      get_context(),
      data->viewport_id->get_position(),
      data->viewport_id->get_size(),
      get_window()->get_size()
    );
  }

 /* template <uint8_t n>
  void process_block_properties_element(auto* shape, fan::graphics::textureid_t<n> tid) {
    shape->m_shader.set_int(get_context(), tid.name, n);
    get_context()->opengl.call(get_context()->opengl.glActiveTexture, fan::opengl::GL_TEXTURE0 + n);
    get_context()->opengl.call(get_context()->opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, get_context()->image_list[tid].texture_id);
  }*/

  loco_bdbt_t bdbt;

  fan::ev_timer_t ev_timer;

  // automatically gets necessary macros for shapes

  #if defined(loco_text_box)
      #define loco_rectangle
      #define loco_letter
      #define loco_text
  #endif
  #if defined(loco_button)
    #define loco_letter
    #define loco_text
  #endif
  #if defined(loco_menu_maker)
    #define loco_letter
    #define loco_text
    #define loco_text_box
    #define loco_button
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
    #include _FAN_PATH(graphics/gui/button.h)
    button_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_text_box)
    #include _FAN_PATH(graphics/gui/fed.h)
    #define sb_shape_var_name text_box
    #include _FAN_PATH(graphics/gui/text_box.h)
    text_box_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_menu_maker)
    #include "wrappers/menu_maker.h"
  #endif
  #if defined(loco_menu_maker)
    #include "wrappers/dropdown.h"
  #endif
  #if defined(loco_model_3d)
    #define sb_shape_var_name model
    #include _FAN_PATH(graphics/opengl/3D/objects/model.h)
    model_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_post_process)
    #define sb_post_process_var_name post_process
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
      window.open(fan::vec2(800, 800));
    #else
      window = p.window;
    #endif

    loco_bdbt_open(&bdbt);

    get_window()->add_buttons_callback([this](const mouse_buttons_cb_data_t& d) {
      fan::vec2 window_size = get_window()->get_size();
      feed_mouse_button(d.button, d.state, get_mouse_position());
    });

    get_window()->add_keys_callback([&](const keyboard_keys_cb_data_t& d) {
      feed_keyboard(d.key, d.state);
    });

    get_window()->add_mouse_move_callback([&](const mouse_move_cb_data_t& d) {
      feed_mouse_move(get_mouse_position());
    });

    get_window()->add_text_callback([&](const fan::window_t::text_cb_data_t& d) {
      feed_text(d.character);
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
    #if defined(loco_text_box)
      text_box.open();
    #endif
    #if defined(loco_menu_maker)
      menu_maker.open();
      dropdown.open();
    #endif
    #if defined(loco_model_3d)
      model.open();
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
    #if defined(loco_text_box)
      text_box.close();
    #endif
    #if defined(loco_button)
      button.close();
    #endif
    #if defined(loco_menu_maker)
      dropdown.close();
      menu_maker.close();
    #endif
    #if defined(loco_model_3d)
      model.close();
    #endif
    #if defined(loco_post_process)
      post_process.close();
    #endif

    #ifndef loco_vulkan
    m_write_queue.close();
    #endif
  }

  vfi_t::shape_id_t push_back_input_hitbox(const vfi_t::properties_t& p) {
    return vfi.push_shape(p);
  }
 /* uint32_t push_back_keyboard_event(uint32_t depth, const fan_2d::graphics::gui::ke_t::properties_t& p) {
    return element_depth[depth].keyboard_event.push_back(p);
  }*/

  void feed_mouse_move(const fan::vec2& mouse_position) {
    vfi.feed_mouse_move(mouse_position);
  }

  void feed_mouse_button(uint16_t button, fan::mouse_state mouse_state, const fan::vec2& mouse_position) {
    vfi.feed_mouse_button(button, mouse_state);
  }

  void feed_keyboard(uint16_t key, fan::keyboard_state keyboard_state) {
    vfi.feed_keyboard(key, keyboard_state);
  }

  void feed_text(wchar_t key) {
    vfi.feed_text(key);
  }

  void process_frame() {
    #if defined(loco_opengl)
     //get_context()->opengl.call(get_context()->opengl.glClearColor, 1, 1, 1, 1);
      get_context()->opengl.call(get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
    #endif

    #ifdef loco_post_process
      post_process.start_capture();
    #endif

    m_write_queue.process(get_context());


    #ifdef loco_window
      #if defined(loco_opengl)
        #include "draw_shapes.h"
        get_context()->render(get_window());
      #elif defined(loco_vulkan)
        get_context()->render(get_window(),
          [this] {
            #include "draw_shapes.h"
          }
        );
      #endif
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

  fan::graphics::core::uniform_write_queue_t m_write_queue;

  void loop(const auto& lambda) {
    while (1) {
      uint32_t window_event = get_window()->handle_events();
      if(window_event & fan::window_t::events::close){
        get_window()->close();
        break;
      }

      lambda();

      ev_timer.process();
      process_frame();
    }
  }

  static loco_t* get_loco(fan::window_t* window) {
    return OFFSETLESS(window, loco_t, window);
  }

protected:

  #ifdef loco_window
    fan::window_t window;
  #else
    fan::window_t* window;
  #endif

  #ifdef loco_context
    fan::graphics::context_t context;
  #else
    fan::graphics::context_t* context;
  #endif
};