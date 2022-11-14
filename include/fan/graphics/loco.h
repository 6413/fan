#ifndef loco_vulkan
  #define loco_opengl
#endif

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(time/timer.h)
#include _FAN_PATH(font.h)

struct loco_t;

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
    #define loco_rectangle
    #define loco_letter
    #define loco_text
    #define loco_text_box
    #define loco_button
  #endif

#if defined(loco_wboit)
  #define loco_vulkan_descriptor_image_sampler
#endif

#ifdef loco_vulkan
  #ifdef loco_line 
    #ifndef loco_vulkan_descriptor_ssbo
      #define loco_vulkan_descriptor_ssbo
    #endif
    #ifndef loco_vulkan_descriptor_uniform_block
      #define loco_vulkan_descriptor_uniform_block
    #endif
  #endif
  #ifdef loco_rectangle 
    #ifndef loco_vulkan_descriptor_ssbo
      #define loco_vulkan_descriptor_ssbo
    #endif
    #ifndef loco_vulkan_descriptor_uniform_block
      #define loco_vulkan_descriptor_uniform_block
    #endif
  #endif
  #ifdef loco_sprite
    #ifndef loco_vulkan_descriptor_ssbo
      #define loco_vulkan_descriptor_ssbo
    #endif
    #ifndef loco_vulkan_descriptor_uniform_block
      #define loco_vulkan_descriptor_uniform_block
    #endif
    #ifndef loco_vulkan_descriptor_image_sampler
      #define loco_vulkan_descriptor_image_sampler
    #endif
  #endif
 #ifdef loco_letter
    #ifndef loco_vulkan_descriptor_ssbo
      #define loco_vulkan_descriptor_ssbo
    #endif
    #ifndef loco_vulkan_descriptor_uniform_block
      #define loco_vulkan_descriptor_uniform_block
    #endif
    #ifndef loco_vulkan_descriptor_image_sampler
      #define loco_vulkan_descriptor_image_sampler
    #endif
  #endif
#endif

#define BDBT_set_prefix loco_bdbt
#define BDBT_set_type_node uint16_t
#define BDBT_set_BitPerNode 2
#define BDBT_set_declare_rest 1
#define BDBT_set_declare_Key 0
#define BDBT_set_BaseLibrary 1
#define BDBT_set_CPP_ConstructDestruct
#include _FAN_PATH(BDBT/BDBT.h)

#define BDBT_set_prefix loco_bdbt
#define BDBT_set_type_node uint16_t
#define BDBT_set_KeySize 0
#define BDBT_set_BitPerNode 2
#define BDBT_set_declare_rest 0
#define BDBT_set_declare_Key 1
#define BDBT_set_base_prefix loco_bdbt
#define BDBT_set_BaseLibrary 1
#define BDBT_set_CPP_ConstructDestruct
#include _FAN_PATH(BDBT/BDBT.h)

#define loco_vfi

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
public:
  struct image_t;

  #if defined(loco_opengl)
    #include _FAN_PATH(graphics/opengl/image_list_builder_settings.h)
  #elif defined(loco_vulkan)
    #include _FAN_PATH(graphics/vulkan/image_list_builder_settings.h)
  #endif
	#include _FAN_PATH(BLL/BLL.h)

  image_list_t image_list;

	template <uint8_t n_>
	struct textureid_t : image_list_NodeReference_t {
		static constexpr std::array<const char*, 32> texture_names = {
			"_t00", "_t01", "_t02", "_t03",
			"_t04", "_t05", "_t06", "_t07",
			"_t08", "_t09", "_t10", "_t11",
			"_t12", "_t13", "_t14", "_t15",
			"_t16", "_t17", "_t18", "_t19",
			"_t20", "_t21", "_t22", "_t23",
			"_t24", "_t25", "_t26", "_t27",
			"_t28", "_t29", "_t30", "_t31"
		};
		static constexpr uint8_t n = n_;
		static constexpr auto name = texture_names[n];

		textureid_t() = default;
		textureid_t(image_t* image) : image_list_NodeReference_t::image_list_NodeReference_t(image) {
		}
	};

  #if defined(loco_opengl)
    #include _FAN_PATH(graphics/opengl/gl_image.h)
  #elif defined(loco_vulkan)
    #include _FAN_PATH(graphics/vulkan/vk_image.h)
  #endif

  struct matrices_t;

  #define BLL_set_declare_NodeReference 1
  #define BLL_set_declare_rest 0
  #if defined(loco_opengl)
    #include _FAN_PATH(graphics/opengl/matrices_list_builder_settings.h)
  #elif defined(loco_vulkan)
    #include _FAN_PATH(graphics/vulkan/matrices_list_builder_settings.h)
  #endif
  #include _FAN_PATH(BLL/BLL.h)

  struct matrices_t {
    void open(loco_t* loco) {
      auto* context = loco->get_context();
      m_view = fan::mat4(1);
      camera_position = 0;
      matrices_reference = loco->matrices_list.NewNode();
      loco->matrices_list[matrices_reference].matrices_id = this;
    }
    void close(loco_t* loco) {
      loco->matrices_list.Recycle(matrices_reference);
    }

    void open_matrices(loco_t* loco, loco_t::matrices_t* matrices, const fan::vec2& x, const fan::vec2& y) {
      matrices->open(loco);
      matrices->set_ortho(fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
    }

    fan::vec3 get_camera_position() const {
      return camera_position;
    }
    void set_camera_position(const fan::vec3& cp) {
      camera_position = cp;

      m_view[3][0] = 0;
      m_view[3][1] = 0;
      m_view[3][2] = 0;
      m_view = m_view.translate(camera_position);
      fan::vec3 position = m_view.get_translation();
      constexpr fan::vec3 front(0, 0, 1);

      m_view = fan::math::look_at_left<fan::mat4>(position, position + front, fan::camera::world_up);
    }

    void set_ortho(const fan::vec2& x, const fan::vec2& y) {
      m_projection = fan::math::ortho<fan::mat4>(
        x.x,
        x.y,
      #if defined (loco_opengl)
        y.y,
        y.x,
        -1,
        0xffffff
      #elif defined(loco_vulkan)
        // znear & zfar is actually flipped for vulkan (camera somehow flipped)
        // znear & zfar needs to be same maybe xd
        y.x,
        y.y,
        -0xffffff,
        0xffffff
      #endif


      );
      coordinates.left = x.x;
      coordinates.right = x.y;
    #if defined (loco_opengl)
      coordinates.bottom = y.y;
      coordinates.top = y.x;
    #elif defined(loco_vulkan)
      coordinates.bottom = y.x;
      coordinates.top = y.y;
    #endif

      m_view[3][0] = 0;
      m_view[3][1] = 0;
      m_view[3][2] = 0;
      m_view = m_view.translate(camera_position);
      fan::vec3 position = m_view.get_translation();
      constexpr fan::vec3 front(0, 0, 1);

      m_view = fan::math::look_at_left<fan::mat4>(position, position + front, fan::camera::world_up);
    }

    fan::mat4 m_projection;
    // temporary
    fan::mat4 m_view;

    fan::vec3 camera_position;

    union {
      struct {
        f32_t left;
        f32_t right;
        f32_t top;
        f32_t bottom;
      };
      fan::vec4 v;
    }coordinates;

    matrices_list_NodeReference_t matrices_reference;
  };

  void open_matrices(matrices_t* matrices, const fan::vec2& x, const fan::vec2& y) {
    matrices->open(this);
    matrices->set_ortho(x, y);
  }

  #define BLL_set_declare_NodeReference 0
  #define BLL_set_declare_rest 1
  #if defined(loco_opengl)
    #include _FAN_PATH(graphics/opengl/matrices_list_builder_settings.h)
  #elif defined(loco_vulkan)
    #include _FAN_PATH(graphics/vulkan/matrices_list_builder_settings.h)
  #endif
  #include _FAN_PATH(BLL/BLL.h)

  matrices_list_t matrices_list;

  uint32_t matrices_index = 0;

  image_t unloaded_image;

#ifdef loco_vulkan
  struct descriptor_pool_t {

    fan::graphics::context_t* get_context() {
      return ((loco_t*)OFFSETLESS(this, loco_t, descriptor_pool))->get_context();
    }

    descriptor_pool_t() {
      uint32_t total = 0;
      VkDescriptorPoolSize pool_sizes[] = {
        #ifdef loco_vulkan_descriptor_ssbo
        {
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          fan::vulkan::MAX_FRAMES_IN_FLIGHT
        },
        #endif
        #ifdef loco_vulkan_descriptor_uniform_block
        {
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          fan::vulkan::MAX_FRAMES_IN_FLIGHT
        },
        #endif
        #ifdef loco_vulkan_descriptor_image_sampler
        {
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          fan::vulkan::MAX_FRAMES_IN_FLIGHT
        },
        #endif
      };

      VkDescriptorPoolCreateInfo pool_info{};
      pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      pool_info.poolSizeCount = std::size(pool_sizes) * 10;
      pool_info.pPoolSizes = pool_sizes;
      pool_info.maxSets = fan::vulkan::MAX_FRAMES_IN_FLIGHT * 10;

      fan::vulkan::validate(vkCreateDescriptorPool(get_context()->device, &pool_info, nullptr, &m_descriptor_pool));
    }
    ~descriptor_pool_t() {
      vkDestroyDescriptorPool(get_context()->device, m_descriptor_pool, nullptr);
    }

    VkDescriptorPool m_descriptor_pool;
  }descriptor_pool;
#endif

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

  f32_t get_delta_time() {
    return get_window()->get_delta_time();
  }

  struct push_constants_t {
    uint32_t texture_id;
    uint32_t matrices_id;
  };

  void process_block_properties_element(auto* shape, loco_t::matrices_list_NodeReference_t matrices_id) {
    #if defined(loco_opengl)
      shape->m_shader.set_matrices(get_context(), matrices_list[matrices_id].matrices_id, &m_write_queue);
    #elif defined(loco_vulkan)
      auto& matrices = matrices_list[matrices_id];
      auto context = get_context();

      uint32_t idx;


      #if defined(loco_line)
        if constexpr(std::is_same<std::remove_pointer<decltype(shape)>::type, line_t>::value) {
          idx = matrices.matrices_index.line;
        }
      #endif
      #if defined(loco_rectangle)
        if constexpr(std::is_same<std::remove_pointer<decltype(shape)>::type, rectangle_t>::value) {
          idx = matrices.matrices_index.rectangle;
        }
      #endif
      #if defined(loco_sprite)
        if constexpr(std::is_same<std::remove_pointer<decltype(shape)>::type, sprite_t>::value) {
          idx = matrices.matrices_index.sprite;
        }
      #endif
      #if defined(loco_letter)
        if constexpr(std::is_same<std::remove_pointer<decltype(shape)>::type, letter_t>::value) {
          idx = matrices.matrices_index.letter;
        }
      #endif
      #if defined(loco_button)
        if constexpr(std::is_same<std::remove_pointer<decltype(shape)>::type, button_t>::value) {
          idx = matrices.matrices_index.button;
        }
      #endif

      vkCmdPushConstants(
        context->commandBuffers[context->currentFrame], 
        shape->m_pipeline.m_layout, 
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 
        offsetof(push_constants_t, matrices_id), 
        sizeof(uint32_t), 
        &idx
      );
    #endif
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

  template <uint8_t n>
  void process_block_properties_element(auto* shape, textureid_t<n> tid) {
    #if defined(loco_opengl)
      shape->m_shader.set_int(get_context(), tid.name, n);
      get_context()->opengl.call(get_context()->opengl.glActiveTexture, fan::opengl::GL_TEXTURE0 + n);
      get_context()->opengl.call(get_context()->opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, image_list[tid].texture_id);
    #elif defined(loco_vulkan)
      auto& img = image_list[tid];
      auto context = get_context();

      uint32_t idx;

      #if defined(loco_sprite)
        if constexpr(std::is_same<std::remove_pointer<decltype(shape)>::type, sprite_t>::value) {
          idx = img.texture_index.sprite;
        }
      #endif
      #if defined(loco_letter)
        if constexpr(std::is_same<std::remove_pointer<decltype(shape)>::type, letter_t>::value) {
          idx = img.texture_index.letter;
        }
      #endif

      vkCmdPushConstants(
        context->commandBuffers[context->currentFrame], 
        shape->m_pipeline.m_layout, 
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 
        offsetof(push_constants_t, texture_id), 
        sizeof(uint32_t), 
        &idx
      );
    #endif
  }

  loco_bdbt_t bdbt;

  fan::ev_timer_t ev_timer;

  fan::graphics::core::memory_write_queue_t m_write_queue;
  
  #if defined(loco_line)
    #define sb_shape_var_name line
    #include _FAN_PATH(graphics/opengl/2D/objects/line.h)
    line_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_rectangle)
    #define vk_shape_wboit
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
    #include _FAN_PATH(graphics/opengl/2D/objects/letter.h)
    letter_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_text)
    #define sb_shape_var_name text
    #include _FAN_PATH(graphics/opengl/2D/objects/text.h)
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
    #include _FAN_PATH(graphics/font.h)
    font_t font;
  #endif

  enum class shape_type_e{
    #if defined(loco_rectangle_text_button)
      rectangle_text_button
    #endif
  };

  static constexpr uint8_t pixel_data[] = {
    1, 0, 0, 1,
    1, 0, 0, 1
  };

  loco_t(const properties_t& p = properties_t()) :
    #ifdef loco_window
      window(fan::vec2(800, 800)),
    #else
      window(p.window),
    #endif
    context(get_window()),
    unloaded_image(this, fan::webp::image_info_t{(void*)pixel_data, 1})
  {
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

    #if defined(loco_opengl)
      fan::print("RENDERER BACKEND: OPENGL");
    #elif defined(loco_vulkan)
      fan::print("RENDERER BACKEND: VULKAN");
    #endif

    #if defined(loco_letter)
      font.open(this, loco_font);
    #endif

    #if defined(loco_post_process)
      fan::opengl::core::renderbuffer_t::properties_t rp;
      rp.size = get_window()->get_size();
      if (post_process.open(rp)) {
        fan::throw_error("failed to initialize frame buffer");
      }
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
        get_context()->begin_render(get_window());
        draw_queue();
        #include "draw_shapes.h"
        get_context()->end_render(get_window());

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

  fan::vec2 get_mouse_position(const fan::graphics::viewport_t& viewport) {
    fan::vec2 x;
    x.x = (get_mouse_position().x - viewport.viewport_position.x - viewport.viewport_size.x / 2) / (viewport.viewport_size.x / 2);
    x.y = ((get_mouse_position().y - viewport.viewport_position.y - viewport.viewport_size.y / 2) / (viewport.viewport_size.y / 2) + (viewport.viewport_position.y / viewport.viewport_size.y) * 2);
    return x;
  }

  void loop(const auto& lambda) {
    while (1) {
      uint32_t window_event = get_window()->handle_events();
      if(window_event & fan::window_t::events::close){
        get_window()->destroy_window();
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

  fan::function_t<void()> draw_queue = []{};
};

loco_t::image_list_NodeReference_t::image_list_NodeReference_t(loco_t::image_t* image) {
  NRI = image->texture_reference.NRI;
}

loco_t::matrices_list_NodeReference_t::matrices_list_NodeReference_t(loco_t::matrices_t* matrices) {
  NRI = matrices->matrices_reference.NRI;
}

#if defined(loco_sprite)
  #if defined(loco_opengl)
    #include _FAN_PATH(graphics/opengl/texture_pack.h)
  #endif
#endif

#undef loco_rectangle
#undef loco_letter
#undef loco_text
#undef loco_text_box
#undef loco_button
#undef loco_wboit