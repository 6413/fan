#ifndef loco_vulkan
#define loco_opengl
#endif

#include <set>

#include _FAN_PATH(types/types.h)

#include _FAN_PATH(types/color.h)


struct loco_t;

#define loco_framebuffer

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(time/timer.h)
#include _FAN_PATH(font.h)
#include _FAN_PATH(physics/collision/circle.h)
#include _FAN_PATH(io/directory.h)

// automatically gets necessary macros for shapes

#if defined(loco_text)
  #define loco_letter
#endif
#if defined(loco_sprite_sheet)
  #define loco_sprite
#endif
#if defined(loco_sprite)
  #define loco_texture_pack
#endif

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
  #error
#endif

#if defined(loco_menu_maker_button)
  #define loco_rectangle
  #define loco_letter
  #define loco_text
  #define loco_button
#endif

#if defined(loco_menu_maker_text_box)
  #define loco_rectangle
  #define loco_letter
  #define loco_text
  #define loco_button
  #define loco_text_box
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
  #ifdef loco_yuv420p
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
  #if defined loco_compute_shader
    #define loco_vulkan_descriptor_ssbo
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

#if defined(loco_window)
#define loco_vfi
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

#include "loco_types.h"

struct loco_t {

  #define get_key_value(type) \
    *p.key.get_value<decltype(p.key)::get_index_with_type<type>()>()

  struct shape_type_t {
    using _t = uint16_t;
    static constexpr _t button = 0;
    static constexpr _t sprite = 1;
    static constexpr _t text = 2;
    static constexpr _t hitbox = 3;
    static constexpr _t line = 4;
  };

  struct draw_t {
    uint64_t id;
    std::vector<fan::function_t<void()>> f;
    bool operator<(const draw_t& b) const
    {
      return id < b.id;
    }
  };

  // maybe can be set
  std::multiset<draw_t> m_draw_queue;
protected:


  #ifdef loco_window
  fan::window_t window;
  #endif

  #ifdef loco_context
  fan::graphics::context_t context;
  #else
  fan::graphics::context_t* context;
  #endif

#if defined(loco_opengl)

  unsigned int quadVAO = 0;
  unsigned int quadVBO;
  void renderQuad()
  {
    if (quadVAO == 0)
    {
      float quadVertices[] = {
        // positions        // texture Coords
        -1.0f,  1.0f, 0, 0.0f, 1.0f,
        -1.0f, -1.0f, 0, 0.0f, 0.0f,
         1.0f,  1.0f, 0, 1.0f, 1.0f,
         1.0f, -1.0f, 0, 1.0f, 0.0f,
      };
      // setup plane VAO
      get_context()->opengl.glGenVertexArrays(1, &quadVAO);
      get_context()->opengl.glGenBuffers(1, &quadVBO);
      get_context()->opengl.glBindVertexArray(quadVAO);
      get_context()->opengl.glBindBuffer(fan::opengl::GL_ARRAY_BUFFER, quadVBO);
      get_context()->opengl.glBufferData(fan::opengl::GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, fan::opengl::GL_STATIC_DRAW);
      get_context()->opengl.glEnableVertexAttribArray(0);
      get_context()->opengl.glVertexAttribPointer(0, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, 5 * sizeof(float), (void*)0);
      get_context()->opengl.glEnableVertexAttribArray(1);
      get_context()->opengl.glVertexAttribPointer(1, 2, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    get_context()->opengl.glBindVertexArray(quadVAO);
    get_context()->opengl.glDrawArrays(fan::opengl::GL_TRIANGLE_STRIP, 0, 4);
    get_context()->opengl.glBindVertexArray(0);
  }

#endif

public:
  struct image_t;

  #if defined(loco_window)

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
    static constexpr f32_t znearfar = 0xffff;

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
      matrices->set_ortho(loco, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
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

    void set_ortho(loco_t* loco, const fan::vec2& x, const fan::vec2& y) {
      m_projection = fan::math::ortho<fan::mat4>(
        x.x,
        x.y,
        #if defined (loco_opengl)
        y.y,
        y.x,
        0.1,
        znearfar / 2
        #elif defined(loco_vulkan)
        // znear & zfar is actually flipped for vulkan (camera somehow flipped)
        // znear & zfar needs to be same maybe xd
        y.x,
        y.y,
        0.1,
        znearfar
        #endif


        );
      coordinates.left = x.x;
      coordinates.right = x.y;
      coordinates.down = y.y;
      coordinates.up = y.x;

      m_view[3][0] = 0;
      m_view[3][1] = 0;
      m_view[3][2] = 0;
      m_view = m_view.translate(camera_position);
      fan::vec3 position = m_view.get_translation();
      constexpr fan::vec3 front(0, 0, 1);

      m_view = fan::math::look_at_left<fan::mat4>(position, position + front, fan::camera::world_up);
      #if defined (loco_vulkan)
      #if defined(loco_line)
      {
        auto idx = loco->matrices_list[matrices_reference].matrices_index.line;
        if (idx != (uint8_t)-1) {
          loco->line.m_shader.set_matrices(loco, this, idx);
        }
      }
      #endif
      #if defined(loco_rectangle)
      {
        auto idx = loco->matrices_list[matrices_reference].matrices_index.rectangle;
        if (idx != (uint8_t)-1) {
          loco->rectangle.m_shader.set_matrices(loco, this, idx);
        }
      }
      #endif
      #if defined(loco_sprite)
      {
        auto idx = loco->matrices_list[matrices_reference].matrices_index.sprite;
        if (idx != (uint8_t)-1) {
          loco->sprite.m_shader.set_matrices(loco, this, idx);
        }
      }
      #endif
      #if defined(loco_letter)
      {
        auto idx = loco->matrices_list[matrices_reference].matrices_index.letter;
        if (idx != (uint8_t)-1) {
          loco->letter.m_shader.set_matrices(loco, this, idx);
        }
      }
      #endif
      #if defined(loco_button)
      {
        auto idx = loco->matrices_list[matrices_reference].matrices_index.button;
        if (idx != (uint8_t)-1) {
          loco->button.m_shader.set_matrices(loco, this, idx);
        }
      }
      #endif
      #if defined(loco_text_box)
      {
        auto idx = loco->matrices_list[matrices_reference].matrices_index.text_box;
        if (idx != (uint8_t)-1) {
          loco->text_box.m_shader.set_matrices(loco, this, idx);
        }
      }
      #endif
      #endif
    }

    fan::mat4 m_projection;
    // temporary
    fan::mat4 m_view;

    fan::vec3 camera_position;

    union {
      struct {
        f32_t left;
        f32_t right;
        f32_t up;
        f32_t down;
      };
      fan::vec4 v;
    }coordinates;

    matrices_list_NodeReference_t matrices_reference;
  };

  void open_matrices(matrices_t* matrices, const fan::vec2& x, const fan::vec2& y) {
    matrices->open(this);
    matrices->set_ortho(this, x, y);
  }

  void open_viewport(fan::graphics::viewport_t* viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    viewport->open(get_context());
    viewport->set(get_context(), viewport_position, viewport_size, get_window()->get_size());
  }

  void set_viewport(fan::graphics::viewport_t* viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    viewport->set(get_context(), viewport_position, viewport_size, get_window()->get_size());
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

  #if defined(loco_texture_pack)
    #include _FAN_PATH(graphics/opengl/texture_pack.h)
  #endif

  #if defined(loco_tp)
    #if defined(loco_opengl)
      #include _FAN_PATH(tp/tp.h)
    #endif
  #endif

  #endif

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
      pool_info.poolSizeCount = std::size(pool_sizes);
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
  struct mouse_button_data_t : vfi_t::mouse_button_data_t {
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

  #include _FAN_PATH(graphics/gui/themes.h)

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
    #ifndef loco_context
    fan::graphics::context_t* context;
    #endif
  };

  static constexpr uint32_t max_depths = 2;

  #ifdef loco_window
  fan::window_t* get_window() {
    return &window;
  }
  #endif

  fan::graphics::context_t* get_context() {
    #ifdef loco_context
    return &context;
    #else
    return context;
    #endif
  }

  #if defined(loco_window)
  f32_t get_delta_time() {
    return get_window()->get_delta_time();
  }
  #endif

  struct push_constants_t {
    uint32_t texture_id;
    uint32_t matrices_id;
  };

  #if defined(loco_window)
  void process_block_properties_element(auto* shape, loco_t::matrices_list_NodeReference_t matrices_id) {
    #if defined(loco_opengl)
    shape->m_shader.set_matrices(get_context(), matrices_list[matrices_id].matrices_id, &m_write_queue);
    #elif defined(loco_vulkan)
    auto& matrices = matrices_list[matrices_id];
    auto context = get_context();

    uint32_t idx;

    #if defined(loco_line)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, line_t>::value) {
      idx = matrices.matrices_index.line;
    }
    #endif
    #if defined(loco_rectangle)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, rectangle_t>::value) {
      idx = matrices.matrices_index.rectangle;
    }
    #endif
    #if defined(loco_sprite)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, sprite_t>::value) {
      idx = matrices.matrices_index.sprite;
    }
    #endif
    #if defined(loco_letter)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, letter_t>::value) {
      idx = matrices.matrices_index.letter;
    }
    #endif
    #if defined(loco_button)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, button_t>::value) {
      idx = matrices.matrices_index.button;
    }
    #endif

    #if defined(loco_text_box)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, text_box_t>::value) {
      idx = matrices.matrices_index.text_box;
    }
    #endif


    #if defined(loco_yuv420p)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, yuv420p_t>::value) {
      idx = matrices.matrices_index.yuv420p;
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
    if constexpr (std::is_same<std::remove_pointer<decltype(shape)>::type, sprite_t>::value) {
      idx = img.texture_index.sprite;
    }
    #endif
    #if defined(loco_letter)
    if constexpr (std::is_same<std::remove_pointer<decltype(shape)>::type, letter_t>::value) {
      idx = img.texture_index.letter;
    }
    #endif

    #if defined(loco_yuv420p)
    if constexpr (std::is_same<std::remove_pointer<decltype(shape)>::type, yuv420p_t>::value) {
      idx = img.texture_index.yuv420p;
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

  void process_block_properties_element(auto* shape, uint16_t depth) {

  }

  #endif

  loco_bdbt_t bdbt;

  fan::ev_timer_t ev_timer;

  fan::graphics::core::memory_write_queue_t m_write_queue;

  #if defined(loco_compute_shader)
    #include _FAN_PATH(graphics/vulkan/compute_shader.h)
  #endif

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
  #if defined(loco_circle)
    #define sb_shape_var_name circle
    #include _FAN_PATH(graphics/opengl/2D/objects/circle.h)
    circle_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_yuv420p)
    #define sb_shape_var_name yuv420p
    #define sb_sprite_name yuv420p_t
    #include _FAN_PATH(graphics/opengl/2D/objects/yuv420p.h)
    yuv420p_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_nv12)
    #define sb_shape_var_name nv12
    #define sb_sprite_name nv12_t
    #include _FAN_PATH(graphics/opengl/2D/objects/nv12.h)
    nv12_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_sprite)
    #define sb_shape_var_name sprite
    #define sb_sprite_name sprite_t
    #include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
    sprite_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_light)
    #define sb_shape_var_name light
    #include _FAN_PATH(graphics/opengl/2D/objects/light.h)
    light_t sb_shape_var_name;
    #undef sb_shape_var_name
  #endif
  #if defined(loco_sprite_sheet)
    #define sb_shape_var_name sprite_sheet;
    #define sb_sprite_sheet_name sprite_sheet_t
    #include _FAN_PATH(graphics/sprite_sheet.h)
    sb_sprite_sheet_name sb_shape_var_name;
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
  #if defined (loco_menu_maker_button)
    #define sb_menu_maker_shape button
    #define sb_menu_maker_var_name menu_maker_button
    #define sb_menu_maker_type_name menu_maker_button_base_t
    #define sb_menu_maker_name menu_maker_button_t
    #include "wrappers/menu_maker.h"
  #endif
  #if defined (loco_menu_maker_text_box)
    #define sb_menu_maker_shape text_box
    #define sb_menu_maker_var_name menu_maker_text_box
    #define sb_menu_maker_type_name menu_maker_text_box_base_t
    #define sb_menu_maker_name menu_maker_text_box_t
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

  static constexpr uint8_t pixel_data[] = {
    1, 0, 0, 1,
    1, 0, 0, 1
  };

  loco_t(const properties_t& p = properties_t()) :
    #ifdef loco_window
    window(fan::vec2(800, 800)),
    #endif
    #if defined(loco_context)
    context(
      #if defined(loco_window)
      get_window()
      #endif
    )
    #endif
    #if defined(loco_window)
    , unloaded_image(this, fan::webp::image_info_t{ (void*)pixel_data, 1 })
    #endif
  {
    #if defined(loco_window)
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
    #endif
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

  #if defined(loco_opengl)
    loco_t::image_t::load_properties_t lp;
    lp.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;
  #if defined(loco_framebuffer)
    m_framebuffer.open(get_context());
    m_framebuffer.bind(get_context());
  #endif
  #endif

  #if defined(loco_opengl)

  #if defined(loco_framebuffer)

	  fan::webp::image_info_t ii;
	  ii.data = nullptr;
    ii.size = get_window()->get_size();

    lp.internal_format = fan::opengl::GL_R8UI;
    lp.format = fan::opengl::GL_RED_INTEGER; // GL_RGB_INTEGER for vec3
    lp.filter = fan::opengl::GL_NEAREST;
    lp.type = fan::opengl::GL_UNSIGNED_BYTE;

    color_buffers[1].load(this, ii, lp);

    color_buffers[1].bind_texture(this);
    fan::opengl::core::framebuffer_t::bind_to_texture(
      get_context(),
      *color_buffers[1].get_texture(this),
      fan::opengl::GL_COLOR_ATTACHMENT1
    );
    lp.internal_format = fan::opengl::GL_RGBA;
    lp.format = fan::opengl::GL_RGBA;
    lp.type = fan::opengl::GL_FLOAT;
    lp.filter = fan::opengl::GL_LINEAR;

    color_buffers[0].load(this, ii, lp);

    color_buffers[0].bind_texture(this);
    fan::opengl::core::framebuffer_t::bind_to_texture(
      get_context(),
      *color_buffers[0].get_texture(this),
      fan::opengl::GL_COLOR_ATTACHMENT0
    );

    color_buffers[2].load(this, ii, lp);

    color_buffers[2].bind_texture(this);
    fan::opengl::core::framebuffer_t::bind_to_texture(
      get_context(),
      *color_buffers[2].get_texture(this),
      fan::opengl::GL_COLOR_ATTACHMENT2
    );

    get_window()->add_resize_callback([this](const auto& d) {
      loco_t::image_t::load_properties_t lp;
      lp.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;

      fan::webp::image_info_t ii;
	    ii.data = nullptr;
      ii.size = get_window()->get_size();

      lp.internal_format = fan::opengl::GL_R8UI;
      lp.format = fan::opengl::GL_RED_INTEGER; // GL_RGB_INTEGER for vec3
      lp.filter = fan::opengl::GL_NEAREST;
      lp.type = fan::opengl::GL_UNSIGNED_BYTE;

      color_buffers[1].reload_pixels(this, ii, lp);

      color_buffers[1].bind_texture(this);
      fan::opengl::core::framebuffer_t::bind_to_texture(
        get_context(),
        *color_buffers[1].get_texture(this),
        fan::opengl::GL_COLOR_ATTACHMENT1
      );
      lp.internal_format = fan::opengl::GL_RGBA;
      lp.format = fan::opengl::GL_RGBA;
      lp.type = fan::opengl::GL_FLOAT;
      lp.filter = fan::opengl::GL_LINEAR;

      color_buffers[0].reload_pixels(this, ii, lp);

      color_buffers[0].bind_texture(this);
      fan::opengl::core::framebuffer_t::bind_to_texture(
        get_context(),
        *color_buffers[0].get_texture(this),
        fan::opengl::GL_COLOR_ATTACHMENT0
      );

      color_buffers[2].reload_pixels(this, ii, lp);

      color_buffers[2].bind_texture(this);
      fan::opengl::core::framebuffer_t::bind_to_texture(
        get_context(),
        *color_buffers[2].get_texture(this),
        fan::opengl::GL_COLOR_ATTACHMENT2
      );

      fan::opengl::core::renderbuffer_t::properties_t rp;
      m_framebuffer.bind(get_context());
      rp.size = ii.size;
      rp.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
      m_rbo.set_storage(get_context(), rp);
    });

    fan::opengl::core::renderbuffer_t::properties_t rp;
    m_framebuffer.bind(get_context());
    rp.size = ii.size;
    rp.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
    m_rbo.open(get_context());
    m_rbo.set_storage(get_context(), rp);
    rp.internalformat = fan::opengl::GL_DEPTH_ATTACHMENT;
    m_rbo.bind_to_renderbuffer(get_context(), rp);

    unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

    for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
      attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
    }

    get_context()->opengl.call(get_context()->opengl.glDrawBuffers, std::size(attachments), attachments);
    // finally check if framebuffer is complete
    if (!m_framebuffer.ready(get_context())) {
      fan::throw_error("framebuffer not ready");
    }

    m_framebuffer.unbind(get_context());

    m_fbo_final_shader.open(get_context());
    m_fbo_final_shader.set_vertex(
      get_context(),
      #include _FAN_PATH(graphics/glsl/opengl/2D/effects/loco_fbo.vs)
    );
    m_fbo_final_shader.set_fragment(
      get_context(),
      #include _FAN_PATH(graphics/glsl/opengl/2D/effects/loco_fbo.fs)
    );
    m_fbo_final_shader.compile(get_context());
  #endif

  #endif

  #if defined(loco_vulkan) && defined(loco_window)
    fan::vulkan::pipeline_t::properties_t pipeline_p;

    auto context = get_context();

    render_fullscreen_shader.open(context, &m_write_queue);
    render_fullscreen_shader.set_vertex(
      context, 
      "graphics/glsl/vulkan/2D/objects/loco_fbo.vert", 
      #include _FAN_PATH(graphics/glsl/vulkan/2D/objects/loco_fbo.vert))
    );
    render_fullscreen_shader.set_fragment(
      context, 
      "graphics/glsl/vulkan/2D/objects/loco_fbo.frag", 
      #include _FAN_PATH(graphics/glsl/vulkan/2D/objects/loco_fbo.frag))
    );
    VkDescriptorSetLayout layouts[] = {
    #if defined(loco_line)
      line.m_ssbo.m_descriptor.m_layout,
    #endif
    #if defined(loco_rectangle)
      rectangle.m_ssbo.m_descriptor.m_layout,
    #endif
    #if defined(loco_sprite)
      sprite.m_ssbo.m_descriptor.m_layout,
    #endif
    #if defined(loco_letter)
      letter.m_ssbo.m_descriptor.m_layout,
    #endif
    #if defined(loco_button)
      button.m_ssbo.m_descriptor.m_layout,
    #endif
    #if defined(loco_text_box)
      text_box.m_ssbo.m_descriptor.m_layout,
    #endif
    #if defined(loco_yuv420p)
      yuv420p.m_ssbo.m_descriptor.m_layout,
    #endif
    };
    pipeline_p.descriptor_layout_count = 1;
    pipeline_p.descriptor_layout = layouts;
    pipeline_p.shader = &render_fullscreen_shader;
    pipeline_p.push_constants_size = sizeof(loco_t::push_constants_t);
    pipeline_p.subpass = 1;
    VkDescriptorImageInfo imageInfo{};

    VkPipelineColorBlendAttachmentState color_blend_attachment[1]{};
    color_blend_attachment[0].colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT
		;
    color_blend_attachment[0].blendEnable = VK_TRUE;
    color_blend_attachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment[0].colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment[0].alphaBlendOp = VK_BLEND_OP_ADD;
    pipeline_p.color_blend_attachment_count = std::size(color_blend_attachment);
    pipeline_p.color_blend_attachment = color_blend_attachment;
    pipeline_p.enable_depth_test = false;
    context->render_fullscreen_pl.open(context, pipeline_p);
  #endif

    default_texture.create_missing_texture(this);
  }

  #if defined(loco_vfi)
  void push_back_input_hitbox(vfi_t::shape_id_t* id, const vfi_t::properties_t& p) {
    vfi.push_back(id, p);
  }
  #endif
  /* uint32_t push_back_keyboard_event(uint32_t depth, const fan_2d::graphics::gui::ke_t::properties_t& p) {
     return element_depth[depth].keyboard_event.push_back(p);
   }*/

  #if defined(loco_vfi)
  void feed_mouse_move(const fan::vec2& mouse_position) {
    vfi.feed_mouse_move(mouse_position);
  }

  void feed_mouse_button(uint16_t button, fan::mouse_state button_state, const fan::vec2& mouse_position) {
    vfi.feed_mouse_button(button, button_state);
  }

  void feed_keyboard(uint16_t key, fan::keyboard_state keyboard_state) {
    vfi.feed_keyboard(key, keyboard_state);
  }

  void feed_text(uint32_t key) {
    vfi.feed_text(key);
  }
  #endif

  void process_frame() {

    #if defined(loco_opengl)
    #if defined(loco_framebuffer)
    get_context()->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
    color_buffers[0].bind_texture(this);

    get_context()->opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
    color_buffers[1].bind_texture(this);

    get_context()->opengl.glActiveTexture(fan::opengl::GL_TEXTURE2);
    color_buffers[2].bind_texture(this);


    #endif
    #endif

    #if defined(loco_opengl)
    #if defined(loco_framebuffer)
    m_framebuffer.bind(get_context());
    //float clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    //auto buffers = fan::opengl::GL_COLOR_ATTACHMENT0 + 2;
    //get_context()->opengl.glClearBufferfv(fan::opengl::GL_COLOR, 0, clearColor);
    //get_context()->opengl.glClearBufferfv(fan::opengl::GL_COLOR, 1, clearColor);
    //get_context()->opengl.glClearBufferfv(fan::opengl::GL_COLOR, 2, clearColor);
    get_context()->opengl.glDrawBuffer(fan::opengl::GL_COLOR_ATTACHMENT2);
    get_context()->opengl.glClearColor(0, 0, 0, 1);
    get_context()->opengl.glClear(fan::opengl::GL_COLOR_BUFFER_BIT);
    get_context()->opengl.glDrawBuffer(fan::opengl::GL_COLOR_ATTACHMENT0);
    #endif
    get_context()->opengl.call(get_context()->opengl.glClearColor, 0, 0, 0, 1);
    get_context()->opengl.call(get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
    #endif

    #ifdef loco_post_process
    post_process.start_capture();
    #endif

    auto it = m_update_callback.GetNodeFirst();

    while (it != m_update_callback.dst) {
      m_update_callback[it](this);
      it = it.Next(&m_update_callback);
    }

    m_write_queue.process(get_context());

    #ifdef loco_window
      #if defined(loco_opengl)

      #include "draw_shapes.h"
    
    #if defined(loco_framebuffer)
      //m_flag_map_fbo.unbind(get_context());

      m_framebuffer.unbind(get_context());

      get_context()->opengl.call(get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
      //float clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
      fan::vec2 window_size = get_window()->get_size();
      fan::opengl::viewport_t::set_viewport(get_context(), 0, window_size, window_size);

      m_fbo_final_shader.use(get_context());
      m_fbo_final_shader.set_int(get_context(), "_t00", 0);
      m_fbo_final_shader.set_int(get_context(), "_t01", 1);
      m_fbo_final_shader.set_int(get_context(), "_t02", 2);

      get_context()->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
      color_buffers[0].bind_texture(this);
     
      get_context()->opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
	    color_buffers[1].bind_texture(this);

      get_context()->opengl.glActiveTexture(fan::opengl::GL_TEXTURE2);
      color_buffers[2].bind_texture(this);

      unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

      for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
        attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
      }

      get_context()->opengl.call(get_context()->opengl.glDrawBuffers, std::size(attachments), attachments);
      renderQuad();
      #endif
      get_context()->render(get_window());
      #elif defined(loco_vulkan)
        get_context()->begin_render(get_window());
        draw_queue();
        #include "draw_shapes.h"
      get_context()->end_render(get_window());

      #endif
    #endif
  }
  #if defined(loco_window)
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

#if defined(loco_vulkan)
  fan::vulkan::shader_t render_fullscreen_shader;
#endif

#if defined(loco_framebuffer)

#if defined(loco_opengl)
  fan::opengl::core::framebuffer_t m_flag_map_fbo;
  
  fan::opengl::core::framebuffer_t m_framebuffer;
  fan::opengl::core::renderbuffer_t m_rbo;
  loco_t::image_t color_buffers[3];
  fan::opengl::shader_t m_fbo_final_shader;

#elif defined(loco_vulkan)

#endif

#endif

  bool process_loop(const auto& lambda) {
    uint32_t window_event = get_window()->handle_events();
    if (window_event & fan::window_t::events::close) {
      get_window()->destroy_window();
      return 1;
    }

    lambda();

    ev_timer.process();
    process_frame();
    return 0;
  }

  void loop(const auto& lambda) {
    while (1) {
      if (process_loop(lambda)) {
        break;
      }
    }
  }

  static loco_t* get_loco(fan::window_t* window) {
    return OFFSETLESS(window, loco_t, window);
  }
  #endif

protected:
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix update_callback
  #define BLL_set_BaseLibrary 1
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType fan::function_t<void(loco_t*)>
  #include _FAN_PATH(BLL/BLL.h)
public:

  using update_callback_nr_t = update_callback_NodeReference_t;

  update_callback_t m_update_callback;

  image_t default_texture;

  struct lighting_t {
    static constexpr const char* ambient_name = "lighting_ambient";
    fan::vec3 ambient = fan::vec3(1, 1, 1);
  }lighting;

  #define make_key_value(type, name) \
      type& name = *key.get_value<decltype(key)::get_index_with_type<type>()>();

  #if defined (loco_no_inline)

  #define make_shape_id(name) \
    struct name ## _id_t { \
 \
 \
      struct properties_t { \
        loco_ ## name ## _vi_t \
        loco_ ## name ## _bm_properties_t \
        loco_ ## name ## _ri_t \
        loco_ ## name ## _properties_t \
      }; \
 \
 \
      fan::graphics::cid_t cid; \
 \
 \
      operator fan::graphics::cid_t* () { \
        return &cid; \
      } \
 \
      name ## _id_t() = default; \
      name ## _id_t(const properties_t&); \
      name ## _id_t& operator[](const properties_t&); \
      ~name ## _id_t(); \
    };

    #if defined (loco_rectangle)
      make_shape_id(rectangle);
    #endif

    #if defined(loco_sprite)
      struct sprite_id_t {

        struct properties_t {
          loco_sprite_vi_t
          loco_sprite_bm_properties_t
          loco_sprite_ri_t 
          loco_sprite_properties_t

          loco_t::texturepack_t::ti_t* ti = 0;
        };
        fan::graphics::cid_t cid;
        operator fan::graphics::cid_t* () { 
          return &cid;
        }

        sprite_id_t() = default;
        sprite_id_t(const properties_t& p);
        sprite_id_t& operator[](const properties_t&);
        ~sprite_id_t();
      };
    #endif

    #if defined(loco_letter)
      make_shape_id(letter);
    #endif

    #if defined(loco_text)
      make_shape_id(text);
    #endif

    #if defined(loco_button)
      make_shape_id(button);
    #endif

    #if defined(loco_text_box)
      make_shape_id(text_box);
    #endif
    
    #if defined(loco_light)
      make_shape_id(light);
    #endif

    struct vfi_id_t {
      using properties_t = loco_t::vfi_t::properties_t;
      operator loco_t::vfi_t::shape_id_t* () {
        return &cid;
      }
      vfi_id_t() = default;
      vfi_id_t(const properties_t&);
      vfi_id_t& operator[](const properties_t&);
      ~vfi_id_t();

      loco_t::vfi_t::shape_id_t cid;
    };

    #endif

};

#if defined(loco_window)
loco_t::image_list_NodeReference_t::image_list_NodeReference_t(loco_t::image_t* image) {
  NRI = image->texture_reference.NRI;
}

loco_t::matrices_list_NodeReference_t::matrices_list_NodeReference_t(loco_t::matrices_t* matrices) {
  NRI = matrices->matrices_reference.NRI;
}

fan::opengl::theme_list_NodeReference_t::theme_list_NodeReference_t(auto* theme) {
  static_assert(std::is_same_v<decltype(theme), loco_t::theme_t*>, "invalid parameter passed to theme");
  NRI = theme->theme_reference.NRI;
}

#endif

#ifndef loco_no_inline
  #undef loco_rectangle_vi_t

  #undef loco_rectangle
  #undef loco_sprite
  #undef loco_letter
  #undef loco_text
  #undef loco_text_box
  #undef loco_button
  #undef loco_wboit
#endif