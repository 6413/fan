#define loco_opengl
#define loco_context
#define loco_window

#define build_loco0

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(time/timer.h)
#include _FAN_PATH(font.h)
#include _FAN_PATH(physics/collision/circle.h)
#include _FAN_PATH(io/directory.h)

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

struct loco_t {

  using cid_t = fan::graphics::cid_t;

  loco_t() :window(fan::vec2(800, 800)), context(get_window()){
    gloco = (loco_t*)this;
  }

  struct camera_t;

  #define get_key_value(type) \
    *p.key.get_value<decltype(p.key)::get_index_with_type<type>()>()

  #define BLL_set_declare_NodeReference 1
  #define BLL_set_declare_rest 0
  #if defined(loco_opengl)
  #include _FAN_PATH(graphics/opengl/camera_list_builder_settings.h)
  #elif defined(loco_vulkan)
  #include _FAN_PATH(graphics/vulkan/camera_list_builder_settings.h)
  #endif
  #include _FAN_PATH(BLL/BLL.h)

  #ifdef loco_window
  fan::window_t window;
  #endif

  #ifdef loco_window
  fan::window_t* get_window() {
    return &window;
  }
  #endif

  #ifdef loco_context
  fan::graphics::context_t context;
  #endif

  #ifdef loco_context
  fan::graphics::context_t* get_context() {
    return &context;
  }
  #endif

  #if defined(loco_window)
  f32_t get_delta_time() {
    return get_window()->get_delta_time();
  }
  #endif

  #if defined(loco_window)
  void process_block_properties_element(auto* shape, loco_t::camera_list_NodeReference_t camera_id) {
    #if defined(loco_opengl)
    shape->m_shader.set_camera(get_context(), camera_list[camera_id].camera_id, &m_write_queue);
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

  /*template <uint8_t n>
  void process_block_properties_element(auto* shape, textureid_t<n> tid) {
    #if defined(loco_opengl)
    if (tid.NRI == (decltype(tid.NRI))-1) {
      return;
    }
    shape->m_shader.set_int(get_context(), tid.name, n);
    get_context()->opengl.call(get_context()->opengl.glActiveTexture, fan::opengl::GL_TEXTURE0 + n);
    get_context()->opengl.call(get_context()->opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, image_list[tid].texture_id);
    #endif
  }*/

  void process_block_properties_element(auto* shape, uint16_t depth) {

  }

  #endif

  struct camera_t {
    static constexpr f32_t znearfar = 0xffff;

    void open(loco_t* loco) {
      m_view = fan::mat4(1);
      camera_position = 0;
      camera_reference = loco->camera_list.NewNode();
      loco->camera_list[camera_reference].camera_id = this;
    }
    void close(loco_t* loco) {
      loco->camera_list.Recycle(camera_reference);
    }

    void open_camera(loco_t* loco, loco_t::camera_t* camera, const fan::vec2& x, const fan::vec2& y) {
      camera->open(loco);
      camera->set_ortho(loco, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
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

    camera_list_NodeReference_t camera_reference;
  };

  void open_camera(camera_t* camera, const fan::vec2& x, const fan::vec2& y) {
    camera->open(this);
    camera->set_ortho(this, x, y);
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
  #include _FAN_PATH(graphics/opengl/camera_list_builder_settings.h)
  #elif defined(loco_vulkan)
  #include _FAN_PATH(graphics/vulkan/camera_list_builder_settings.h)
  #endif
  #include _FAN_PATH(BLL/BLL.h)

  camera_list_t camera_list;

  #include "loco_types.h"

  #define sb_shape_var_name rectangle
  #include _FAN_PATH(graphics/opengl/2D/objects/rectangle.h)

  fan::ev_timer_t ev_timer;
  loco_bdbt_t bdbt;

  #if defined(loco_context)
  fan::graphics::core::memory_write_queue_t m_write_queue;
  #endif

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

  struct lighting_t {
    static constexpr const char* ambient_name = "lighting_ambient";
    fan::vec3 ambient = fan::vec3(1, 1, 1);
  }lighting;

  struct instance_t{
    instance_t() = default;
    loco_t::cid_t cid;
    bool initialized = false;
    instance_t(const auto& properties);

    void set_position(const fan::vec3& position);
  };

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

  fan::vec2 transform_position(const fan::vec2& p, const fan::graphics::viewport_t& viewport) {
    fan::vec2 x;
    x.x = (p.x - viewport.viewport_position.x - viewport.viewport_size.x / 2) / (viewport.viewport_size.x / 2);
    x.y = ((p.y - viewport.viewport_position.y - viewport.viewport_size.y / 2) / (viewport.viewport_size.y / 2) + (viewport.viewport_position.y / viewport.viewport_size.y) * 2);
    return x;
  }

  fan::vec2 get_mouse_position(const fan::graphics::viewport_t& viewport) {
    return transform_position(get_mouse_position(), viewport);
  }
};

#if defined(loco_window)

loco_t::camera_list_NodeReference_t::camera_list_NodeReference_t(loco_t::camera_t* camera) {
  NRI = camera->camera_reference.NRI;
}

//fan::opengl::theme_list_NodeReference_t::theme_list_NodeReference_t(auto* theme) {
//  static_assert(std::is_same_v<decltype(theme), loco_t::theme_t*>, "invalid parameter passed to theme");
//  NRI = theme->theme_reference.NRI;
//}

#endif

template <typename ...shapes_t>
struct engine_wrap : loco_t {

  std::tuple<shapes_t...> shapes;

  template<typename T>
  void push_shape(loco_t::cid_t* cid, const T& properties) {
    auto iterate_shape = [cid, this, &properties]<typename T2>(T2& shape) {
      if constexpr (std::is_same_v<T, T2::properties_t>) {
        std::get<T2>(shapes).push_back(cid, properties);
      }
    };
    std::apply([&](auto&&... args) { (iterate_shape(args), ...); }, shapes);
  }

  void set_position(loco_t::cid_t* cid, const fan::vec3& position) {
    /*auto iterate_shape = [cid, this, &position]<typename T2>(T2 & shape) {
      if constexpr (std::is_same_v<T, T2::position>) {
        std::get<T2>(shapes).set(cid, &T2::vi_t::position, position);
      }
    };
    std::apply([&](auto&&... args) { (iterate_shape(args), ...); }, shapes);*/
  }

  //void erase_shape(loco_t::cid_t* cid) {
  //  // switch cid
  //}

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

    m_write_queue.process(get_context());

    #ifdef loco_window
    #if defined(loco_opengl)

    //#include "draw_shapes.h"

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

    //get_context()->opengl.call(get_context()->opengl.glDrawBuffers, std::size(attachments), attachments);

    renderQuad();
    #endif

    std::apply([](auto&&... args) { (args.draw(), ...); }, shapes);

    for (auto it = m_draw_queue.begin(); it != m_draw_queue.end(); ++it) {
      for (const auto& f : it->f) {
        f();
      }
    }

    m_draw_queue.clear();
    get_context()->render(get_window());
    #endif
    #endif
  }

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
};

using rectangle_t = loco_t::rectangle_t::properties_t;