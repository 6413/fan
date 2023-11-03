#ifndef loco_vulkan
#define loco_opengl
#endif

#include <set>

#define loco_no_inline

struct loco_t;

#ifdef loco_no_inline
// doesnt support different kind of builds of loco
inline struct global_loco_t {

  loco_t* loco = nullptr;

  operator loco_t* () {
    return loco;
  }
  global_loco_t& operator=(loco_t* l) {
    loco = l;
    return *this;
  }
  loco_t* operator->() {
    return loco;
  }
}gloco;
#endif


struct loco_t;

#define loco_framebuffer

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(time/timer.h)
#include _FAN_PATH(font.h)
#include _FAN_PATH(physics/collision/circle.h)
#include _FAN_PATH(io/directory.h)
#include _FAN_PATH(event/event.h)

#include _FAN_PATH(trees/quad_tree.h)
#include _FAN_PATH(graphics/divider.h)

#include <variant>


#if defined(loco_imgui) && defined(fan_platform_linux)
static void imgui_xorg_init();
static void imgui_xorg_new_frame();
#endif

#if defined(loco_cuda)
#include "cuda_runtime.h"
#include <cuda.h>
#include <nvcuvid.h>

namespace fan {
  namespace cuda {
    void check_error(auto result) {
      if (result != CUDA_SUCCESS) {
        if constexpr (std::is_same_v<decltype(result), CUresult>) {
          const char* err_str = nullptr;
          cuGetErrorString(result, &err_str);
          fan::throw_error("function failed with:" + std::to_string(result) + ", " + err_str);
        }
        else {
          fan::throw_error("function failed with:" + std::to_string(result) + ", ");
        }
      }
    }
  }
}

extern "C" {
  extern __host__ cudaError_t CUDARTAPI cudaGraphicsGLRegisterImage(struct cudaGraphicsResource** resource, fan::opengl::GLuint image, fan::opengl::GLenum target, unsigned int flags);
}

#endif

// automatically gets necessary macros for shapes

#if defined(loco_sprite_sheet)
#define loco_sprite
#endif
#if defined(loco_sprite)
#define loco_texture_pack
#define loco_unlit_sprite
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

#if defined(loco_dropdown)
#define loco_rectangle
#define loco_letter
#define loco_text
#define loco_button
#define loco_text_box
#endif

#if defined(loco_text)
#define loco_letter
#define loco_responsive_text
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

// increase this to increase shape limit
using bdbt_key_type_t = uint16_t;

#define BDBT_set_prefix loco_bdbt
#define BDBT_set_type_node bdbt_key_type_t
#define BDBT_set_BitPerNode 2
#define BDBT_set_declare_rest 1
#define BDBT_set_declare_Key 0
#define BDBT_set_BaseLibrary 1
#define BDBT_set_CPP_ConstructDestruct
#include _FAN_PATH(BDBT/BDBT.h)

#define BDBT_set_prefix loco_bdbt
#define BDBT_set_type_node bdbt_key_type_t
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

#if defined(loco_pixel_format_renderer)

namespace fan {
  struct pixel_format {
    enum {
      undefined,
      yuv420p,
      nv12,
    };

    constexpr static uint8_t get_texture_amount(uint8_t format) {
      switch (format) {
        case undefined: {
          return 0;
        }
        case yuv420p: {
          return 3;
        }
        case nv12: {
          return 2;
        }
        default: {
          fan::throw_error("invalid format");
          return undefined;
        }
      }
    }
    constexpr static std::array<fan::vec2ui, 4> get_image_sizes(uint8_t format, const fan::vec2ui& image_size) {
      switch (format) {
        case yuv420p: {
          return std::array<fan::vec2ui, 4>{image_size, image_size / 2, image_size / 2};
        }
        case nv12: {
          return std::array<fan::vec2ui, 4>{image_size, fan::vec2ui{ image_size.x, image_size.y }};
        }
        default: {
          fan::throw_error("invalid format");
          return std::array<fan::vec2ui, 4>{};
        }
      }
    }
    template <typename T>
    constexpr static std::array<T, 4> get_image_properties(uint8_t format);
  };
}
#endif

namespace fan {
  namespace graphics {
  #if defined(loco_physics)
    static void open_bcol();
  #endif
    using direction_e = fan::graphics::viewport_divider_t::direction_e;
    inline fan::vec2 default_camera_ortho_x{-1, 1};
    inline fan::vec2 default_camera_ortho_y{-1, 1};
  }
}

struct loco_t {

  struct position2_t : public  fan::vec2 {
    //using fan::vec2::vec2;
  };

  struct position3_t : public fan::vec3 {
    using fan::vec3::vec3;
    using fan::vec3::operator=;
    position3_t& operator=(const position3_t& p) {
      fan::vec3::operator=(p);
      return *this;
    }
    //using fan::vec3::vec3;
    //position3_t(const fan::vec3& v) : fan::vec3(v) {}
    //position3_t& operator=(const fan::vec2& v) = default;
    /*
    {
      *(fan::vec3*)this = fan::vec3::operator=(v);
      return *this;
    }
    */
    /*position3_t& operator=(const fan::vec3& v) {
      *(fan::vec3*)this = fan::vec3::operator=(v);
      return *this;
    }*/
    // private:
      // using fan::vec3::operator=;
  };

  void use() {
    gloco = this;
  }

  std::vector<fan::function_t<void()>> m_draw_queue_light;

  using cid_t = fan::graphics::cid_t;

  #define get_key_value(type) \
    *p.key.get_value<decltype(p.key)::get_index_with_type<type>()>()

  struct shape_type_t {
    using _t = uint16_t;
    static constexpr _t invalid = -1;
    static constexpr _t button = 0;
    static constexpr _t sprite = 1;
    static constexpr _t text = 2;
    static constexpr _t hitbox = 3;
    static constexpr _t line = 4;
    static constexpr _t mark = 5;
    static constexpr _t rectangle = 6;
    static constexpr _t light = 7;
    static constexpr _t unlit_sprite = 8;
    static constexpr _t letter = 9;
    static constexpr _t text_box = 10;
    static constexpr _t circle = 11;
    static constexpr _t pixel_format_renderer = 12;
    static constexpr _t responsive_text = 13;
  };

  static constexpr const char* shape_names[] = {
    "Button",
    "Sprite",
    "Text",
    "Hitbox",
    "Line",
    "Mark",
    "Rectangle",
    "Light",
    "Unlit Sprite",
    "Letter",
    "Text Box",
    "Circle",
    "Pixel Format Renderer",
    "Reponsive Text"
  };


#pragma pack(push, 1)
  template <typename T, bool order_matters = false>
  struct make_push_key_t {
    T data;
    using key_t = loco_bdbt_Key_t<sizeof(decltype(data)) * 8, order_matters>;
    key_t k;
  };
  #pragma pack(pop)

#pragma pack(push, 1)
  template <typename T, bool order_matters = false>
  struct make_erase_key_t {
    T data;
    using key_t = loco_bdbt_Key_t<sizeof(decltype(data)) * 8, order_matters>;
    typename key_t::KeySize_t key_size;
    loco_bdbt_NodeReference_t key_nr;
  };
#pragma pack(pop)

  struct redraw_key_t {
    uint8_t blending;
  };

  struct lighting_t {
    static constexpr const char* ambient_name = "lighting_ambient";
    fan::vec3 ambient = fan::vec3(1, 1, 1);

    //void set_ambient() {

    //  //m_current_shader->set_vec3(loco->get_context(), loco_t::lighting_t::ambient_name, loco->lighting.ambient);
    //}
  }lighting;

protected:

  struct gloco_priority_init_t {
    gloco_priority_init_t(loco_t* l) {
      gloco = l;
    }
  }gloco_dummy;

  #ifdef loco_window
  fan::window_t window;
  #endif

  #ifdef loco_context
  fan::graphics::context_t context;
  #endif

  #if defined(loco_opengl) && defined(loco_context)

public:
  using viewport_t = fan::graphics::viewport_t;
protected:

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

  struct camera_t;

  #define BLL_set_declare_NodeReference 1
  #define BLL_set_declare_rest 0
  #if defined(loco_opengl)
  #include _FAN_PATH(graphics/opengl/camera_list_builder_settings.h)
  #elif defined(loco_vulkan)
  #include _FAN_PATH(graphics/vulkan/camera_list_builder_settings.h)
  #endif
  #include _FAN_PATH(BLL/BLL.h)

  struct viewport_resize_cb_data_t {
    camera_t* camera;
    fan::vec3 position;
    fan::vec2 size;
  };
  using viewport_resize_cb_t = fan::function_t<void(const viewport_resize_cb_data_t&)>;

  #define BLL_set_Mark 2
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_prefix viewport_resize_callback
  #define BLL_set_NodeData viewport_resize_cb_t data;
  #define BLL_set_CPP_nrsic 1
  #include _FAN_PATH(window/cb_list_builder_settings.h)
  #include _FAN_PATH(BLL/BLL.h)

  viewport_resize_callback_t m_viewport_resize_callback;

  struct camera_t {

    using resize_cb_data_t = loco_t::viewport_resize_cb_data_t;
    using resize_cb_t = loco_t::viewport_resize_cb_t;
    struct resize_callback_id_t : loco_t::viewport_resize_callback_NodeReference_t {
      using inherit_t = viewport_resize_callback_NodeReference_t;
      resize_callback_id_t() : loco_t::viewport_resize_callback_NodeReference_t() {}
      resize_callback_id_t(const inherit_t& i) : inherit_t(i) {}
      resize_callback_id_t(resize_callback_id_t&& i) : inherit_t(i) {
        i.sic();
      }

      resize_callback_id_t& operator=(const resize_callback_id_t& i) = delete;

      resize_callback_id_t& operator=(resize_callback_id_t&& i) {
        if (this != &i) {
          *(inherit_t*)this = *(inherit_t*)&i;
          i.sic();
        }
        return *this;
      }

      operator loco_t::viewport_resize_callback_NodeReference_t() {
        return *(loco_t::viewport_resize_callback_NodeReference_t*)this;
      }
      ~resize_callback_id_t() {
        if (iic()) {
          return;
        }
        gloco->m_viewport_resize_callback.unlrec(*this);
        sic();
      }
    };

    resize_callback_id_t add_resize_callback(resize_cb_t function) {
      auto nr = gloco->m_viewport_resize_callback.NewNodeLast();
      gloco->m_viewport_resize_callback[nr].data = function;
      return resize_callback_id_t(nr);
    }

    //void remove_resize_callback(resize_callback_id_t id) {
    //  gloco->m_viewport_resize_callback.Unlink(id);
    //}

    camera_t() {
      camera_reference.sic();
    }

    static constexpr f32_t znearfar = 0xffff;

    void open() {
      auto* context = gloco->get_context();
      m_view = fan::mat4(1);
      camera_position = 0;
      camera_reference = gloco->camera_list.NewNode();
      gloco->camera_list[camera_reference].camera_id = this;
    }
    void close() {
      gloco->camera_list.Recycle(camera_reference);
    }

    void open_camera(loco_t::camera_t* camera, const fan::vec2& x, const fan::vec2& y) {
      camera->open();
      camera->set_ortho(fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
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

    fan::vec2 get_camera_size() const {
      return fan::vec2(std::abs(coordinates.right - coordinates.left), std::abs(coordinates.down - coordinates.up));
    }

    bool calculate_aspect_ratio = false;

    fan::vec2 some_function(fan::vec2 d, fan::vec2 c) {
      return c / (c / d).min();
    }

    void set_ortho(fan::vec2 x, fan::vec2 y, loco_t::viewport_t* aspect_ratio_viewport = nullptr) {

      if (aspect_ratio_viewport) {
        //fan::vec2 ratio = aspect_ratio_viewport->get_size().square_normalize();
        //fan::vec2 desired_ratio(1, 1);
        //fan::print(ratio);
        //// Calculate the orthographic projection matrix
        //fan::vec2 ortho = desired_ratio * ratio;
        //ortho *= fan::vec2(9.f / 16.f, 1) / ortho.min();

        //fan::vec2 ws = aspect_ratio_viewport->get_size();
        //f32_t windowWidth = ws.x;
        //f32_t windowHeight = ws.y;

        //float aspectRatio = windowWidth / windowHeight;

        //// Define the desired aspect ratio (16:9)
        //float desiredAspectRatio = 16.0f / 9.0f;

        //// Calculate the scaling factors
        //float xScale = 1.0f;
        //float yScale = 1.0f;

        //if (aspectRatio > desiredAspectRatio) {
        //  // The window is wider than 16:9
        //  xScale = aspectRatio / desiredAspectRatio;
        //}
        //else {
        //  // The window is taller than 16:9
        //  yScale = desiredAspectRatio / aspectRatio;
        //}


        //x = { -8.0f * xScale, 8.0f * xScale };
        //y = { -4.5f * yScale, 4.5f * yScale };

        //x /= 4.5f;
        //y /= 4.5f;

        fan::vec2 desired_res = { 1, 1 };
        fan::vec2 current_res = aspect_ratio_viewport->get_size();

        auto ortho = some_function(desired_res, current_res);
        //fan::print(ortho);

        x = { -ortho.x, +ortho.x };
        y = { -ortho.y, +ortho.y };

        calculate_aspect_ratio = true;
      }

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
        auto idx = loco->camera_list[camera_reference].camera_index.line;
        if (idx != (uint8_t)-1) {
          loco->line.m_shader.set_camera(loco, this, idx);
        }
      }
      #endif
      #if defined(loco_rectangle)
      {
        auto idx = loco->camera_list[camera_reference].camera_index.rectangle;
        if (idx != (uint8_t)-1) {
          loco->rectangle.m_shader.set_camera(loco, this, idx);
        }
      }
      #endif
      #if defined(loco_sprite)
      {
        auto idx = loco->camera_list[camera_reference].camera_index.sprite;
        if (idx != (uint8_t)-1) {
          loco->sprite.m_shader.set_camera(loco, this, idx);
        }
      }
      #endif
      #if defined(loco_letter)
      {
        auto idx = loco->camera_list[camera_reference].camera_index.letter;
        if (idx != (uint8_t)-1) {
          loco->letter.m_shader.set_camera(loco, this, idx);
        }
      }
      #endif
      #if defined(loco_button)
      {
        auto idx = loco->camera_list[camera_reference].camera_index.button;
        if (idx != (uint8_t)-1) {
          loco->button.m_shader.set_camera(loco, this, idx);
        }
      }
      #endif
      #if defined(loco_text_box)
      {
        auto idx = loco->camera_list[camera_reference].camera_index.text_box;
        if (idx != (uint8_t)-1) {
          loco->text_box.m_shader.set_camera(loco, this, idx);
        }
      }
      #endif
      #endif

      auto it = gloco->m_viewport_resize_callback.GetNodeFirst();

      while (it != gloco->m_viewport_resize_callback.dst) {

        gloco->m_viewport_resize_callback.StartSafeNext(it);

        resize_cb_data_t cbd;
        cbd.camera = this;
        cbd.position = get_camera_position();
        cbd.size = get_camera_size();
        gloco->m_viewport_resize_callback[it].data(cbd);

        it = gloco->m_viewport_resize_callback.EndSafeNext();
      }
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
    camera->open();
    camera->set_ortho(x, y);
  }

  void open_viewport(fan::graphics::viewport_t* viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    viewport->open();
    viewport->set(viewport_position, viewport_size, get_window()->get_size());
  }

  void set_viewport(fan::graphics::viewport_t* viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    viewport->set(viewport_position, viewport_size, get_window()->get_size());
  }

  struct camera_impl_t {

    camera_impl_t() = default;
    camera_impl_t(fan::graphics::direction_e split_direction) {
      fan::graphics::viewport_divider_t::iterator_t it = gloco->viewport_divider.insert(split_direction);
      fan::vec2 p = it.parent->position;
      fan::vec2 s = it.parent->size;

      fan::vec2 window_size = gloco->get_window()->get_size();
      gloco->open_viewport(&viewport, (p - s / 2) * window_size, (s)*window_size);
      gloco->open_camera(&camera, fan::graphics::default_camera_ortho_x, fan::graphics::default_camera_ortho_y);
    }
    loco_t::camera_t camera;
    loco_t::viewport_t viewport;
  };

  #define BLL_set_declare_NodeReference 0
  #define BLL_set_declare_rest 1
  #if defined(loco_opengl)
  #include _FAN_PATH(graphics/opengl/camera_list_builder_settings.h)
  #elif defined(loco_vulkan)
  #include _FAN_PATH(graphics/vulkan/camera_list_builder_settings.h)
  #endif
  #include _FAN_PATH(BLL/BLL.h)

  camera_list_t camera_list;

  uint32_t camera_index = 0;

  image_t unloaded_image;

  #endif

  #if defined(loco_texture_pack)
  #include _FAN_PATH(graphics/opengl/texture_pack.h)
  #endif

  #if defined(loco_tp)
  #if defined(loco_opengl)
  #include _FAN_PATH(tp/tp0.h)
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

  #if defined(loco_no_inline)
protected:
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix cid_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeData fan::graphics::cid_t cid;
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  struct cid_nt_t : cid_list_NodeReference_t {
    loco_t::cid_t* operator->() {
      return &gloco->cid_list[*(cid_list_NodeReference_t*)(this)].cid;
    }
    using base_t = cid_list_NodeReference_t;
    void init() {
      *(base_t*)this = gloco->cid_list.NewNodeLast();
    }

    bool is_invalid() const {
      return cid_list_inric(*this);
    }

    void invalidate_soft() {
      *(base_t*)this = gloco->cid_list.gnric();
    }

    void invalidate() {
      if (is_invalid()) {
        return;
      }
      gloco->cid_list.unlrec(*this);
      *(base_t*)this = gloco->cid_list.gnric();
    }

    uint32_t* gdp4() {
      return (uint32_t*)&(*this)->bm_id;
    }
  };

  struct cid_nr_t : cid_nt_t {

    cid_nr_t() { *(cid_list_NodeReference_t*)this = cid_list_gnric(); }
    cid_nr_t(const cid_nt_t& c) : cid_nt_t(c) {

    }

    cid_nr_t(const cid_nr_t& nr) : cid_nr_t() {
      if (nr.is_invalid()) {
        return;
      }
      init();
      gloco->cid_list[*this].cid.shape_type = gloco->cid_list[nr].cid.shape_type;
    }

    cid_nr_t(cid_nr_t&& nr) {
      NRI = nr.NRI;
      nr.invalidate_soft();
    }

    loco_t::cid_nr_t& operator=(const cid_nr_t& id) {
      if (!is_invalid()) {
        invalidate();
      }
      if (id.is_invalid()) {
        return *this;
      }

      if (this != &id) {
        init();
        gloco->cid_list[*this].cid.shape_type = gloco->cid_list[id].cid.shape_type;
      }
      return *this;
    }

    loco_t::cid_nr_t& operator=(cid_nr_t&& id) {
      if (!is_invalid()) {
        invalidate();
      }
      if (id.is_invalid()) {
        return *this;
      }

      if (this != &id) {
        if (!is_invalid()) {
          invalidate();
        }
        NRI = id.NRI;

        id.invalidate_soft();
      }
      return *this;
    }
  };

  #endif

  #if defined(loco_vfi)

  #define vfi_var_name vfi
  #include _FAN_PATH(graphics/gui/vfi.h)

  struct mouse_move_data_t : vfi_t::mouse_move_data_t {
    mouse_move_data_t(const vfi_t::mouse_move_data_t& mm) : vfi_t::mouse_move_data_t(mm) {

    }

    loco_t::cid_nt_t id;
  };
  struct mouse_button_data_t : vfi_t::mouse_button_data_t {
    mouse_button_data_t(const vfi_t::mouse_button_data_t& mm) : vfi_t::mouse_button_data_t(mm) {

    }

    loco_t::cid_nt_t id;
  };
  struct keyboard_data_t : vfi_t::keyboard_data_t {
    keyboard_data_t(const vfi_t::keyboard_data_t& mm) : vfi_t::keyboard_data_t(mm) {

    }

    loco_t::cid_nt_t id;
  };

  struct text_data_t : vfi_t::text_data_t {
    text_data_t(const vfi_t::text_data_t& mm) : vfi_t::text_data_t(mm) {

    }

    loco_t::cid_nt_t id;
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
    bool vsync = false;
  };

  static constexpr uint32_t max_depths = 2;

  #ifdef loco_window
  fan::window_t* get_window() {
    return &window;
  }
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

  struct push_constants_t {
    uint32_t texture_id;
    uint32_t camera_id;
  };

  #if defined(loco_window)
  void process_block_properties_element(auto* shape, loco_t::camera_list_NodeReference_t camera_id) {
    #if defined(loco_opengl)
    shape->m_current_shader->set_camera(get_context(), camera_list[camera_id].camera_id, &m_write_queue);
    #elif defined(loco_vulkan)
    auto& camera = camera_list[camera_id];
    auto context = get_context();

    uint32_t idx;

    #if defined(loco_line)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, line_t>::value) {
      idx = camera.camera_index.line;
    }
    #endif
    #if defined(loco_rectangle)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, rectangle_t>::value) {
      idx = camera.camera_index.rectangle;
    }
    #endif
    #if defined(loco_sprite)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, sprite_t>::value) {
      idx = camera.camera_index.sprite;
    }
    #endif
    #if defined(loco_letter)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, letter_t>::value) {
      idx = camera.camera_index.letter;
    }
    #endif
    #if defined(loco_button)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, button_t>::value) {
      idx = camera.camera_index.button;
    }
    #endif

    #if defined(loco_text_box)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, text_box_t>::value) {
      idx = camera.camera_index.text_box;
    }
    #endif


    #if defined(loco_yuv420p)
    if constexpr (std::is_same<typename std::remove_pointer<decltype(shape)>::type, yuv420p_t>::value) {
      idx = camera.camera_index.yuv420p;
    }
    #endif

    vkCmdPushConstants(
      context->commandBuffers[context->currentFrame],
      shape->m_pipeline.m_layout,
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      offsetof(push_constants_t, camera_id),
      sizeof(uint32_t),
      &idx
    );
    #endif
  }
  void process_block_properties_element(auto* shape, fan::graphics::viewport_list_NodeReference_t viewport_id) {
    auto data = &get_context()->viewport_list[viewport_id];
    data->viewport_id->set(
      data->viewport_id->get_position(),
      data->viewport_id->get_size(),
      get_window()->get_size()
    );
  }

  template <uint8_t n>
  void process_block_properties_element(auto* shape, textureid_t<n> tid) {
    #if defined(loco_opengl)
    if (tid.NRI == (decltype(tid.NRI))-1) {
      return;
    }
    shape->m_current_shader->use(get_context());
    shape->m_current_shader->set_int(get_context(), tid.name, n);
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
  loco_bdbt_NodeReference_t root;

  fan::ev_timer_t ev_timer;

  #if defined(loco_context)
  fan::graphics::core::memory_write_queue_t m_write_queue;
  #endif

  #if defined (loco_no_inline)

  cid_list_t cid_list;

  #define fan_create_id_definition_declare(rt, name, ...) rt name(__VA_ARGS__)
  #define fan_create_id_definition_define(rt, name, ...) rt name(__VA_ARGS__)

  #define fan_create_set_declare(rt, name) \
        fan_create_id_definition_declare(void, set_##name, const rt& data);

  #define fan_create_set_declare_custom(rt, name, custom) \
        fan_create_id_definition_declare(void, set_##name, const rt& data);

  #define fan_create_get_set_declare(rt, name) \
    fan_create_id_definition_declare(rt, get_##name); \
    fan_create_set_declare(rt, name)

  #define fan_create_get_set_declare_extra(rt, name, set_extra, get_extra) \
    fan_create_id_definition_declare(rt, get_##name); \
    fan_create_id_definition_declare(void, set_##name, const rt& data);

  #define fan_create_set_define(rt, name) \
        fan_create_id_definition_define(void, set_##name, const rt& data){ gloco->shape_##set_##name(*this, data); }

  #define fan_create_set_ptr_define(rt, name) \
        fan_create_id_definition_define(void, set_##name, rt data){ gloco->shape_##set_##name(*this, data); }

  #define fan_create_set_dataless_define(name) \
      fan_create_id_definition_define(void, set_##name){ gloco->shape_##set_##name(*this); }

  #define fan_create_set_define_custom(rt, name, custom) \
        fan_create_id_definition_define(void, set_##name, const rt& data){ custom }

  #define fan_create_get_set_define(rt, name) \
    fan_create_id_definition_define(rt, get_##name){ return gloco->shape_##get_##name(*this);} \
    fan_create_set_define(rt, name)

  #define fan_create_get_set_ptr_define(rt, name) \
    fan_create_id_definition_define(rt, get_##name){ return gloco->shape_##get_##name(*this);} \
    fan_create_set_ptr_define(rt, name)

  #define fan_create_get_define(rt, name) \
    fan_create_id_definition_define(rt, get_##name){ return gloco->shape_##get_##name(*this);} \

  #define fan_create_get_set_define_extra(rt, name, set_extra, get_extra) \
    fan_create_id_definition_define(rt, get_##name){ get_extra return gloco->shape_##get_##name(*this);} \
    fan_create_id_definition_define(void, set_##name, const rt& data){ set_extra gloco->shape_##set_##name(*this, data); }


  struct shape_t : cid_nr_t {
    using inherit_t = cid_nr_t;

    shape_t() { sic(); };
    template <typename T>
    shape_t(const T& properties) {
      inherit_t::init();
      gloco->push_shape(*this, properties);
    }

    inline shape_t(const shape_t& id) : 
      inherit_t(id)
    {
      if (id.is_invalid()) {
        return;
      }
      gloco->shape_get_properties(*(shape_t*)&id, [&]<typename T>(const T& properties) {
        gloco->push_shape(*this, properties);
      });
    }
    inline shape_t(shape_t&& id) : 
      inherit_t(std::move(id))
    {

    }
    
    loco_t::shape_t& operator=(const shape_t& id) {
      if (!is_invalid()) {
        erase();
      }
      if (id.is_invalid()) {
        return *this;
      }
      if (this != &id) {
        gloco->shape_get_properties(*(shape_t*)&id, [&]<typename T>(const T& properties) {
          init();
          gloco->push_shape(*this, properties);
        });
      }
      return *this;
    }
    

    loco_t::shape_t& operator=(shape_t&& id) {
      if (!is_invalid()) {
        erase();
      }
      if (id.is_invalid()) {
        return *this;
      }
      if (this != &id) {
        if (!is_invalid()) {
          erase();
        }
        *(inherit_t*)this = std::move(id);
        id.invalidate();
      }
      return *this;
    }

    ~shape_t() {
      erase();
    }

    void erase() {
      if (is_invalid()) {
        return;
      }
      gloco->shape_erase(*this);
      inherit_t::invalidate();
    }
    bool append_letter(wchar_t wc, bool force = false) {
      return gloco->shape_append_letter(*this, wc, force);
    }

    operator fan::opengl::cid_t* () {
      return &gloco->cid_list[*this].cid;
    }

    loco_t* get_loco() {
      return gloco;
    }

    fan_create_get_set_define_extra(fan::vec3, position,
      if (get_position().z != data.z) {
        gloco->shape_set_depth(*this, data.z);
      }
    , ;);
    fan_create_set_define_custom(fan::vec2, position,
      gloco->shape_set_position(*this, fan::vec3(data, get_position().z));
    );

    fan_create_get_set_define(fan::vec3, position_ar);
    fan_create_get_set_define(fan::vec2, size_ar);

    fan_create_get_set_ptr_define(loco_t::viewport_t*, viewport);
    fan_create_get_set_ptr_define(loco_t::camera_t*, camera);

    fan_create_get_set_define(fan::vec2, size);
    fan_create_get_set_define(fan::color, color);
    fan_create_get_set_define(f32_t, angle);
    fan_create_get_set_define(fan::string, text);
    fan_create_get_set_define(fan::vec2, rotation_point);
    fan_create_get_set_define(f32_t, font_size);

    fan_create_get_set_define(loco_t::textureid_t<0>, image);

    fan_create_get_set_define(fan::vec2, text_size);

    fan_create_get_set_define(fan::color, outline_color);
    fan_create_get_set_define(f32_t, outline_size);

    fan_create_set_define(f32_t, depth);

    fan_create_set_dataless_define(focus);

    void set_line(const fan::vec3& src, const fan::vec3& dst) {
      gloco->shape_set_line(*this, src, dst);
    }

    bool get_blending() {
      return gloco->shape_get_blending(*(shape_t*)this);
    }
    auto get_ri() {
      return gloco->shape_get_ri(*(shape_t*)this);
    }
  };
  #endif

  #if defined(loco_compute_shader)
  #include _FAN_PATH(graphics/vulkan/compute_shader.h)
  #endif

  #if defined(loco_line)
  #define sb_depth_var src
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
  #if defined(loco_pixel_format_renderer)
  #define sb_pfr_var_name pixel_format_renderer
  #define sb_pfr_name pixel_format_renderer_t
  #include _FAN_PATH(graphics/opengl/2D/objects/pixel_format_renderer.h)
  pixel_format_renderer_t sb_pfr_var_name;
  // #undef sb_pfr_name
  // #undef sb_pfr_var_name
  #endif
  #if defined(loco_sprite)
  #define sb_shape_var_name sprite
  #define sb_sprite_name sprite_t
  #include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
  sprite_t sb_shape_var_name;
  #undef sb_shape_var_name
  #endif
  #if defined(loco_unlit_sprite)
  #define sb_shape_var_name unlit_sprite
  #define sb_sprite_name unlit_sprite_t
  #define sb_custom_shape_type loco_t::shape_type_t::unlit_sprite
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/unlit_sprite.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
  unlit_sprite_t unlit_sprite;
  #undef sb_shape_var_name
  #undef sb_custom_shape_type
  #endif
  #if defined(loco_blended_sprite)
  #define sb_shape_var_name blended_sprite
  #define sb_sprite_name blended_sprite_t
  #define sb_custom_shape_type loco_t::shape_type_t::blended_sprite
  #define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/2D/objects/blended_sprite.fs)
  #include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
  unlit_sprite_t blended_sprite;
  #undef sb_shape_var_name
  #undef sb_custom_shape_type
  #endif
  #if defined(loco_light)
  #define sb_shape_name light_t
  #define sb_shape_var_name light
  #define sb_fragment_shader light.fs
  #define sb_is_light
  #include _FAN_PATH(graphics/opengl/2D/objects/light.h)
  sb_shape_name sb_shape_var_name;
  #undef sb_shape_var_name
  #undef sb_fragment_shader
  #undef sb_shape_name
  #undef sb_is_light
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
  #if defined(loco_responsive_text)
  #include _FAN_PATH(graphics/gui/responsive_text.h)
  responsive_text_t responsive_text;
  #endif
  #if defined(loco_button)
  #define sb_shape_var_name button
  #include _FAN_PATH(graphics/gui/button.h)
  button_t sb_shape_var_name;
  #undef sb_shape_var_name
  #endif
  #if defined(loco_text_box)
  #define sb_mark 1
  #include _FAN_PATH(graphics/gui/fed.h)
  #define sb_shape_var_name text_box
  #include _FAN_PATH(graphics/gui/text_box.h)
  text_box_t sb_shape_var_name;
  #undef sb_shape_var_name
  #endif
  #if defined(loco_dropdown)
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

  #if defined(loco_custom_id_t_types)
  using custom_id_t_types_t = std::tuple<loco_custom_id_t_types>;
  #endif

  loco_t(properties_t p = properties_t{ true })
    #ifdef loco_window
    :
  gloco_dummy(this),
    window(fan::vec2(1300, 1300)),
    #endif
    #if defined(loco_context)
    context(
      #if defined(loco_window)
      get_window()
      #endif
    )
    #endif
    #if defined(loco_window)
    , unloaded_image(fan::webp::image_info_t{ (void*)pixel_data, 1 })
    #endif
  {
    #if defined(loco_window)

    root = loco_bdbt_NewNode(&bdbt);

    // set_vsync(p.vsync);

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
    font.open(loco_font);
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

    lp.internal_format = fan::opengl::GL_RGBA;
    lp.format = fan::opengl::GL_RGBA;
    lp.min_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;
    lp.mag_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;
    lp.type = fan::opengl::GL_FLOAT;

    color_buffers[0].load(ii, lp);
    get_context()->opengl.call(get_context()->opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

    color_buffers[0].bind_texture();
    fan::opengl::core::framebuffer_t::bind_to_texture(
      get_context(),
      *color_buffers[0].get_texture(),
      fan::opengl::GL_COLOR_ATTACHMENT0
    );

    lp.internal_format = fan::opengl::GL_RGBA;
    lp.format = fan::opengl::GL_RGBA;

    color_buffers[1].load(ii, lp);

    color_buffers[1].bind_texture();
    fan::opengl::core::framebuffer_t::bind_to_texture(
      get_context(),
      *color_buffers[1].get_texture(),
      fan::opengl::GL_COLOR_ATTACHMENT1
    );

    get_context()->opengl.call(get_context()->opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

    get_window()->add_resize_callback([this](const auto& d) {
      loco_t::image_t::load_properties_t lp;
      lp.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;

      fan::webp::image_info_t ii;
      ii.data = nullptr;
      ii.size = get_window()->get_size();

      lp.internal_format = fan::opengl::GL_RGBA;
      lp.format = fan::opengl::GL_RGBA;
      lp.type = fan::opengl::GL_FLOAT;
      lp.min_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;
      lp.mag_filter = fan::opengl::GL_LINEAR_MIPMAP_LINEAR;

      color_buffers[0].reload_pixels(ii, lp);

      color_buffers[0].bind_texture();
      fan::opengl::core::framebuffer_t::bind_to_texture(
        get_context(),
        *color_buffers[0].get_texture(),
        fan::opengl::GL_COLOR_ATTACHMENT0
      );

      get_context()->opengl.call(get_context()->opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

      lp.internal_format = fan::opengl::GL_RGBA;
      lp.format = fan::opengl::GL_RGBA;

      color_buffers[1].reload_pixels(ii, lp);

      color_buffers[1].bind_texture();
      fan::opengl::core::framebuffer_t::bind_to_texture(
        get_context(),
        *color_buffers[1].get_texture(),
        fan::opengl::GL_COLOR_ATTACHMENT1
      );

      get_context()->opengl.call(get_context()->opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

      fan::opengl::core::renderbuffer_t::properties_t rp;
      m_framebuffer.bind(get_context());
      rp.size = ii.size;
      rp.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
      m_rbo.set_storage(get_context(), rp);

      fan::vec2 window_size = gloco->get_window()->get_size();

      default_viewport.set(fan::vec2(0, 0), d.size, d.size);

      fan::vec2 ratio = window_size.square_normalize();
      //fan::vec2 ortho = fan::vec2(1, 1) * ratio;
      //ortho *= 1.f / ortho.min();
      default_camera->camera.set_ortho(
        fan::vec2(-1, 1),
        fan::vec2(-1, 1),
        &default_viewport
      );
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


  //loco_update_aspect_ratios_cb = [this] {
  //  auto it = cid_list.GetNodeFirst();
  //  while (it != cid_list.dst) {
  //    auto* shape_ptr = (loco_t::shape_t*)&it;

  //    switch ((*((loco_t::cid_nt_t*)shape_ptr))->shape_type) {
  //      case loco_t::shape_type_t::button:
  //      case loco_t::shape_type_t::text_box: {
  //        
  //        //fan::vec2 ratio = shape_ptr->get_viewport()->get_size() / fan::vec2(gloco->get_window()->get_size()).max();

  //        //fan::vec3 position = shape_ptr->get_position();
  //        //
  //        //fan::vec2 size = shape_ptr->get_size() * ratio.y;

  //        //*(fan::vec2*)&position *= ratio.y;
  //        //
  //        //auto* camera = shape_ptr->get_camera();

  //        //// can be bad
  //        //if (position.x - size.x < camera->coordinates.left) {
  //        //  fan::vec3 cp = camera->get_camera_position();
  //        //  cp.x = (position.x - size.x) - camera->coordinates.left;
  //        //  camera->set_camera_position(cp);
  //        //}
  //        //if (position.y - size.y < camera->coordinates.up) {
  //        //  fan::vec3 cp = camera->get_camera_position();
  //        //  cp.y = (position.y - size.y) - camera->coordinates.up;
  //        //  camera->set_camera_position(cp);
  //        //}
  //        /*
  //        else {
  //          fan::vec3 cp = 0;
  //          camera->set_camera_position(cp);
  //        }*/

  //        //shape_ptr->set_position_ar(position);
  //        //shape_ptr->set_size_ar(size);
  //        break;
  //      }
  //      // should be optional to viewport
  //      case loco_t::shape_type_t::responsive_text:
  //      case loco_t::shape_type_t::text: {

  //        fan::vec3 position = shape_ptr->get_position();
  //        fan::vec2 size = shape_ptr->get_size();
  //        auto* camera = shape_ptr->get_camera();
  //       /* if (position.x - size.x < camera->coordinates.left) {
  //          fan::vec3 cp = camera->get_camera_position();
  //          cp.x = (position.x - size.x) - camera->coordinates.left;
  //          camera->set_camera_position(cp);
  //        }
  //        if (position.y - size.y < camera->coordinates.up) {
  //          fan::vec3 cp = camera->get_camera_position();
  //          cp.y = (position.y - size.y) - camera->coordinates.up;
  //          camera->set_camera_position(cp);
  //        }*/
  //        break;
  //      }
  //    }

  //    it = it.Next(&cid_list);
  //  }
  //};

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

        default_texture.create_missing_texture();

        #if defined(loco_line)
        *types.get_value<line_t*>() = &line;
        #endif
        #if defined(loco_rectangle)
        *types.get_value<rectangle_t*>() = &rectangle;
        #endif
        #if defined(loco_sprite)
        *types.get_value<sprite_t*>() = &sprite;
        #endif
        #if defined(loco_unlit_sprite)
        *types.get_value<unlit_sprite_t*>() = &unlit_sprite;
        #endif
        #if defined(loco_circle)
        *types.get_value<circle_t*>() = &circle;
        #endif
        #if defined(loco_button)
        *types.get_value<button_t*>() = &button;
        #endif
        #if defined(loco_letter)
        *types.get_value<letter_t*>() = &letter;
        #endif
        #if defined(loco_text)
        *types.get_value<text_t*>() = &text;
        #endif
        #if defined(loco_responsive_text)
        *types.get_value<responsive_text_t*>() = &responsive_text;
        #endif
        #if defined(loco_light)
        *types.get_value<light_t*>() = &light;
        #endif
        #if defined(loco_text_box)
        *types.get_value<text_box_t*>() = &text_box;
        #endif
        #if defined(loco_vfi)
        *types.get_value<vfi_t*>() = &vfi;
        #endif
        #if defined(loco_pixel_format_renderer)
        *types.get_value<pixel_format_renderer_t*>() = &pixel_format_renderer;
        #endif

        #if defined(loco_t_id_t_types)
        #if !defined(loco_t_id_t_ptrs)
        #error loco_t_id_t_ptrs not defined
        #else
        std::apply([&](const auto&... args) {
          ((*types.get_value<std::remove_const_t<std::remove_reference_t<decltype(args)>>>() = args), ...);
          }, std::tuple<loco_t_id_t_types>{ loco_t_id_t_ptrs });
        #endif
        #endif

    fan::vec2 window_size = get_window()->get_size();
    open_viewport(&default_viewport, fan::vec2(0, 0), window_size);

    default_camera = add_camera(fan::graphics::direction_e::right);

    open_camera(&default_camera->camera,
      fan::vec2(-1, 1),
      fan::vec2(-1, 1)
    );

  #if defined(loco_physics)
    fan::graphics::open_bcol();
  #endif

    #if defined(loco_imgui)
    auto hwnd = get_window()->get_handle();

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_ViewportsEnable;

    get_window()->add_buttons_callback([&](const auto& d) {
      io.AddMouseButtonEvent(d.button - fan::mouse_left, (bool)d.state);
    });
    get_window()->add_keys_callback([&](const auto& d) {
      ImGuiKey imgui_key = fan::window_input::fan_to_imguikey(d.key);
      io.AddKeyEvent(imgui_key, (int)d.state);
    });
    get_window()->add_text_callback([&](const auto& d) {
      io.AddInputCharacter(d.character);
    });

    loco_t::imgui_themes::dark();

    #if defined(fan_platform_windows)
    ImGui_ImplWin32_Init(hwnd);
    #elif defined(fan_platform_linux)
    imgui_xorg_init();
    #endif
    ImGui_ImplOpenGL3_Init();
    #endif

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
    color_buffers[0].bind_texture();

    get_context()->opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
    color_buffers[1].bind_texture();


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
    get_context()->opengl.glDrawBuffer(fan::opengl::GL_COLOR_ATTACHMENT1);
    get_context()->opengl.glClearColor(0, 0, 0, 1);
    get_context()->opengl.glClear(fan::opengl::GL_COLOR_BUFFER_BIT);
    get_context()->opengl.glDrawBuffer(fan::opengl::GL_COLOR_ATTACHMENT0);
    #endif
    get_context()->opengl.glClearColor(0.10, 0.10, 0.131, 1.0f);
    get_context()->opengl.call(get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
    #endif

    #ifdef loco_post_process
    post_process.start_capture();
    #endif

    auto it = m_update_callback.GetNodeFirst();
    while (it != m_update_callback.dst) {
      m_update_callback.StartSafeNext(it);
      m_update_callback[it](this);
      it = m_update_callback.EndSafeNext();
    }

    m_write_queue.process(get_context());

    #ifdef loco_window
    #if defined(loco_opengl)

    #include "draw_shapes.h"

    #if defined(loco_framebuffer)
    //m_flag_map_fbo.unbind(get_context());

    m_framebuffer.unbind(get_context());

    get_context()->opengl.glClearColor(0, 0, 0, 1);
    get_context()->opengl.call(get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
    fan::vec2 window_size = get_window()->get_size();
    fan::opengl::viewport_t::set_viewport(0, window_size, window_size);

    m_fbo_final_shader.use(get_context());
    m_fbo_final_shader.set_int(get_context(), "_t00", 0);
    m_fbo_final_shader.set_int(get_context(), "_t01", 1);

    get_context()->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
    color_buffers[0].bind_texture();

    get_context()->opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
    color_buffers[1].bind_texture();

    unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];
    for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
      attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
    }

    //get_context()->opengl.call(get_context()->opengl.glDrawBuffers, std::size(attachments), attachments);

    renderQuad();

    #if defined(loco_imgui)
    ImGui_ImplOpenGL3_NewFrame();
    #if defined(fan_platform_windows)
    ImGui_ImplWin32_NewFrame();
    #elif defined(fan_platform_linux)
    imgui_xorg_new_frame();
    #endif
    ImGui::NewFrame();

    {
      auto it = m_imgui_draw_cb.GetNodeFirst();
      while (it != m_imgui_draw_cb.dst) {
        m_imgui_draw_cb.StartSafeNext(it);
        m_imgui_draw_cb[it]();
        it = m_imgui_draw_cb.EndSafeNext();
      }
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    #endif

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

  fan::vec2 get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport) {
    fan::vec2 mouse_pos = get_window()->get_mouse_position();
    fan::vec2 translated_pos;
    translated_pos.x = fan::math::map(mouse_pos.x, viewport.get_position().x, viewport.get_position().x + viewport.get_size().x, camera.coordinates.left, camera.coordinates.right);
    translated_pos.y = fan::math::map(mouse_pos.y, viewport.get_position().y, viewport.get_position().y + viewport.get_size().y, camera.coordinates.up, camera.coordinates.down);
    return translated_pos;
  }

  fan::vec2 get_mouse_position() {
    return get_mouse_position(gloco->default_camera->camera, gloco->default_camera->viewport);
  }


  #if defined(loco_vulkan)
  fan::vulkan::shader_t render_fullscreen_shader;
  #endif

  #if defined(loco_framebuffer)

  #if defined(loco_opengl)
  fan::opengl::core::framebuffer_t m_flag_map_fbo;

  fan::opengl::core::framebuffer_t m_framebuffer;
  fan::opengl::core::renderbuffer_t m_rbo;
  loco_t::image_t color_buffers[2];
  fan::opengl::shader_t m_fbo_final_shader;

  #elif defined(loco_vulkan)

  #endif

  #endif

  bool process_loop(const auto& lambda) {

    auto it = get_window()->add_resize_callback([this, &lambda](const auto& d) {
      gloco->process_loop(lambda);
    });

    uint32_t window_event = get_window()->handle_events();
    if (window_event & fan::window_t::events::close) {
      get_window()->destroy_window();
      return 1;
    }

    get_window()->remove_resize_callback(it);

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

  #if defined(loco_cuda)

  struct cuda_textures_t {

    cuda_textures_t() {
      inited = false;
    }
    ~cuda_textures_t() {
    }
    void close(loco_t* loco, loco_t::shape_t& cid) {
      uint8_t image_amount = fan::pixel_format::get_texture_amount(loco->pixel_format_renderer.sb_get_ri(cid).format);
      auto& ri = loco->pixel_format_renderer.sb_get_ri(cid);
      for (uint32_t i = 0; i < image_amount; ++i) {
        wresources[i].close();
        ri.images[i].unload();
      }
    }

    void resize(loco_t* loco, loco_t::cid_nt_t& id, uint8_t format, fan::vec2ui size, uint32_t filter = loco_t::image_t::filter::linear) {
      auto& ri = loco->pixel_format_renderer.sb_get_ri(id);
      uint8_t image_amount = fan::pixel_format::get_texture_amount(format);
      if (inited == false) {
        // purge cid's images here
        // update cids images
        loco->pixel_format_renderer.reload(id, format, size, filter);
        for (uint32_t i = 0; i < image_amount; ++i) {
          wresources[i].open(loco->image_list[ri.images[i].texture_reference].texture_id);
        }
        inited = true;
      }
      else {

        if (ri.images[0].size == size) {
          return;
        }

        // update cids images
        for (uint32_t i = 0; i < fan::pixel_format::get_texture_amount(ri.format); ++i) {
          wresources[i].close();
        }

        loco->pixel_format_renderer.reload(id, format, size, filter);

        for (uint32_t i = 0; i < image_amount; ++i) {
          wresources[i].open(loco->image_list[ri.images[i].texture_reference].texture_id);
        }
      }
    }

    cudaArray_t& get_array(uint32_t index) {
      return wresources[index].cuda_array;
    }

    struct graphics_resource_t {
      void open(int texture_id) {
        fan::cuda::check_error(cudaGraphicsGLRegisterImage(&resource, texture_id, fan::opengl::GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
        map();
      }
      void close() {
        unmap();
        fan::cuda::check_error(cudaGraphicsUnregisterResource(resource));
        resource = nullptr;
      }
      void map() {
        fan::cuda::check_error(cudaGraphicsMapResources(1, &resource, 0));
        fan::cuda::check_error(cudaGraphicsSubResourceGetMappedArray(&cuda_array, resource, 0, 0));
        fan::print("+", resource);
      }
      void unmap() {
        fan::print("-", resource);
        fan::cuda::check_error(cudaGraphicsUnmapResources(1, &resource));
        //fan::cuda::check_error(cudaGraphicsResourceSetMapFlags(resource, 0));
      }
      //void reload(int texture_id) {
      //  close();
      //  open(texture_id);
      //}
      cudaGraphicsResource_t resource = nullptr;
      cudaArray_t cuda_array = nullptr;
    };

    bool inited = false;
    graphics_resource_t wresources[4];
  };

  #endif

protected:
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_SafeNext 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix update_callback
  #define BLL_set_BaseLibrary 1
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType fan::function_t<void(loco_t*)>
  #include _FAN_PATH(BLL/BLL.h)
public:
#if defined(loco_imgui)
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_SafeNext 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix imgui_draw_cb
  #define BLL_set_BaseLibrary 1
  #define BLL_set_Link 1
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeDataType fan::function_t<void()>
  #include _FAN_PATH(BLL/BLL.h)
#endif

  using update_callback_nr_t = update_callback_NodeReference_t;

  update_callback_t m_update_callback;

#if defined(loco_imgui)
  using imgui_draw_cb_nr_t = imgui_draw_cb_NodeReference_t;
  imgui_draw_cb_t m_imgui_draw_cb;
#endif

  image_t default_texture;

  struct comma_dummy_t {
    uint8_t member_pointer;
    static constexpr typename loco_t::shape_type_t::_t shape_type = -1;
  };

  // requires shape_type create to shape.h, init in constructor, add type_t to properties
  // make get_properties for custom type
  fan::masterpiece_t <
    comma_dummy_t*
    #if defined(loco_rectangle)
    , rectangle_t*
    #endif
    #if defined(loco_sprite)
    , sprite_t*
    #endif
    #if defined(loco_unlit_sprite)
    , unlit_sprite_t*
    #endif
    #if defined(loco_button)
    , button_t*
    #endif
    #if defined(loco_letter)
    , letter_t*
    #endif
    #if defined(loco_text)
    , text_t*
    #endif
    #if defined(loco_responsive_text)
    , responsive_text_t*
    #endif
    #if defined(loco_light)
    , light_t*
    #endif
    #if defined(loco_t_id_t_types)
    , loco_t_id_t_types
    #endif
    #if defined(loco_line)
    , line_t*
    #endif
    #if defined(loco_circle)
    , circle_t*
    #endif
    #if defined(loco_text_box)
    , text_box_t*
    #endif
    #if defined(loco_pixel_format_renderer)
    , pixel_format_renderer_t*
    #endif
    #if defined(loco_vfi)
    , vfi_t*
    #endif
  > types;

  struct vfi_id_t {
    using properties_t = loco_t::vfi_t::properties_t;
    operator loco_t::vfi_t::shape_id_t* () {
      return &cid;
    }
    vfi_id_t() = default;
    vfi_id_t(const properties_t& p) {
      gloco->vfi.push_back(*this, *(properties_t*)&p);
    }
    vfi_id_t& operator[](const properties_t& p) {
      gloco->vfi.push_back(*this, *(properties_t*)&p);
      return *this;
    }
    ~vfi_id_t() {
      gloco->vfi.erase(*this);
    }

    loco_t::vfi_t::shape_id_t cid;
  };

  #define make_key_value(type, name) \
      type& name = *key.get_value<decltype(key)::get_index_with_type<type>()>();

  template <typename T>
  void push_shape(loco_t::cid_nt_t& id, T properties) {
    if constexpr (std::is_same_v<loco_t::vfi_t, typename T::type_t>) {
      loco_t::vfi_t::shape_id_t shape_id;
      (*types.get_value<typename T::type_t*>())->push_back(&shape_id, properties);
      id->shape_type = loco_t::shape_type_t::hitbox;
      *id.gdp4() = shape_id.NRI;
    }
    else if constexpr (!std::is_same_v<std::nullptr_t, T>) {
      (*types.get_value<typename T::type_t*>())->push_back(id, properties);
    }
  }

  void shape_get_properties(loco_t::cid_nt_t& id, auto lambda) {
    types.iterate([&]<typename T>(auto shape_index, T shape) {
      using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>;
      if (shape_t::shape_type == id->shape_type) {
        if constexpr (has_get_properties_v<shape_t, loco_t::cid_nt_t&>) {
          lambda((*shape)->get_properties(id));
        }
        else if constexpr (has_sb_get_properties_v<shape_t, loco_t::cid_nt_t&>) {
          lambda((*shape)->sb_get_properties(id));
        }
      }
    });
  }

  bool shape_get_blending(loco_t::cid_nt_t& id) {
    bool blending = false;
    types.iterate([&]<typename T>(auto shape_index, T shape) {
      using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>;
      if (shape_t::shape_type == id->shape_type) {
        if constexpr (has_sb_get_ri_v<shape_t, loco_t::cid_nt_t>) {
          blending = (*shape)->sb_get_ri(id).blending;
        }
      }
    });
    return blending;
  }
  std::pair<void*, uint16_t> shape_get_ri(loco_t::cid_nt_t& id) {
    std::pair<void*, uint16_t> ret;
    types.iterate([&]<typename T>(auto shape_index, T shape) {
      using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>;
      if (shape_t::shape_type == id->shape_type) {
        if constexpr (has_sb_get_ri_v<shape_t, loco_t::cid_nt_t&>) {
          ret.second = sizeof((*shape)->sb_get_ri(id));
          memcpy(ret.first, &(*shape)->sb_get_ri(id), ret.second);
        }
      }
    });
    return ret;
  }

  #define make_global_function_declare(func_name, content, ...) \
  fan_has_function_concept(func_name);\
  void shape_ ## func_name(__VA_ARGS__);

  #define make_global_function_ret_define(ret, func_name, content, ...) \
  fan_has_function_concept(func_name);\
  ret shape_ ## func_name(__VA_ARGS__) { \
    ret data; \
    types.iterate([&]<typename T>(auto shape_index, T shape) { \
      using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>; \
      if (shape_t::shape_type == id->shape_type) { \
        content \
      } \
    }); \
    return data; \
  }

  #define make_global_function_define(func_name, content, ...) \
  fan_has_function_concept(func_name);\
  void shape_ ## func_name(__VA_ARGS__) { \
    types.iterate_ret([&]<typename T>(auto shape_index, T shape) -> int{ \
      using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>; \
      if (shape_t::shape_type == id->shape_type) { \
        content \
        return 1; \
      } \
      return 0; \
    }); \
  }

  #define make_global_function_define_custom_shape(func_name, content, ...) \
  fan_has_function_concept(func_name);\
  void shape_ ## func_name(__VA_ARGS__) { \
    types.iterate([&]<typename T>(auto shape_index, T shape) { \
      using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>; \
      if (shape_t::shape_type == shape_type) { \
        content \
      } \
    }); \
  }

  make_global_function_define(erase,
    if constexpr (has_erase_v<shape_t, loco_t::cid_nt_t&>) {
      (*shape)->erase(id);
    }
    else if constexpr (has_erase_v<shape_t, loco_t::vfi_t::shape_id_t*>) {
      (*shape)->erase((loco_t::vfi_t::shape_id_t*)id.gdp4());
    }
    ,
    loco_t::cid_nt_t& id
    );

  make_global_function_ret_define(bool, append_letter,
    if constexpr (has_append_letter_v<shape_t, loco_t::cid_nt_t&, wchar_t>) {
      data = (*shape)->append_letter(id, wc, force);
    },
      loco_t::cid_nt_t& id,
      wchar_t wc,
      bool force = false
      );

  fan_has_function_concept(get);
  fan_has_function_concept(set);

  #define fan_build_get_declare(rt, name) \
  fan_has_variable_struct(name); \
  fan_has_function_concept(get_##name); \
  rt shape_get_##name(loco_t::cid_nt_t&);

  #define fan_build_get_define(rt, name) \
  fan_has_variable_struct(name); \
  fan_has_function_concept(get_##name); \
  rt shape_get_##name(loco_t::cid_nt_t& id) { \
    rt data; \
    types.iterate([&]<typename T>(auto shape_index, T shape) {\
      using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>; \
      if (shape_t::shape_type == id->shape_type) {\
        if constexpr (has_get_instance_v<shape_t, loco_t::cid_nt_t&>) { \
          if constexpr(has_##name##_v<decltype((*shape)->get_instance(id))>) {\
            data = (*shape)->get_instance(id).name; \
          }\
        }\
        else if constexpr (has_get_##name##_v<shape_t, loco_t::cid_nt_t&>) {\
          data = (*shape)->get_##name(id);\
        }\
        else if constexpr (has_get_v<shape_t, loco_t::cid_nt_t&, decltype(&comma_dummy_t::member_pointer)>) {\
          if constexpr(has_##name##_v<typename shape_t::properties_t>) {\
            data = (*shape)->get(id, &shape_t::vi_t::name); \
          }\
        }\
        else if constexpr (has_get_properties_v<shape_t, loco_t::cid_nt_t&>) { \
          if constexpr(has_##name##_v<typename shape_t::properties_t>) {\
            data = (*shape)->get_properties(id).name; \
          }\
        } \
        else if constexpr (has_get_##name##_v<shape_t, loco_t::vfi_t::shape_id_t*>) {\
          data = (*shape)->get_##name((loco_t::vfi_t::shape_id_t*)id.gdp4());\
        }\
      }\
    });\
    return data; \
  }

  #define fan_build_set_declare(rt, name) \
  make_global_function_declare(set_##name,\
    if constexpr (has_set_##name##_v<shape_t, loco_t::cid_nt_t&, const rt&>) { \
      if constexpr(has_##name##_v<typename shape_t::properties_t>) {\
        (*shape)->set_##name(id, data); \
      } \
    }  \
    else if constexpr (has_set_##name##_v<shape_t, loco_t::vfi_t::shape_id_t*, const rt&>) { \
      (*shape)->set_##name((loco_t::vfi_t::shape_id_t*)id.gdp4(), data); \
    }\
    else if constexpr (has_set_v<shape_t, loco_t::cid_nt_t&, decltype(&comma_dummy_t::member_pointer), void*>) { \
      if constexpr(has_##name##_v<typename shape_t::properties_t>) {\
        (*shape)->set(id, &shape_t::vi_t::name, data); \
      }\
    } , \
    loco_t::cid_nt_t& id, \
    const auto& data \
  );

  #define fan_build_set_define(rt, name) \
  make_global_function_define(set_##name,\
    if constexpr (has_set_##name##_v<shape_t, loco_t::cid_nt_t&, const rt&>) { \
      if constexpr(has_##name##_v<typename shape_t::properties_t>) {\
        (*shape)->set_##name(id, data); \
      } \
    } \
    else if constexpr (has_set_##name##_v<shape_t, loco_t::vfi_t::shape_id_t*, const rt&>) { \
      (*shape)->set_##name((loco_t::vfi_t::shape_id_t*)id.gdp4(), data); \
    } \
    else if constexpr (has_set_v<shape_t, loco_t::cid_nt_t&, decltype(&comma_dummy_t::member_pointer), void*>) { \
      if constexpr(has_##name##_v<typename shape_t::properties_t>) {\
        (*shape)->set(id, &shape_t::vi_t::name, data); \
      }\
    }, \
    loco_t::cid_nt_t& id, \
    const auto& data \
  );

  fan_has_function_concept(get_instance);

  #define fan_build_get_generic_declare(rt, name) \
  fan_has_variable_struct(name); \
  fan_has_function_concept(get_##name); \
  rt shape_get_##name(loco_t::cid_nt_t& id);

  // NOTE for including in c/.h need to add loco_t:: infront of function
  #define fan_build_get_generic_define(rt, name) \
  fan_has_variable_struct(name); \
  fan_has_function_concept(get_##name); \
  rt shape_get_##name(loco_t::cid_nt_t& id) { \
    rt data; \
    types.iterate([&]<typename T>(auto shape_index, T shape) {\
      using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>; \
      if (shape_t::shape_type == id->shape_type) {\
        if constexpr (has_get_##name##_v<shape_t, loco_t::cid_nt_t&>) {\
            data = (*shape)->get_##name(id);\
        } \
        else if constexpr (has_get_properties_v<shape_t, loco_t::cid_nt_t&>) { \
          if constexpr(has_##name##_v<typename shape_t::properties_t>) {\
            data = (*shape)->get_properties(id).name; \
          }\
        }\
      }\
    });\
    return data; \
  }

  #define fan_build_set_generic_declare(rt, name) \
  make_global_function_declare(set_##name,\
    if constexpr (has_set_##name##_v<shape_t, loco_t::cid_nt_t&, const rt&>) { \
      (*shape)->set_##name(id, data); \
    }, \
    loco_t::cid_nt_t& id, \
    const auto& data \
  );

  #define fan_build_set_generic_define(rt, name) \
  make_global_function_define(set_##name,\
    if constexpr (has_set_##name##_v<shape_t, loco_t::cid_nt_t&, const rt&>) { \
      (*shape)->set_##name(id, data); \
    }, \
    loco_t::cid_nt_t& id, \
    const auto& data \
  );

  #define fan_build_set_ptr_generic_define(rt, name) \
  make_global_function_define(set_##name,\
    if constexpr (has_set_##name##_v<shape_t, loco_t::cid_nt_t&, rt>) { \
      (*shape)->set_##name(id, data); \
    }, \
    loco_t::cid_nt_t& id, \
    auto data \
  );

  #define fan_build_set_generic_dataless_define(name) \
  make_global_function_define(set_##name,\
    if constexpr (has_set_##name##_v<shape_t, loco_t::cid_nt_t&>) { \
      (*shape)->set_##name(id); \
    }, \
    loco_t::cid_nt_t& id \
  );

  #define fan_build_get_set_generic_declare( rt, name) \
    fan_build_get_generic_declare(rt, name); \
    fan_build_set_generic_declare(rt, name);

  #define fan_build_get_set_generic_define( rt, name) \
    fan_build_get_generic_define(rt, name); \
    fan_build_set_generic_define(rt, name);

  #define fan_build_get_set_ptr_generic_define( rt, name) \
    fan_build_get_generic_define(rt, name); \
    fan_build_set_ptr_generic_define(rt, name);

  #define fan_build_get_set_declare(rt, name) \
    fan_build_get_declare(rt, name); \
    fan_build_set_declare(rt, name);

  #define fan_build_get_set_define(rt, name) \
    fan_build_get_define(rt, name); \
    fan_build_set_define(rt, name);

  fan_build_get_set_generic_define(fan::vec3, position_ar);
  fan_build_get_set_generic_define(fan::vec2, size_ar);

  fan_build_get_set_ptr_generic_define(loco_t::viewport_t*, viewport);
  fan_build_get_set_ptr_generic_define(loco_t::camera_t*, camera);

  fan_build_get_set_define(fan::vec3, position);
  fan_build_get_set_define(fan::vec2, size);
  fan_build_get_set_define(fan::color, color);
  fan_build_get_set_define(f32_t, angle);
  fan_build_get_set_define(fan::vec2, rotation_point);

  fan_build_get_set_generic_define(loco_t::textureid_t<0>, image);

  fan_build_get_set_generic_define(f32_t, font_size);
  fan_build_get_set_generic_define(fan::vec2, text_size);
  fan_build_set_generic_dataless_define(focus);

  fan_build_get_set_generic_define(fan::string, text);

  fan_build_get_set_define(fan::color, outline_color);
  fan_build_get_set_define(f32_t, outline_size);

  make_global_function_define(set_line,
    if constexpr (has_set_line_v<shape_t, loco_t::cid_nt_t&, const fan::vec3&, const fan::vec3&>) {
      (*shape)->set_line(id, src, dst);
    },
      loco_t::cid_nt_t& id,
      const fan::vec3& src,
      const fan::vec3& dst
      );

  fan_has_function_concept(sb_set_depth);
  make_global_function_define(set_depth,
    if constexpr (has_set_depth_v<shape_t, loco_t::cid_nt_t&, f32_t>) {
      (*shape)->set_depth(id, data);
    }
    else if constexpr (has_sb_set_depth_v<shape_t, loco_t::cid_nt_t&, f32_t>) {
      (*shape)->sb_set_depth(id, data);
    },
      loco_t::cid_nt_t& id,
      const auto& data
      );

  make_global_function_define_custom_shape(draw,
    if constexpr (has_draw_v<shape_t, const redraw_key_t&, loco_bdbt_NodeReference_t>) {
      (*shape)->draw(redraw_key, nr);
    },
      shape_type_t::_t shape_type,
      const redraw_key_t& redraw_key,
      loco_bdbt_NodeReference_t nr
      );

  fan_has_function_concept(sb_get_properties);
  fan_has_function_concept(get_properties);
  fan_has_function_concept(sb_get_ri);
  fan_has_function_concept(get_ri);

  //make_global_function(get_properties,
  //  if constexpr (has_get_properties_v<shape_t, loco_t::cid_t*>) {
  //    lambda((*shape)->get_properties(cid));
  //  },
  //  cid_t* cid,
  //  auto lambda
  //);

  template <typename T>
  T* get_shape() {
    return *types.get_value<T*>();
  }

  fan::vec2 get_camera_view_size(loco_t::camera_t camera) {
    return fan::vec2(
      std::abs(camera.coordinates.right) + std::abs(camera.coordinates.left),
      std::abs(camera.coordinates.down) + std::abs(camera.coordinates.up)
    );
  }

  static inline std::vector<camera_impl_t*> viewport_handler;
  camera_impl_t* add_camera(fan::graphics::direction_e split_direction) {
    viewport_handler.push_back(new camera_impl_t(split_direction));
    int index = 0;
    fan::vec2 window_size = gloco->get_window()->get_size();
    gloco->viewport_divider.iterate([&index, window_size](auto& node) {
      viewport_handler[index]->viewport.set(
        (node.position - node.size / 2) * window_size,
        ((node.size) * window_size), window_size
      );
      index++;
    });
    return viewport_handler.back();
  }

  loco_t::theme_t default_theme = loco_t::themes::gray();
  camera_impl_t* default_camera;
  loco_t::viewport_t default_viewport;

  fan::graphics::viewport_divider_t viewport_divider;

  #undef make_global_function
  #undef fan_build_get
  #undef fan_build_set
};

#if defined(loco_pixel_format_renderer)
template <typename T>
constexpr std::array<T, 4> fan::pixel_format::get_image_properties(uint8_t format) {
  switch (format) {
    case yuv420p: {
      return std::array<loco_t::image_t::load_properties_t, 4>{
        loco_t::image_t::load_properties_t{
          .internal_format = loco_t::image_t::format::r8_unorm,
          .format = loco_t::image_t::format::r8_unorm
        },
          loco_t::image_t::load_properties_t{
            .internal_format = loco_t::image_t::format::r8_unorm,
            .format = loco_t::image_t::format::r8_unorm
        },
          loco_t::image_t::load_properties_t{
            .internal_format = loco_t::image_t::format::r8_unorm,
            .format = loco_t::image_t::format::r8_unorm
        }
      };
    }
    case nv12: {
      return std::array<loco_t::image_t::load_properties_t, 4>{
        loco_t::image_t::load_properties_t{
          .internal_format = loco_t::image_t::format::r8_unorm,
          .format = loco_t::image_t::format::r8_unorm
        },
          loco_t::image_t::load_properties_t{
            .internal_format = loco_t::image_t::format::rg8_unorm,
            .format = loco_t::image_t::format::rg8_unorm
        }
      };
    }
    default: {
      fan::throw_error("invalid format");
      return std::array<loco_t::image_t::load_properties_t, 4>{};
    }
  }
}

#endif


#ifndef loco_no_inline
#undef loco_rectangle_vi_t

/* #undef loco_rectangle
 #undef loco_sprite
 #undef loco_letter
 #undef loco_text
 #undef loco_text_box
 #undef loco_button
 #undef loco_wboit*/
#endif

#define loco_make_shape(type, ...) fan_init_struct(type, __VA_ARGS__)

inline void fan::opengl::viewport_t::open() {
  viewport_reference = gloco->get_context()->viewport_list.NewNode();
  gloco->get_context()->viewport_list[viewport_reference].viewport_id = this;
}

inline void fan::opengl::viewport_t::close() {
  gloco->get_context()->viewport_list.Recycle(viewport_reference);
}

inline void fan::opengl::viewport_t::set_viewport(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  gloco->get_context()->opengl.call(
    gloco->get_context()->opengl.glViewport,
    viewport_position_.x,
    window_size.y - viewport_size_.y - viewport_position_.y,
    viewport_size_.x, viewport_size_.y
  );
}

namespace fan {
  namespace graphics {

    using camera_t = loco_t::camera_impl_t;
    // use bll to avoid 'new'
    static auto add_camera(fan::graphics::direction_e split_direction) {
      return gloco->add_camera(split_direction);
    }

    struct line_properties_t {
      fan::graphics::camera_t* camera = gloco->default_camera;
      fan::vec3 src = fan::vec3(0, 0, 0);
      fan::vec2 dst = fan::vec2(1, 1);
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = false;
    };

    struct line_t : loco_t::shape_t {
      line_t(line_properties_t p = line_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            loco_t::line_t::properties_t,
            .camera = &p.camera->camera,
            .viewport = &p.camera->viewport,
            .src = p.src,
            .dst = p.dst,
            .color = p.color,
            .blending = p.blending
          ));
      }
    };

    struct rectangle_properties_t {
      fan::graphics::camera_t* camera = gloco->default_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = false;
    };

    struct rectangle_t : loco_t::shape_t {
      rectangle_t(rectangle_properties_t p = rectangle_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            loco_t::rectangle_t::properties_t,
            .camera = &p.camera->camera,
            .viewport = &p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .color = p.color,
            .blending = p.blending
          ));
      }
    };

    struct circle_properties_t {
      fan::graphics::camera_t* camera = gloco->default_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      f32_t radius = 0.1;
      fan::color color = fan::color(1, 1, 1, 1);
      bool blending = false;
    };

    struct circle_t : loco_t::shape_t {
      circle_t(circle_properties_t p = circle_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            loco_t::circle_t::properties_t,
            .camera = &p.camera->camera,
            .viewport = &p.camera->viewport,
            .position = p.position,
            .radius = p.radius,
            .color = p.color,
            .blending = p.blending
          ));
      }
    };

    struct unlit_sprite_properties_t {
      fan::graphics::camera_t* camera = gloco->default_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::color color = fan::color(1, 1, 1, 1);
      loco_t::image_t* image = &gloco->default_texture;
      bool blending = false;
      fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    };

    struct unlit_sprite_t : loco_t::shape_t {
      unlit_sprite_t(unlit_sprite_properties_t p = unlit_sprite_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            loco_t::unlit_sprite_t::properties_t,
            .camera = &p.camera->camera,
            .viewport = &p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .image = p.image,
            .color = p.color,
            .blending = p.blending,
            .rotation_vector = p.rotation_vector
          ));
      }
    };

    struct sprite_properties_t {
      fan::graphics::camera_t* camera = gloco->default_camera;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      fan::color color = fan::color(1, 1, 1, 1);
      loco_t::image_t* image = &gloco->default_texture;
      bool blending = false;
      fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
    };

    struct sprite_t : loco_t::shape_t {
      sprite_t(sprite_properties_t p = sprite_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            loco_t::sprite_t::properties_t,
            .camera = &p.camera->camera,
            .viewport = &p.camera->viewport,
            .position = p.position,
            .size = p.size,
            .image = p.image,
            .color = p.color,
            .blending = p.blending,
            .rotation_vector = p.rotation_vector
          ));
      }
    };

    #if defined(loco_text)

    struct letter_properties_t {
      loco_t::camera_t* camera = &gloco->default_camera->camera;
      loco_t::viewport_t* viewport = &gloco->default_viewport;
      fan::color color = fan::colors::white;
      fan::vec3 position = fan::vec3(0, 0, 0);
      f32_t font_size = 1;
      uint32_t letter_id;
    };

    struct letter_t : loco_t::shape_t {
      letter_t(letter_properties_t p = letter_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::letter_t::properties_t,
            .camera = p.camera,
            .viewport = p.viewport,
            .position = p.position,
            .font_size = p.font_size,
            .letter_id = p.letter_id,
            .color = p.color
          ));
      }
    };

    struct text_properties_t {
      loco_t::camera_t* camera = &gloco->default_camera->camera;
      loco_t::viewport_t* viewport = &gloco->default_viewport;
      std::string text = "";
      fan::color color = fan::colors::white;
      fan::vec3 position = fan::vec3(fan::math::inf, -0.9, 0);
    };

    struct text_t : loco_t::shape_t {
      text_t(text_properties_t p = text_properties_t()) {
        *(loco_t::shape_t*)this = loco_t::shape_t(
          fan_init_struct(
            typename loco_t::responsive_text_t::properties_t,
            .camera = p.camera,
            .viewport = p.viewport,
            .position = p.position.x == fan::math::inf ? fan::vec3(-1 + 0.025 * p.text.size(), -0.9, 0) : p.position,
            .text = p.text,
            .line_limit = 1,
            .letter_size_y_multipler = 1,
            .size = fan::vec2(0.025 * p.text.size(), 0.1),
            .color = p.color
          ));
      }
    };
    #endif

    #if defined(loco_button)
    struct button_properties_t {
      loco_t::theme_t* theme = &gloco->default_theme;
      loco_t::camera_t* camera = &gloco->default_camera->camera;
      loco_t::viewport_t* viewport = &gloco->default_viewport;
      fan::vec3 position = fan::vec3(0, 0, 0);
      fan::vec2 size = fan::vec2(0.1, 0.1);
      std::string text = "button";
      loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> int { return 0; };
      loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> int { return 0; };
    };

    struct button_t : loco_t::shape_t {
      button_t(button_properties_t p = button_properties_t()) : loco_t::shape_t(
        fan_init_struct(
          loco_t::button_t::properties_t,
          .theme = p.theme,
          .camera = p.camera,
          .viewport = p.viewport,
          .position = p.position,
          .size = p.size,
          .text = p.text,
          .mouse_button_cb = p.mouse_button_cb
        )) {}
    };
    #endif

    #if defined(loco_imgui)
    struct imgui_element_nr_t : loco_t::imgui_draw_cb_nr_t {
      using base_t = loco_t::imgui_draw_cb_nr_t;

      imgui_element_nr_t() { /**(nr_t*)this = cid_list_gnric(); should be default*/ }

      imgui_element_nr_t(const imgui_element_nr_t& nr) : imgui_element_nr_t() {
        if (nr.is_invalid()) {
          return;
        }
        init();
      }

      imgui_element_nr_t(imgui_element_nr_t&& nr) {
        NRI = nr.NRI;
        nr.invalidate_soft();
      }
      ~imgui_element_nr_t() {
        invalidate();
      }


      imgui_element_nr_t& operator=(const imgui_element_nr_t& id) {
        if (!is_invalid()) {
          invalidate();
        }
        if (id.is_invalid()) {
          return *this;
        }

        if (this != &id) {
          init();
        }
        return *this;
      }

      imgui_element_nr_t& operator=(imgui_element_nr_t&& id) {
        if (!is_invalid()) {
          invalidate();
        }
        if (id.is_invalid()) {
          return *this;
        }

        if (this != &id) {
          if (!is_invalid()) {
            invalidate();
          }
          NRI = id.NRI;

          id.invalidate_soft();
        }
        return *this;
      }

      void init() {
        *(base_t*)this = gloco->m_imgui_draw_cb.NewNodeLast();
      }

      bool is_invalid() const {
        return loco_t::imgui_draw_cb_inric(*this);
      }

      void invalidate_soft() {
        *(base_t*)this = gloco->m_imgui_draw_cb.gnric();
      }

      void invalidate() {
        if (is_invalid()) {
          return;
        }
        gloco->m_imgui_draw_cb.unlrec(*this);
        *(base_t*)this = gloco->m_imgui_draw_cb.gnric();
      }

      void set(const auto& lambda) {
        gloco->m_imgui_draw_cb[*this] = lambda;
      }
    };

    struct imgui_element_t : imgui_element_nr_t {
      imgui_element_t() = default;
      imgui_element_t(const auto& lambda) {
        imgui_element_nr_t::init();
        imgui_element_nr_t::set(lambda);
      }
    };

    struct imgui_shape_element_t : imgui_element_t, loco_t::shape_t {
      imgui_shape_element_t(const auto& properties, const auto& lambda)
        : imgui_element_t(lambda), loco_t::shape_t(properties) {
      }
    };
    #endif

    // REQUIRES to be allocated by new since lambda captures this
    // also container that it's stored in, must not change pointers
    struct vfi_root_t {
      //vfi_root_t() = default;

      void set_root(const loco_t::vfi_t::properties_t& p) {
        loco_t::vfi_t::properties_t in = p;
        in.shape_type = loco_t::vfi_t::shape_t::rectangle;
        in.shape.rectangle->viewport = &gloco->default_camera->viewport;
        in.shape.rectangle->camera = &gloco->default_camera->camera;
        in.keyboard_cb = [this, user_cb = p.keyboard_cb](const auto& d) -> int {
          if (d.key == fan::key_c && 
            (d.keyboard_state == fan::keyboard_state::press ||
            d.keyboard_state == fan::keyboard_state::repeat)) {
            this->resize = true;
            return user_cb(d);
          }
          this->resize = false;
          return 0;
        };
        in.mouse_button_cb = [this, user_cb = p.mouse_button_cb](const auto& d) -> int {
          if (d.button != fan::mouse_left) {
            return 0;
          }
          if (d.button_state != fan::mouse_state::press) {
            this->move = false;
            d.flag->ignore_move_focus_check = false;
            return 0;
          }
          if (d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
            return 0;
          }
          d.flag->ignore_move_focus_check = true;
          this->move = true;
          this->click_offset = fan::vec2(get_position()) - d.position;
          gloco->vfi.set_focus_keyboard(d.vfi->focus.mouse);
          return user_cb(d);
        };
        in.mouse_move_cb = [this, user_cb = p.mouse_move_cb](const auto& d) -> int {
          if (this->resize && this->move) {
            fan::vec2 new_size = (d.position - fan::vec2(get_position()));
            static constexpr fan::vec2 min_size(10, 10);
            new_size.constrain(min_size);
            this->set_size(new_size.x);
            return user_cb(d);
          }
          else if (this->move) {
            fan::vec3 p = get_position();
            this->set_position(fan::vec3(d.position + click_offset, p.z));
            return user_cb(d);
          }
          return 0;
        };
        vfi_root = in;
      }
      void push_child(const loco_t::shape_t& shape) {
        children.push_back(shape);
      }
      fan::vec3 get_position() {
        return vfi_root.get_position();
      }
      void set_position(const fan::vec3& position) {
        fan::vec2 root_pos = vfi_root.get_position();
        fan::vec2 offset = fan::vec2(position) - root_pos;
        vfi_root.set_position(fan::vec3(root_pos + offset, position.z));
        for (auto& child : children) {
          child.set_position(fan::vec3(fan::vec2(child.get_position()) + offset, position.z));
        }
      }
      fan::vec2 get_size() {
        return vfi_root.get_size();
      }
      void set_size(const fan::vec2& size) {
        fan::vec2 root_pos = vfi_root.get_size();
        fan::vec2 offset = size - root_pos;
        vfi_root.set_size(root_pos + offset);
        for (auto& child : children) {
          child.set_size(fan::vec2(child.get_size()) + offset);
      }
    }
    fan::vec2 click_offset = 0;
    bool move = false;
    bool resize = false;
    loco_t::shape_t vfi_root;
    std::vector<loco_t::shape_t> children;
    };
  }
}

// for pch
#if defined(fan_build_pch)
inline void fan::opengl::viewport_t::set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  viewport_position = viewport_position_;
  viewport_size = viewport_size_;

  gloco->get_context()->opengl.call(
    gloco->get_context()->opengl.glViewport,
    viewport_position.x, window_size.y - viewport_size.y - viewport_position.y,
    viewport_size.x, viewport_size.y
  );
}

inline loco_t::image_list_NodeReference_t::image_list_NodeReference_t(loco_t::image_t* image) {
  NRI = image->texture_reference.NRI;
}

inline loco_t::camera_list_NodeReference_t::camera_list_NodeReference_t(loco_t::camera_t* camera) {
  NRI = camera->camera_reference.NRI;
}

namespace fan::opengl {
  // Primary template for the constructor
  inline theme_list_NodeReference_t::theme_list_NodeReference_t(void* theme) {
    //static_assert(std::is_same_v<decltype(theme), loco_t::theme_t*>, "invalid parameter passed to theme");
    NRI = ((loco_t::theme_t*)theme)->theme_reference.NRI;
  }
}

inline fan::opengl::viewport_list_NodeReference_t::viewport_list_NodeReference_t(fan::opengl::viewport_t* viewport) {
  NRI = viewport->viewport_reference.NRI;
}
#endif

#if defined(loco_imgui) && defined(fan_platform_linux)
static void imgui_xorg_init() {
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = gloco->get_window()->get_size();
  gloco->get_window()->add_mouse_move_callback([](const auto& d) {
    auto& io = ImGui::GetIO();
    if (!io.WantSetMousePos) {
      io.AddMousePosEvent(d.position.x, d.position.y);
    }
  });
}
static void imgui_xorg_new_frame() {
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = gloco->get_window()->get_size();
}
#endif

#include _FAN_PATH(graphics/collider.h)