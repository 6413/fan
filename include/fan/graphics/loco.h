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

#include _FAN_PATH(types/types.h)


struct loco_t;

#define loco_framebuffer

#include _FAN_PATH(graphics/graphics.h)
#include _FAN_PATH(time/timer.h)
#include _FAN_PATH(font.h)
#include _FAN_PATH(physics/collision/circle.h)
#include _FAN_PATH(io/directory.h)

#include <variant>

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

#if defined(loco_text)
  #define loco_letter
#endif
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
  };

  struct redraw_key_t {
    uint8_t blending;
  };

  struct draw_t {
    uint64_t id;
    std::vector<fan::function_t<void()>> f;
    bool operator<(const draw_t& b) const
    {
      return id < b.id;
    }
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

  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_prefix viewport_resize_callback
	#define BLL_set_NodeData viewport_resize_cb_t data;
	#include _FAN_PATH(window/cb_list_builder_settings.h)
	#include _FAN_PATH(BLL/BLL.h)

  viewport_resize_callback_t m_viewport_resize_callback;

  struct camera_t {

    using resize_cb_data_t = loco_t::viewport_resize_cb_data_t;
    using resize_cb_t = loco_t::viewport_resize_cb_t;
    using resize_callback_NodeReference_t = loco_t::viewport_resize_callback_NodeReference_t;

    resize_callback_NodeReference_t add_resize_callback(resize_cb_t function) {
      auto nr = gloco->m_viewport_resize_callback.NewNodeLast();
      gloco->m_viewport_resize_callback[nr].data = function;
      return nr;
    }

    void remove_resize_callback(resize_callback_NodeReference_t id) {
      gloco->m_viewport_resize_callback.Unlink(id);
      gloco->m_viewport_resize_callback.Recycle(id);
    }

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

    void set_ortho(const fan::vec2& x, const fan::vec2& y) {
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

        resize_cb_data_t cbd;
        cbd.camera = this;
        cbd.position = get_camera_position();
        cbd.size = get_camera_size();
        gloco->m_viewport_resize_callback[it].data(cbd);

        it = it.Next(&gloco->m_viewport_resize_callback);
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
    #define BLL_set_Link 0
    #include _FAN_PATH(BLL/BLL.h)
  public:

  struct cid_nt_t : cid_list_NodeReference_t {
    loco_t::cid_t* operator->() {
      return &gloco->cid_list[*(cid_list_NodeReference_t*)(this)].cid;
    }
    using base_t = cid_list_NodeReference_t;
    void init() {
      *(base_t*)this = gloco->cid_list.NewNode();
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
      gloco->cid_list.Recycle(*this);
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

  #define fan_create_set_dataless_define(name) \
      fan_create_id_definition_define(void, set_##name){ gloco->shape_##set_##name(*this); }

  #define fan_create_set_define_custom(rt, name, custom) \
        fan_create_id_definition_define(void, set_##name, const rt& data){ custom }

  #define fan_create_get_set_define(rt, name) \
    fan_create_id_definition_define(rt, get_##name){ return gloco->shape_##get_##name(*this);} \
    fan_create_set_define(rt, name)

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

    inline shape_t(const shape_t& id) : inherit_t(id) {
      if (id.is_invalid()) {
        return;
      }
      gloco->shape_get_properties(*(shape_t*)&id, [&](const auto& properties) {
        gloco->push_shape(*this, properties);
      });
    }
    inline shape_t(shape_t&& id) : inherit_t(std::move(id)) {

    }
    loco_t::shape_t& operator=(const shape_t& id) {
      if (!is_invalid()) {
        erase();
      }
      if (id.is_invalid()) {
        return *this;
      }
      if (this != &id) {
        gloco->shape_get_properties(*(shape_t*)&id, [&](const auto& properties) {
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

    operator fan::opengl::cid_t *(){
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

    fan_create_set_define(loco_t::camera_list_NodeReference_t, camera);
    fan_create_set_define(fan::graphics::viewport_list_NodeReference_t, viewport);

    fan_create_set_define(wchar_t, append_letter);

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

    lp.internal_format = fan::opengl::GL_RGBA16F;
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
    #if defined(loco_light)
      *types.get_value<light_t*>() = &light;
    #endif
    #if defined(loco_text_box)
      *types.get_value<text_box_t*>() = &text_box;
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
    get_context()->opengl.call(get_context()->opengl.glClearColor, 0, 0, 0, 1);
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

      get_context()->opengl.call(get_context()->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
      //float clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
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

  fan::vec2 get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport) {
    fan::vec2 mouse_pos = get_mouse_position();
    fan::vec2 translated_pos;
    translated_pos.x = fan::math::map(mouse_pos.x, viewport.get_position().x, viewport.get_position().x + viewport.get_size().x, camera.coordinates.left, camera.coordinates.right);
    translated_pos.y = fan::math::map(mouse_pos.y, viewport.get_position().y, viewport.get_position().y + viewport.get_size().y, camera.coordinates.up, camera.coordinates.down);
    return translated_pos;
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

  #if defined(loco_cuda)

  struct cuda_textures_t {

    cuda_textures_t(){
      inited = false;
    }
    ~cuda_textures_t() {
    }
    void close(loco_t* loco, fan::graphics::cid_t *cid) {
      uint8_t image_amount = fan::pixel_format::get_texture_amount(loco->pixel_format_renderer.sb_get_ri(cid).format);
      auto& ri = loco->pixel_format_renderer.sb_get_ri(cid);
      for (uint32_t i = 0; i < image_amount; ++i) {
        wresources[i].close();
        ri.images[i].unload(loco);
      }
    }

    void resize(loco_t* loco, fan::graphics::cid_t* cid, uint8_t format, fan::vec2ui size, uint32_t filter = loco_t::image_t::filter::linear) {
      auto& ri = loco->pixel_format_renderer.sb_get_ri(cid);
      uint8_t image_amount = fan::pixel_format::get_texture_amount(format);
      if (inited == false) {
        // purge cid's images here
        // update cids images
        loco->pixel_format_renderer.reload(cid, format, size, filter);
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

        loco->pixel_format_renderer.reload(cid, format, size, filter);

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

  using update_callback_nr_t = update_callback_NodeReference_t;

  update_callback_t m_update_callback;

  image_t default_texture;

  struct comma_dummy_t {
    uint8_t member_pointer;
    static constexpr typename loco_t::shape_type_t::_t shape_type = -1;
  };

  // requires shape_type create to shape.h, init in constructor, add type_t to properties
  // make get_properties for custom type
  fan::masterpiece_t<
    comma_dummy_t*
    #if defined(loco_rectangle)
    ,rectangle_t*
    #endif
    #if defined(loco_sprite)
    ,sprite_t*
    #endif
    #if defined(loco_unlit_sprite)
    ,unlit_sprite_t*
    #endif
    #if defined(loco_button)
    ,button_t*
    #endif
    #if defined(loco_text)
    , letter_t*
    #endif
    #if defined(loco_text)
    , text_t*
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
    if constexpr(!std::is_same_v<std::nullptr_t, T>){
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

  #define make_global_function_define(func_name, content, ...) \
  fan_has_function_concept(func_name);\
  void shape_ ## func_name(__VA_ARGS__) { \
    types.iterate([&]<typename T>(auto shape_index, T shape) { \
      using shape_t = std::remove_pointer_t<std::remove_pointer_t<T>>; \
      if (shape_t::shape_type == id->shape_type) { \
        content \
      } \
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
    },
    loco_t::cid_nt_t& id
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
          if constexpr(has_##name##_v<typename shape_t::properties_t>) {\
            data = (*shape)->get_instance(id).name; \
          }\
        }\
        else if constexpr (has_get_properties_v<shape_t, loco_t::cid_nt_t&>) { \
          if constexpr(has_##name##_v<typename shape_t::properties_t>) {\
            data = (*shape)->get_properties(id).name; \
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
    } \
    else if constexpr (has_set_v<shape_t, loco_t::cid_nt_t&, decltype(&comma_dummy_t::member_pointer), void*>) { \
      if constexpr(has_##name##_v<typename shape_t::properties_t>) {\
        (*shape)->set(id, &shape_t::vi_t::name, data); \
      }\
    }, \
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

  #define fan_build_get_set_declare(rt, name) \
    fan_build_get_declare(rt, name); \
    fan_build_set_declare(rt, name);

    #define fan_build_get_set_define(rt, name) \
    fan_build_get_define(rt, name); \
    fan_build_set_define(rt, name);

  fan_build_get_set_define(fan::vec3, position);
  fan_build_get_set_define(fan::vec2, size);
  fan_build_get_set_define(fan::color, color);
  fan_build_get_set_define(f32_t, angle);
  fan_build_get_set_define(fan::vec2, rotation_point);

  fan_build_get_set_generic_define(loco_t::textureid_t<0>, image);

  fan_build_get_set_generic_define(f32_t, font_size);
  fan_build_get_set_generic_define(loco_t::camera_list_NodeReference_t, camera);
  fan_build_get_set_generic_define(fan::graphics::viewport_list_NodeReference_t, viewport);

  fan_build_get_set_generic_define(fan::vec2, text_size);
  fan_build_set_generic_dataless_define(focus);
  
  fan_build_get_set_generic_define(fan::string, text);

  fan_build_get_set_define(fan::color, outline_color);
  fan_build_get_set_define(f32_t, outline_size);

  fan_build_set_generic_define(wchar_t, append_letter);


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
    if constexpr (has_draw_v<shape_t, const redraw_key_t &, loco_bdbt_NodeReference_t>) {
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

#if defined(loco_window)
loco_t::image_list_NodeReference_t::image_list_NodeReference_t(loco_t::image_t* image) {
  NRI = image->texture_reference.NRI;
}

loco_t::camera_list_NodeReference_t::camera_list_NodeReference_t(loco_t::camera_t* camera) {
  NRI = camera->camera_reference.NRI;
}

fan::opengl::theme_list_NodeReference_t::theme_list_NodeReference_t(auto* theme) {
  static_assert(std::is_same_v<decltype(theme), loco_t::theme_t*>, "invalid parameter passed to theme");
  NRI = theme->theme_reference.NRI;
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

void fan::opengl::viewport_t::set(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size)  {
  viewport_position = viewport_position_;
  viewport_size = viewport_size_;

  gloco->get_context()->opengl.call(
    gloco->get_context()->opengl.glViewport, 
    viewport_position.x, window_size.y - viewport_size.y - viewport_position.y,
    viewport_size.x, viewport_size.y
  );
}

inline void fan::opengl::viewport_t::set_viewport(const fan::vec2& viewport_position_, const fan::vec2& viewport_size_, const fan::vec2& window_size) {
  gloco->get_context()->opengl.call(
    gloco->get_context()->opengl.glViewport,
    viewport_position_.x,
    window_size.y - viewport_size_.y - viewport_position_.y,
    viewport_size_.x, viewport_size_.y
  );
}