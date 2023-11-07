#if defined(loco_letter)
  #define loco_letter_vi_t \
    loco_t::position3_t position; \
    f32_t outline_size;\
    fan::vec2 size;\
    fan::vec2 tc_position;\
    fan::color color = fan::colors::white;\
    fan::color outline_color;\
    fan::vec2 tc_size;\
    f32_t angle=0; \
    f32_t pad[1];

  #define loco_letter_bm_properties_t \
      using parsed_masterpiece_t = fan::masterpiece_t< \
        loco_t::camera_list_NodeReference_t, \
        fan::graphics::viewport_list_NodeReference_t \
      >; \
      struct key_t : parsed_masterpiece_t {}key;

  #define loco_letter_ri_t \
    f32_t font_size; \
    uint32_t letter_id; \
    bool blending = true;

  #define loco_letter_properties_t \
    loco_t::camera_t* camera = &gloco->default_camera->camera; \
    fan::graphics::viewport_t* viewport = &gloco->default_camera->viewport;
#endif

#if defined(loco_text)
  #define loco_text_vi_t loco_letter_vi_t
  #define loco_text_ri_t loco_letter_ri_t
  #define loco_text_bm_properties_t loco_letter_bm_properties_t

  #define loco_text_properties_t \
    loco_t::camera_t* camera = &gloco->default_camera->camera; \
    fan::graphics::viewport_t* viewport = &gloco->default_camera->viewport; \
    \
    fan::string text;

#endif

#if defined(loco_button)
#define loco_button_vi_t \
    loco_t::position3_t position = 0; \
    f32_t angle = 0; \
    fan::vec2 size = 0; \
    fan::vec2 rotation_point = 0; \
    fan::color color = fan::colors::white; \
    fan::color outline_color; \
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1); \
    f32_t outline_size;

#define loco_button_bm_properties_t \
    using parsed_masterpiece_t = fan::masterpiece_t<\
      uint16_t,\
      loco_t::camera_list_NodeReference_t,\
      fan::graphics::viewport_list_NodeReference_t\
    >;\
\
    struct key_t : parsed_masterpiece_t {}key;

// todo make text_id raw nr to prevent hard copy in ri_t copy
#define loco_button_ri_t \
  uint8_t selected = 0; \
  loco_t::theme_t* theme = 0; \
  loco_t::shape_t text_id; \
  loco_t::vfi_t::shape_id_t vfi_id; \
  uint64_t udata; \
  loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> int { return 0; }; \
  loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> int { return 0; }; \
  loco_t::keyboard_cb_t keyboard_cb = [](const loco_t::keyboard_data_t&) -> int { return 0; }; \
  bool blending = false; \
  fan::vec3 original_position; \
  fan::vec2 original_size;


#define loco_button_properties_t \
 \
  fan::string text; \
  f32_t font_size = 0.1; \
 \
 \
  bool disable_highlight = false; \
 \
  loco_t::camera_t* camera = &gloco->default_camera->camera; \
  fan::graphics::viewport_t* viewport = &gloco->default_camera->viewport;

#endif

#if defined(loco_text_box)
  #define loco_text_box_vi_t \
    loco_t::position3_t position = 0; \
    f32_t angle = 0; \
    fan::vec2 size = 0; \
    fan::vec2 rotation_point = 0; \
    fan::color color = fan::colors::white; \
    fan::color outline_color; \
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1); \
    f32_t outline_size;

  #define loco_text_box_bm_properties_t \
    using parsed_masterpiece_t = fan::masterpiece_t< \
      loco_t::camera_list_NodeReference_t, \
      fan::graphics::viewport_list_NodeReference_t \
    >; \
    struct key_t : parsed_masterpiece_t {}key;

// todo make text_id raw nr to prevent hard copy in ri_t copy
#define loco_text_box_ri_t \
    uint8_t selected = 0; \
    fan::graphics::theme_list_NodeReference_t theme; \
    loco_t::vfi_t::shape_id_t vfi_id; \
    uint64_t udata; \
 \
    loco_t::shape_t text_id; \
    fed_t fed; \
    bool blending = false;  \
    fan::vec3 original_position; \
    fan::vec2 original_size;

#define loco_text_box_properties_t \
    loco_t::camera_t* camera = &gloco->default_camera->camera; \
    fan::graphics::viewport_t* viewport = &gloco->default_camera->viewport; \
    fan::string text; \
    f32_t font_size = 0.1; \
 \
    loco_t::vfi_t::iflags_t vfi_flags; \
 \
    bool disable_highlight = false; \
 \
    loco_t::mouse_button_cb_t mouse_button_cb = [](const loco_t::mouse_button_data_t&) -> int { return 0; }; \
    loco_t::mouse_move_cb_t mouse_move_cb = [](const loco_t::mouse_move_data_t&) -> int { return 0; }; \
    loco_t::keyboard_cb_t keyboard_cb = [](const loco_t::keyboard_data_t&) -> int { return 0; }; \
    loco_t::text_cb_t text_cb = [](const loco_t::text_data_t&) -> int { return 0; }; \
    bool blending = false;

#endif

#if defined(loco_light)
#define loco_light_vi_t \
  loco_t::position3_t position = 0; \
  f32_t parallax_factor = 0; \
  fan::vec2 size = 0; \
  fan::vec2 rotation_point = 0; \
  fan::color color = fan::colors::white; \
  fan::vec3 rotation_vector = fan::vec3(0, 0, 1); \
  f32_t angle = 0;

#define loco_light_bm_properties_t \
  using parsed_masterpiece_t = fan::masterpiece_t< \
    loco_t::camera_list_NodeReference_t, \
    fan::graphics::viewport_list_NodeReference_t \
  >; \
  struct key_t : parsed_masterpiece_t {}key;

#define loco_light_ri_t \
  bool blending = true;

#define loco_light_properties_t \
  loco_t::camera_t* camera = &gloco->default_camera->camera; \
  fan::graphics::viewport_t* viewport = &gloco->default_camera->viewport;

#endif