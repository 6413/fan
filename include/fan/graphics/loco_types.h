#if defined(loco_rectangle)
#define loco_rectangle_vi_t \
  fan::vec3 position = 0; \
  f32_t pad[1]; \
  fan::vec2 size = 0; \
  fan::vec2 rotation_point = 0; \
  fan::color color = fan::colors::white; \
  fan::vec3 rotation_vector = fan::vec3(0, 0, 1); \
  f32_t angle = 0;

#define loco_rectangle_bm_properties_t \
  using parsed_masterpiece_t = fan::masterpiece_t< \
    uint16_t, \
    loco_t::matrices_list_NodeReference_t, \
    fan::graphics::viewport_list_NodeReference_t \
  >; \
  struct key_t : parsed_masterpiece_t {}key;

#define loco_rectangle_ri_t \
  loco_t::rectangle_t::cid_t* cid;

#define loco_rectangle_properties_t \
  loco_t::matrices_t* matrices = 0; \
  fan::graphics::viewport_t* viewport = 0;

#endif

#if defined(loco_sprite)
#define loco_sprite_vi_t \
    fan::vec3 position = 0; \
    uint32_t flag = 0; \
    fan::vec2 size = 0; \
    fan::vec2 rotation_point = 0; \
    fan::color color = fan::colors::white; \
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1); \
    f32_t angle = 0; \
    fan::vec2 tc_position = 0; \
    fan::vec2 tc_size = 1;

#define loco_sprite_bm_properties_t \
    using parsed_masterpiece_t = fan::masterpiece_t< \
      uint16_t, \
      loco_t::textureid_t<0>, \
      loco_t::matrices_list_NodeReference_t, \
      fan::graphics::viewport_list_NodeReference_t \
    >; \
    struct key_t : parsed_masterpiece_t {}key;

#define loco_sprite_ri_t \
  loco_t::sprite_t::cid_t* cid;

#define loco_sprite_properties_t \
    loco_t::image_t* image = 0; \
    loco_t::matrices_t* matrices = 0; \
    fan::graphics::viewport_t* viewport = 0; \
 \
    bool load_tp(loco_t::texturepack_t::ti_t* ti) { \
      auto& im = *ti->image; \
      image = &im; \
   \
      tc_position = ti->position / im.size; \
      tc_size = ti->size / im.size; \
 \
      return 0; \
    }

#endif

#if defined(loco_letter)
  #define loco_letter_vi_t \
    fan::vec3 position; \
    f32_t outline_size;\
    fan::vec2 size;\
    fan::vec2 tc_position;\
    fan::color color = fan::colors::white;\
    fan::color outline_color;\
    fan::vec2 tc_size;\
    f32_t pad[2];

  #define loco_letter_bm_properties_t \
      using parsed_masterpiece_t = fan::masterpiece_t< \
        uint16_t, \
        loco_t::matrices_list_NodeReference_t, \
        fan::graphics::viewport_list_NodeReference_t \
      >; \
      struct key_t : parsed_masterpiece_t {}key;

  #define loco_letter_ri_t \
    loco_t::letter_t::cid_t* cid; \
    f32_t font_size; \
    uint32_t letter_id; 

  #define loco_letter_properties_t \
    loco_t::matrices_t* matrices = 0; \
    fan::graphics::viewport_t* viewport = 0;
#endif

#if defined(loco_text)
  #define loco_text_vi_t loco_letter_vi_t
  #define loco_text_ri_t loco_letter_ri_t
  #define loco_text_bm_properties_t loco_letter_bm_properties_t

  #define loco_text_properties_t \
    loco_t::matrices_t* matrices = 0; \
    fan::graphics::viewport_t* viewport = 0; \
    \
    fan::string text;

#endif

#if defined(loco_button)
#define loco_button_vi_t \
    fan::vec3 position = 0; \
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
      loco_t::matrices_list_NodeReference_t,\
      fan::graphics::viewport_list_NodeReference_t\
    >;\
\
    struct key_t : parsed_masterpiece_t {}key;

#define loco_button_ri_t \
  loco_t::button_t::cid_t* cid; \
  uint8_t selected = 0; \
  loco_t::theme_t* theme = 0; \
  fan::graphics::cid_t text_id; \
  loco_t::vfi_t::shape_id_t vfi_id; \
  uint64_t udata;

#define loco_button_properties_t \
 \
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
  loco_t::matrices_t* matrices = 0; \
  fan::graphics::viewport_t* viewport = 0;

#endif

#if defined(loco_text_box)
  #define loco_text_box_vi_t \
    fan::vec3 position = 0; \
    f32_t angle = 0; \
    fan::vec2 size = 0; \
    fan::vec2 rotation_point = 0; \
    fan::color color = fan::colors::white; \
    fan::color outline_color; \
    fan::vec3 rotation_vector = fan::vec3(0, 0, 1); \
    f32_t outline_size;

  #define loco_text_box_bm_properties_t \
    using parsed_masterpiece_t = fan::masterpiece_t< \
      uint16_t, \
      loco_t::matrices_list_NodeReference_t, \
      fan::graphics::viewport_list_NodeReference_t \
    >; \
    struct key_t : parsed_masterpiece_t {}key;

#define loco_text_box_ri_t \
    loco_t::text_box_t::cid_t* cid; \
    uint8_t selected = 0; \
    fan::graphics::theme_list_NodeReference_t theme; \
    loco_t::vfi_t::shape_id_t vfi_id; \
    uint64_t udata; \
 \
    fan::graphics::cid_t text_id; \
    fed_t fed;

#define loco_text_box_properties_t \
    loco_t::matrices_t* matrices = 0; \
    fan::graphics::viewport_t* viewport = 0; \
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
    loco_t::text_cb_t text_cb = [](const loco_t::text_data_t&) -> int { return 0; };

#endif