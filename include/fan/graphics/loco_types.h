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

#endif