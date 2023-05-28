// LIMITIATIONS
// CID VECTOR IS PREALLOCATED, SO IT CANT BE MODIFIED OTHERWISE
// CIDS WILL INVALIDATE BECAUSE POINTERS CHANGE IN VECTOR RESIZE

struct sheet_t {

  void push_back(loco_t::textureid_t<0> image) {
    m_textures.push_back(image);
  }

  uint32_t size() const {
    return m_textures.size();
  }

  uint32_t start_index = 0;
  uint64_t animation_speed = 1e+9;
  std::vector<loco_t::textureid_t<0>> m_textures;
};

struct sb_sprite_sheet_name {

  loco_t* get_loco() {
    return OFFSETLESS(this, loco_t, sb_shape_var_name);
  }

  #define make_key_value(type, name) \
    type& name = *key.get_value<decltype(key)::get_index_with_type<type>()>();

  struct properties_t : loco_t::sprite_t::context_key_t {
    fan::vec2 position;
    fan::vec2 size;

    make_key_value(loco_t::camera_list_NodeReference_t, camera);
    make_key_value(fan::graphics::viewport_list_NodeReference_t, viewport);

    sheet_t* sheet;
  };

  #undef make_key_value
protected:
  #include "sprite_sheet_list_builder_settings.h"
  #include _FAN_PATH(BLL/BLL.h)
public:

  sheet_list_t sheet_list;

  using nr_t = sheet_list_NodeReference_t;

  nr_t push_back(const properties_t& p) {
    auto loco = get_loco();

    if (p.sheet->size() == 0) {
      fan::throw_error("empty sheet");
    }

    auto nr = sheet_list.NewNodeLast();
    auto& node = sheet_list[nr];
    node.sheet = *p.sheet;
    sprite_t::properties_t sp;
    sp.position = p.position;
    sp.size = p.size;

    sp.camera = p.camera;
    sp.viewport = p.viewport;

    sp.image = p.sheet->m_textures[0];
    //sp.
    loco->sprite.push_back(&node.cid, sp);
    return nr;
  }

  void start(nr_t n) {
    auto loco = get_loco();
    auto& node = sheet_list[n];
    node.timer.cb = [n_ = n, this, loco] (const fan::ev_timer_t::cb_data_t& timer) {
      auto& node = sheet_list[n_];
      node.sheet.start_index = (node.sheet.start_index + 1) % node.sheet.size();
      loco->sprite.set_image(&node.cid, node.sheet.m_textures[node.sheet.start_index]);
      loco->ev_timer.start(&node.timer, node.sheet.animation_speed);
    };
    loco->ev_timer.start(&node.timer, node.sheet.animation_speed);
  }
  void stop(nr_t n) {
    auto loco = get_loco();
    loco->ev_timer.stop(&sheet_list[n].timer);
  }
};