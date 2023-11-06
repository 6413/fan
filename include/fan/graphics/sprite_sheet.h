struct sheet_t {

  void push_back(loco_t::image_t* image) {
    m_textures.push_back(image);
  }

  uint32_t size() const {
    return m_textures.size();
  }

  uint32_t start_index = 0;
  uint64_t animation_speed = 1e+9;
  std::vector<loco_t::image_t*> m_textures;
};

struct sb_sprite_sheet_name {

  struct properties_t : loco_t::sprite_t::context_key_t {
    fan::vec2 position;
    fan::vec2 size;

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
    if (p.sheet->size() == 0) {
      fan::throw_error("empty sheet");
    }

    auto nr = sheet_list.NewNodeLast();
    auto& node = sheet_list[nr];
    node.sheet = *p.sheet;
    loco_t::sprite_t::properties_t sp;
    sp.position = p.position;
    sp.size = p.size;
    sp.image = p.sheet->m_textures[0];
    node.shape = sp;
    return nr;
  }

  void start(nr_t n) {
    auto& node = sheet_list[n];
    node.timer.cb = [n_ = n, this] (const fan::ev_timer_t::cb_data_t& timer) {
      auto& node = sheet_list[n_];
      node.sheet.start_index = (node.sheet.start_index + 1) % node.sheet.size();
      node.shape.set_image(node.sheet.m_textures[node.sheet.start_index]);
      gloco->ev_timer.start(&node.timer, node.sheet.animation_speed);
    };
    gloco->ev_timer.start(&node.timer, node.sheet.animation_speed);
  }
  void stop(nr_t n) {
    gloco->ev_timer.stop(&sheet_list[n].timer);
  }
};