struct sb_sprite_sheet_name {

  struct sheet_t {
    uint32_t start_index = 0;
    uint64_t animation_speed = 1e+9;
    uint32_t count = 0;
    loco_t::image_t* images = nullptr;
  };

  static constexpr loco_t::shape_type_t shape_type = loco_t::shape_type_t::sprite_sheet;

  struct properties_t : loco_t::sprite_t::context_key_t, sheet_t {
    using type_t = sb_sprite_sheet_name;

    fan::vec3 position;
    fan::vec2 size;
    bool blending = false;
  };

protected:
  #include "sprite_sheet_list_builder_settings.h"
  #include _FAN_PATH(BLL/BLL.h)
public:

  sheet_list_t sheet_list;

  using nr_t = sheet_list_NodeReference_t;

  nr_t push_back(const properties_t& p) {
    auto nr = sheet_list.NewNodeLast();
    auto& node = sheet_list[nr];
    node.sheet = *dynamic_cast<const sheet_t*>(&p);
    typename loco_t::sprite_t::properties_t sp;
    sp.position = p.position;
    sp.size = p.size;
    sp.blending = p.blending;
    node.shape = sp;
    return nr;
  }

  void start(nr_t n, uint32_t start, uint32_t count) {
    auto& node = sheet_list[n];
    node.sheet.start_index = (node.sheet.start_index + 1) % count + start;
    node.shape.set_image(&node.sheet.images[node.sheet.start_index]);
    gloco->ev_timer.stop(node.timer_id);
    node.timer_id = gloco->ev_timer.start(node.sheet.animation_speed, [n_ = n, this, start, count]() {
      auto& node = sheet_list[n_];
      node.sheet.start_index = (node.sheet.start_index + 1) % count + start;
      node.shape.set_image(&node.sheet.images[node.sheet.start_index]);
    });
  }

  void start(nr_t n) {
    auto& node = sheet_list[n];
    node.sheet.start_index = (node.sheet.start_index + 1) % node.sheet.count;
    node.shape.set_image(&node.sheet.images[node.sheet.start_index]);
    gloco->ev_timer.stop(node.timer_id);
    gloco->ev_timer.start(node.sheet.animation_speed, [n_ = n, this]() {
      auto& node = sheet_list[n_];
      node.sheet.start_index = (node.sheet.start_index + 1) % node.sheet.count;
      node.shape.set_image(&node.sheet.images[node.sheet.start_index]);
    });
  }
  void stop(nr_t n) {
    gloco->ev_timer.stop(sheet_list[n].timer_id);
  }

  fan::vec3 get_position(nr_t n) {
    return sheet_list[n].shape.get_position();
  }
  void set_position(nr_t n, const fan::vec2& position) {
    return sheet_list[n].shape.set_position(position);
  }
  void set_position(nr_t n, const fan::vec3& position) {
    return sheet_list[n].shape.set_position(position);
  }
  void set_image(nr_t nr, loco_t::image_t* image) {
    sheet_list[nr].shape.set_image(image);
  }

  auto& get_sheet_data(nr_t n) {
    return sheet_list[n].sheet;
  }
};