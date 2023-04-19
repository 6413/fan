#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define fan_unit_test

struct idlist_output_t {
  struct {
    uint8_t blending;
    uint16_t depth;
    uint8_t shape_type;
    void* shape_key;
  }key;
  uint8_t NeedToBeExist : 1, TraverseFound : 1;
};

std::unordered_map<uint32_t, idlist_output_t> idlist;

#define loco_window
#define loco_context
#define loco_rectangle
#define loco_circle
#define loco_line
#define loco_button
#define loco_sprite
#define loco_menu_maker_button
#define loco_menu_maker_text_box
#include _FAN_PATH(graphics/loco.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    fan::vec2 window_size = loco.get_window()->get_size();
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      viewport.set(loco.get_context(), 0, d.size, d.size);
      });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
};

pile_t* pile = new pile_t;

fan_has_variable_struct(image);
fan_has_variable_struct(theme);


fan_has_variable_struct(src);
fan_has_variable_struct(dst);

fan_has_variable_struct(position);
fan_has_variable_struct(size);
fan_has_variable_struct(color);
fan_has_variable_struct(text);

loco_t::theme_t theme(pile->loco.get_context(), loco_t::themes::deep_red());

void push_things() {
  while (idlist.size() < 1) {
     pile->loco.types.iterate([&]<typename T>(const auto & i, const T & o) {
      using type_t = std::remove_pointer_t<std::remove_pointer_t<T>>;
      if constexpr (!std::is_same_v<type_t, loco_t::comma_dummy_t>) {

        idlist_output_t output;

        typename type_t::properties_t p;
        p.camera = &pile->camera;
        p.viewport = &pile->viewport;

        if constexpr (has_image_v<decltype(p)>) {
          p.image = &pile->loco.default_texture;
        }
        if constexpr (has_theme_v<decltype(p)>) {
          p.theme = &theme;
        }

        if constexpr (has_src_v<decltype(p)>) {
          p.src = fan::random::vec2(-1, 1);
          output.key.depth = p.src.z;
        }
        if constexpr (has_dst_v<decltype(p)>) {
          p.dst = fan::random::vec2(-1, 1);
          output.key.depth = p.dst.z;
        }
        if constexpr (has_position_v<decltype(p)>) {
          p.position = fan::random::vec2(-1, 1);
          output.key.depth = p.position.z;
        }
        if constexpr (has_size_v<decltype(p)>) {
          p.size = 0.1;
        }
        if constexpr (has_color_v<decltype(p)>) {
          p.color = fan::random::color();
        }
        if constexpr (has_text_v<decltype(p)>) {
          p.text = fan::random::string(fan::random::value_i64(0, 10));
        }
        p.blending = true;

        loco_t::id_t* id = new loco_t::id_t(p);
        uint32_t key = id->cid.NRI;
        //std::construct_at((loco_t::id_t*)&key, std::move(id));

        if (idlist.find(key) != idlist.end()) {
          fan::throw_error(__LINE__);
        }
        output.key.blending = p.blending;
        output.key.shape_key = new uint8_t[sizeof(p.key)];
        pile->loco.shape_get_properties(*id, [&](const auto& properties) {
          memcpy(output.key.shape_key, &properties.key, sizeof(p.key));
        });
        output.key.shape_type = type_t::shape_type;
        output.NeedToBeExist = true;
        output.TraverseFound = false;
        idlist.insert(std::make_pair(key, output));
        fan::print(idlist.size());
      }
    });
  }
}

void ResetInfo(){
  //for(auto& i : idlist){
  //  i.second.TraverseFound = false;
  //}
}

void QueryAll() {
  fan::print(idlist.size());
  for(auto& i : idlist){
    loco_t::id_t* id = (loco_t::id_t *) & i.first;
    //id.cid.NRI = i.first;
    //auto* id = ()i.first;
    if (id->get_blending() != i.second.key.blending) {
      fan::throw_error(__LINE__);
    }
    if (id->get_position().z != i.second.key.depth) {
      fan::throw_error(__LINE__);
    }
    if (pile->loco.cid_list[id->cid].cid.shape_type != i.second.key.shape_type) {
      fan::throw_error(__LINE__);
    }
    pile->loco.shape_get_properties(*id, [&](const auto& properties) {
      if (std::memcmp(&properties.key, i.second.key.shape_key, sizeof(properties.key)) != 0) {
        fan::throw_error(__LINE__);
      }
    });
  }
}

void CheckAll(){
  pile->loco.process_loop([] {});
  QueryAll();

  //for(auto& i : idlist){
  //  if(i.second.NeedToBeExist == true && i.second.TraverseFound == false){
  //    fan::throw_error(__LINE__);
  //  }
  //}

  ResetInfo();
}

int main() {


  //// pushback
  push_things();
  CheckAll();


  //// erase
  //for (uint32_t i = 0; i < ids.size(); ++i) {
  //  auto idx = rand() % ids.size();
  //  ids[idx].erase();
  //  ids.erase(ids.begin() + idx);
  //  if (!ids.empty()) {
  //    // put test functions here
  //    ids[rand() % ids.size()].set_depth(rand() % 0xffff);
  //  }
  //}
}