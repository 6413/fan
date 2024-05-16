#include <fan/pch.h>
// todo add this to pch if needed
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


struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    fan::vec2 window_size = loco.window.get_size();
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.window.add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      viewport.set(0, d.size, d.size);
      });
    viewport.open();
    viewport.set(0, window_size, window_size);
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

fan_has_variable_struct(letter_id);

loco_t::theme_t theme{ loco_t::themes::deep_red() };

void push_things() {
  while (idlist.size() < 100) {
     pile->loco.types.iterate([&]<typename T>(const auto & i, const T & o) {
      using type_t = std::remove_pointer_t<std::remove_pointer_t<T>>;
      if constexpr (
        !std::is_same_v<type_t, loco_t::comma_dummy_t> &&
        !std::is_same_v<type_t, loco_t::vfi_t> //&&
        //!std::is_same_v<type_t, loco_t::responsive_text_t> &&
        //!std::is_same_v<type_t, loco_t::button_t>
        ) {

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
          p.src.z = rand() % 0xfffe + 1;//fan::random::value_i64(1, 0xffff);
          output.key.depth = p.src.z;
        }
        if constexpr (has_dst_v<decltype(p)>) {
          p.dst = fan::random::vec2(-1, 1);
        }
        if constexpr (has_position_v<decltype(p)>) {
          p.position = fan::random::vec2(-1, 1);
          p.position.z = rand() % 0xfffe + 1;//fan::random::value_i64(1, 0xffff);
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
      #if defined(loco_letter)
        if constexpr (has_letter_id_v<decltype(p)>) {
          auto it = gloco->font.info.characters.begin();
          std::advance(it, fan::random::value_i64(0, gloco->font.info.characters.size() - 1));
          p.letter_id = it->first;
        }
      #endif
        p.blending = true;
        if (idlist.size() != 0) {
          uint32_t nri = 0;
          fan::print("salsas0", ((loco_t::shape_t*)&nri)->get_position().z);
        }
        loco_t::shape_t* id = new loco_t::shape_t(p);
        if (idlist.size() != 0) {
          uint32_t nri = 0;
          fan::print("salsas1", ((loco_t::shape_t*)&nri)->get_position().z);
        }
        uint32_t key = id->NRI;
        //std::construct_at((loco_t::shape_t*)&key, std::move(id));

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
  for(auto& i : idlist){
    loco_t::shape_t* id = (loco_t::shape_t *)&i.first;
    //id.cid.NRI = i.first;
    //auto* id = ()i.first;
    if (id->get_blending() != i.second.key.blending) {
      fan::throw_error(__LINE__);
    }
    f32_t p = id->get_position().z;
    if (p != i.second.key.depth) {
      fan::throw_error(__LINE__);
    }
    if ((*id)->shape_type != i.second.key.shape_type) {
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

  //auto it = idlist.end();
  //while (idlist.size()) {
  //  idlist.erase(it);
  //  std::advance(it, -1);
  //}

  //// erase
  for (uint32_t i = 0; i < idlist.size(); ++i) {
    auto it = idlist.begin();
    loco_t::shape_t* id = (loco_t::shape_t*)&it->first;
    fan::print((uint16_t)(*id)->shape_type);
    id->set_depth(rand() % 0xffff);
    id->erase();
    idlist.erase(it);
    //std::advance(it, 1/*fan::random::value_i64(0, idlist.size() - 1)*/);
    /*if (!idlist.empty()) {
      auto it2 = idlist.begin();
      std::advance(it2, fan::random::value_i64(0, idlist.size() - 1));
      ((loco_t::shape_t*)&it2->first)->set_depth(rand() % 0xffff);
    }*/
  }

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