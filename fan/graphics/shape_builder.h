#ifndef sb_depth_var
#define sb_depth_var position
#endif

#ifndef sb_vertex_count
#define sb_vertex_count 6
#endif

#ifndef sb_has_own_key_root
#define sb_has_own_key_root 0
#endif
#ifndef sb_ignore_3_key
#define sb_ignore_3_key 0
#endif

#if sb_has_own_key_root == 1
loco_bdbt_NodeReference_t key_root;
#endif

using key_t = fan::masterpiece_t <
  #if sb_ignore_3_key == 0
  loco_t::redraw_key_t,
  uint16_t,
  loco_t::shape_type_t,
  #endif
  context_key_t
>;
static constexpr bool key_equality_assert = fan::assert_equality_v<sizeof(key_t), (
  sizeof(context_key_t) +
  sizeof(loco_t::shape_type_t) * (sb_ignore_3_key == 0) +
  sizeof(uint16_t) * (sb_ignore_3_key == 0) +
  sizeof(loco_t::redraw_key_t) * (sb_ignore_3_key == 0)
  )>;

using push_key_t = fan::masterpiece_t <
  #if sb_ignore_3_key == 0
  loco_t::make_push_key_t<loco_t::redraw_key_t>,
  loco_t::make_push_key_t<uint16_t, true>,
  loco_t::make_push_key_t<loco_t::shape_type_t>,
  #endif
  loco_t::make_push_key_t<context_key_t>
>;

struct block_element_t {
  key_t key;
  uint8_t vi[sizeof(vi_t)];
  ri_t ri;
};

#if defined(loco_opengl)
	#include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)
#elif defined(loco_vulkan)
	#include _FAN_PATH(graphics/vulkan/2D/objects/shape_builder.h)
#endif

constexpr void iterate_keys(uint16_t max_depth, uint16_t depth, auto& key, auto& bm_id, auto old_o) {
  if (depth == max_depth) {
    return;
  }
  if (depth != max_depth - 1) {
    auto o = loco_bdbt_NewNode(&gloco->bdbt);
    key.get_value(depth, [&](const auto& data) {
      data->k.a(&gloco->bdbt, &data->data, 0, old_o, o);
      });
    iterate_keys(max_depth, depth + 1, key, bm_id, o);
  }
  else {
    bm_id = push_new_bm(key);
    key.get_value(depth, [&](const auto& data) {
      data->k.a(&gloco->bdbt, &data->data, 0, old_o, bm_id.NRI);
      });
  }
};

void sb_erase_key_from(loco_t::cid_nt_t& id) {
  auto bm_id = *(shape_bm_NodeReference_t*)&id->bm_id;
  auto bm_node = bm_list.GetNodeByReference(bm_id);

  loco_bdbt_NodeReference_t key_nr =
    #if sb_has_own_key_root == 1
    key_root
    #else
    gloco->root
    #endif
    ;

  fan::masterpiece_t <
    #if sb_ignore_3_key == 0
    loco_t::make_erase_key_t<loco_t::redraw_key_t>,
    loco_t::make_erase_key_t<uint16_t, true>,
    loco_t::make_erase_key_t<loco_t::shape_type_t>,
    #endif
    loco_t::make_erase_key_t<context_key_t::key_t>
  > key{
  #if sb_ignore_3_key == 0
    loco_t::make_erase_key_t<loco_t::redraw_key_t>{.data = {.blending = sb_get_ri(id).blending}},
    loco_t::make_erase_key_t<uint16_t, true>{.data = (uint16_t)sb_get_vi(id).sb_depth_var.z},
    {.data = shape_type },
  #endif
    {.data = bm_node->data.key.get_value<context_key_t>()->key}
  };

  key.iterate([&]<typename T>(const auto & i, const T & data) {
    data->key_nr = key_nr;
    typename std::remove_pointer_t<T>::key_t k;
    k.q(&gloco->bdbt, &data->data, &data->key_size, &key_nr);
    #if fan_debug >= 2
    if (data->key_size != sizeof(data->data) * 8) {
      __abort();
    }
    #endif
  });

  key.reverse_iterate_ret([&]<typename T>(auto i, const T & data) -> int {
    typename std::remove_pointer_t<T>::key_t k;
    k.r(&gloco->bdbt, &data->data, data->key_nr);

    if (loco_bdbt_inrhc(&gloco->bdbt, data->key_nr) == true) {
      return 1;
    }
    if constexpr (i.value != 0) { // if its not last to iterate
      loco_bdbt_Recycle(&gloco->bdbt, data->key_nr);
    }
    return 0;
  });
}

void sb_erase(loco_t::cid_nt_t& id) {
  suck_block_element(id);

  #if fan_debug >= 2
  id.sic();
  #endif
}

template <typename T, typename T2>
auto get(loco_t::cid_nt_t& id, T T2::* member) {
  return sb_get_vi(id).*member;
}

key_t& get_bm_key(loco_t::cid_nt_t& id) {
  auto bm_id = *(shape_bm_NodeReference_t*)&id->bm_id;
  auto bm_node = bm_list.GetNodeByReference(bm_id);
  return bm_node->data.key;
}

context_key_t& get_context_key(loco_t::cid_nt_t& id) {
  return *get_bm_key(id).get_value<context_key_t>();
}

template <typename T = void>
loco_t::camera_t* get_camera(loco_t::cid_nt_t& id) requires fan::has_camera_t<properties_t> {
  return gloco->camera_list[*get_context_key(id).key.get_value<loco_t::camera_list_NodeReference_t>()].camera_id;
}
template <typename T = void>
void set_camera(loco_t::cid_nt_t& id, loco_t::camera_t* camera) requires fan::has_camera_t<properties_t> {
  sb_set_context_key<loco_t::camera_list_NodeReference_t>(id, camera);
}

template <typename T = void>
loco_t::viewport_t* get_viewport(loco_t::cid_nt_t& id) requires fan::has_viewport_t<properties_t> {
  return gloco->get_context().viewport_list[*get_context_key(id).key.get_value<fan::graphics::viewport_list_NodeReference_t>()].viewport_id;
}
template <typename T = void>
void set_viewport(loco_t::cid_nt_t& id, loco_t::viewport_t* viewport) requires fan::has_viewport_t<properties_t> {
  sb_set_context_key<fan::graphics::viewport_list_NodeReference_t>(id, viewport);
}

template <typename T = void>
loco_t::image_t* get_image(loco_t::cid_nt_t& id) requires fan::has_image_t<properties_t> {
  properties_t p;
  loco_t::image_t* ptr = nullptr;
  [&id, &ptr] <typename T2>(T2 & p, auto * This) mutable {
    if constexpr (fan::has_image_t<T2>) {
      auto nr = This->get_context_key(id).key.template get_value<loco_t::textureid_t<0>>();
      if constexpr (std::is_same_v< std::remove_reference_t<decltype(*nr)>, loco_t::textureid_t<0>>) {
        ptr = gloco->image_list[*(loco_t::textureid_t<0>*)nr].image;
      }
    }
  }(p, this);
  return ptr;
}

properties_t sb_get_properties(loco_t::cid_nt_t& id) {
  properties_t p;
  *(context_key_t*)&p = get_context_key(id);
  *(vi_t*)&p = sb_get_vi(id);
  *(ri_t*)&p = sb_get_ri(id);

  [&id] <typename T>(T & p, auto * This) {
    if constexpr (fan::has_camera_t<T>) {
      p.camera = This->get_camera(id);
    }
  }(p, this);

  [&id] <typename T>(T & p, auto * This) {
    if constexpr (fan::has_viewport_t<T>) {
      p.viewport = This->get_viewport(id);
    }
  }(p, this);

  [&id] <typename T>(T & p, auto * This) {
    if constexpr (fan::has_image_t<T>) {
      p.image = This->get_image(id);
    }
  }(p, this);
  return p;
}

template <typename T>
void sb_set_context_key(loco_t::cid_nt_t& id, auto value) {
  block_element_t block_element;
  suck_block_element(id, &block_element);
  *block_element.key.get_value<context_key_t>()->key.get_value<T>() = value;
  unsuck_block_element(id, block_element);
}

void sb_set_depth(loco_t::cid_nt_t& id, f32_t depth) {
  #if sb_ignore_3_key == 0
  block_element_t block_element;
  suck_block_element(id, &block_element);
  ((vi_t*)block_element.vi)->sb_depth_var.z = depth;
  *block_element.key.get_value<1>() = (uint16_t)depth;
  unsuck_block_element(id, block_element);
  #endif
}

#undef sb_has_own_key_root
#undef sb_ignore_3_key

#undef sb_vertex_count
#undef sb_depth_var