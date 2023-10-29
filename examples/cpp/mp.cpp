#include fan_pch

struct temp_t {
  int x;
};

struct a_t {
  int x;
  float y;
  temp_t c;
};

struct b_t {
  a_t a;
  int* b;
  int c[2];
};

struct recursive0_t {
  b_t b;
};

struct recursive1_t {
  recursive0_t r;
};

struct recursive2_t {
  recursive1_t r;
};

struct iterate_a_t {
  int x;
};

struct iterate_b_t {
  float y;
};

struct iterate_c_t {
  iterate_a_t a;
  iterate_b_t b;
};


  struct sprite_t {

    //static constexpr typename loco_t::shape_type_t::_t shape_type = loco_t::shape_type_t::sprite;

    //fan::vec3 position;
    fan::vec2 size;
  //  fan::color color;
   // fan::string image_name;

    // global
    fan::string id;
    uint32_t group_id;

  /*  void init(loco_t::shape_t& shape) {
      position = shape.get_position();
      size = shape.get_size();
      color = shape.get_color();
    }

    void from_string(const fan::string& str, uint64_t& off) {
      position = fan::read_data<fan::vec3>(str, off);
      size = fan::read_data<fan::vec3>(str, off);
      color = fan::read_data<fan::color>(str, off);
      position = fan::read_data<fan::vec3>(str, off);
    }*/

   /* shapes_t::global_t* get_shape(fgm_t* fgm) {
      fgm_t::shapes_t::global_t* ret = new fgm_t::shapes_t::global_t(
        fgm,
       fan::graphics::sprite_t{{
           .position = position,
           .size = size,
           .color = color
         }}
      );

      ret->shape_data.sprite.image_name = image_name;
      ret->id = id;
      ret->group_id = group_id;

      if (image_name.empty()) {
        return ret;
      }

      loco_t::texturepack_t::ti_t ti;
      if (fgm->texturepack.qti(image_name, &ti)) {
        fan::print_no_space("failed to load texture:", image_name);
      }
      else {
        auto& data = fgm->texturepack.get_pixel_data(ti.pack_id);
        gloco->sprite.load_tp(ret->children[0], &ti);
      }

      return ret;
    }*/
  };
//
//  struct shapes_t {
//    sprite_t sprite;
//  };
//};


template <typename T>
constexpr auto make_struct_tuple0(const T& st) {
  typename T::type_t t;
  return fan::generate_variable_list_nref<fan::count_struct_members<typename T::type_t>()>(t);
}


int main() {
  ////
  //{
  //  fan::mp_t<a_t> mp_a{1, 2.2};
  //  mp_a.iterate([]<auto i>(auto & v) {      
  //   // fan::print("before addition", v);
  // //   v += 10;
  //   // fan::print("after addition", v);
  //  });
  //  //std::cout << mp_a;
  //  fan::print(mp_a);
  //  //fan::print()
  //}
  //fan::print("\n");
  //{
  //  fan::mp_t<b_t> mp_b{a_t{1, 2.2}, nullptr, {1, 2}};
  //  fan::print(mp_b);
  //}
  //fan::print("\n");
  //{
  //  fan::print(recursive2_t{});
  //}

  //fan::mp_t<sprite_t> mp;
  struct bbb {
    fan::color v;
  };
  fan::print(fan::count_struct_members<bbb>());
  //  fan::print(fan::count_struct_members<sprite_t>());
  //auto [x, y] = mp;
 // make_struct_tuple0(mp);
  //fan::print(

  //fan::count_struct_members<const fan::mp_t<sprite_t>&>();
  //mp.();

  //return mp.size();
 // mp.iterate([]<auto i0, typename T>(T & v0) {
    /*fan::mp_t<T> inner;
    inner.iterate([]<auto i1>(auto & v1) {
      fan::print(v1);
    });*/
 // });

}