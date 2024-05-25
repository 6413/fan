#include <fan/pch.h>

enum class custom_shapes {
  hexagon = loco_t::shape_type_t::last,
  triangle
};

struct hexagon_t {

  static constexpr uint16_t shape_type = (uint16_t)custom_shapes::hexagon;
  static constexpr int kpi = loco_t::kp::common;

#pragma pack(push, 1)

  struct KeyPack_t {
    loco_t::blending_t blending;
    loco_t::depth_t depth;
    loco_t::viewport_t viewport;
    loco_t::camera_t camera;
    shaper_t::ShapeTypeIndex_t ShapeType;
  };

#pragma pack(pop)

  struct vi_t {
    fan::vec3 position;
  };
  struct ri_t {

  };
  struct properties_t {
    fan::vec3 position;

    bool blending = false;

    loco_t::camera_t camera = gloco->orthographic_camera.camera;
    loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
  };

  inline static std::vector<shaper_t::ShapeType_t::init_t> locations = {
  shaper_t::ShapeType_t::init_t{0, sizeof(vi_t::position) / sizeof(f32_t), fan::opengl::GL_FLOAT, sizeof(vi_t), (void*)offsetof(vi_t, position)}
  };

  loco_t::shape_t push_back(const properties_t& properties) {
    KeyPack_t KeyPack;
    KeyPack.ShapeType = shape_type;
    KeyPack.depth = properties.position.z;
    KeyPack.blending = properties.blending;
    KeyPack.camera = properties.camera;
    KeyPack.viewport = properties.viewport;
    //KeyPack.ShapeType = shape_type;
    vi_t vi;
    vi.position = properties.position;
    ri_t ri;

    return gloco->shaper.add(KeyPack.ShapeType, &KeyPack, &vi, &ri);
  }

}hexagon;


struct custom_t : loco_t::shape_t {

  template <typename T>
  custom_t(const T& properties) {
    if constexpr (std::is_same_v<T, hexagon_t::properties_t>) {
      *dynamic_cast<loco_t::shape_t*>(this) = hexagon.push_back(properties);
    }
  }

};

int main() {
  loco_t loco;

  gloco->shape_open<hexagon_t>(
    &hexagon,
    "shaders/hexagon.vs",
    "shaders/hexagon.fs"
  );

  custom_t c = hexagon_t::properties_t{.position=fan::vec3(1, 2, 3)};

  loco.loop([&] {

  });
}