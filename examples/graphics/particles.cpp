#include fan_pch

struct particle_base_t : loco_t::shape_t {

  using shape_t::shape_t;

  struct properties_t {
    loco_t::shape_t shape;
    uint64_t alive = 5e+9;
    fan::vec2 velocity = {1, 1};
  };

  particle_base_t(const properties_t& p) : loco_t::shape_t(p.shape) {
    keep_alive_timer.start(fan::time::nanoseconds(p.alive));
    velocity = p.velocity;
  }

  bool is_dead() const {
    return keep_alive_timer.finished();
  }

  fan::function_t<void(particle_base_t*, f32_t dt)> update;
  fan::vec2 velocity;
  fan::time::clock keep_alive_timer;
};

struct smoke_t : particle_base_t{
  struct properties_t {
    fan::vec3 position;
    fan::vec2 size;
    loco_t::image_t* image;
    f32_t begin_angle = 0;
    f32_t end_angle = fan::math::pi * 2;
    fan::color color = fan::color(1, 1, 1, 1);
  };
  smoke_t(const properties_t& p) : particle_base_t(particle_base_t::properties_t{
      .shape =
      fan_init_struct(
        loco_t::sprite_t::properties_t,
        .position = p.position,
        .size = p.size,
        .image = p.image,
        .blending = true
      ),
        .alive = (uint64_t)5e+9,
        .velocity = fan::random::vec2_direction(p.begin_angle, p.end_angle) * 100
    }) {
    update = [color = p.color] (particle_base_t* ptr, f32_t dt) {
      fan::vec2 position = ptr->get_position();
      fan::vec2 size = ptr->get_size();
      f32_t angle = ptr->get_angle();
      fan::color c = ptr->get_color();
      ptr->set_position(position + ptr->velocity * dt);
      ptr->set_size(size + dt * 100);
      ptr->set_angle(angle + dt / 4);
      c.r = color.r;
      c.g = color.g;
      c.b = color.b;
      c.a -= dt;
      ptr->set_color(c);
    };
  }

};

struct particle_system_t {

  void add(const auto& particle) {
    auto it = particle_list.NewNodeLast();
    auto& node = particle_list[it];
     node = particle;
  }

  void update() {
    auto it = particle_list.GetNodeFirst();
    while (it != particle_list.dst) {
      particle_list.StartSafeNext(it);
      auto& node = particle_list[it];
      if (node.is_dead()) {
        particle_list.unlrec(it);
        it = particle_list.EndSafeNext();
        continue;
      }
      node.update(&node, gloco->get_delta_time());
      it = particle_list.EndSafeNext();
    }
  }

protected:
  #define BLL_set_StoreFormat 1
  #define BLL_set_SafeNext 1
  //#define BLL_set_CPP_CopyAtPointerChange
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix particle_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType particle_base_t
  #define BLL_set_Link 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  particle_list_t particle_list;
};

int main() {
  loco_t loco;

  fan::vec2 viewport_size = loco.get_window()->get_size();
  loco.default_camera->camera.set_ortho(
    fan::vec2(0, viewport_size.x),
    fan::vec2(0, viewport_size.y)
  );

  bool click = false;

  loco.get_window()->add_buttons_callback([&](const auto& d) {
    if (d.button != fan::mouse_left) {
      return;
    }
    if (d.state != fan::mouse_state::press) {
      click = false;
      return;
    }
    click = true;
  });

  f32_t z = 0;

  particle_system_t ps;

  loco_t::image_t smoke_texture{"images/smoke.webp"};

  loco.loop([&] {
    if (click) {
      ps.add(smoke_t({
        .position = fan::vec3(gloco->get_mouse_position(), z),
        .size = 50,
        .image = &smoke_texture,
        .color = fan::color::rgb(100, 100, 1000)
      }));
      z = fmodf(z + 1, 5000);
    }
    ps.update();
  });
}