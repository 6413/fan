#include <fan/pch.h>

#include <fan/graphics/gui/tilemap_editor/renderer0.h>
#include <deque>

static constexpr f32_t tile_size = 32;

f32_t zoom = 2.5;
void init_zoom() {
  auto& window = gloco->window;
  auto update_ortho = [] {
    fan::vec2 s = gloco->window.get_size();
    gloco->camera_set_ortho(
      gloco->orthographic_camera.camera,
      fan::vec2(-s.x, s.x) / zoom,
      fan::vec2(-s.y, s.y) / zoom
    );
  };

  update_ortho();

  window.add_resize_callback([&](const auto& d) {
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    zoom = d.size.y / (f32_t)mode->height * 5;
    update_ortho();
  });
}

void init_bcol() {
  fan::graphics::bcol.PreSolve_Shape_cb = [](
    bcol_t* bcol,
    const bcol_t::ShapeInfoPack_t* sip0,
    const bcol_t::ShapeInfoPack_t* sip1,
    bcol_t::Contact_Shape_t* Contact
    ) {
      // player
      auto* obj0 = bcol->GetObjectExtraData(sip0->ObjectID);
      // wall
      auto* obj1 = bcol->GetObjectExtraData(sip1->ObjectID);
      if (obj1->collider_type == fan::collider::types_e::collider_sensor) {
        bcol->Contact_Shape_DisableContact(Contact);
      }

      switch (obj1->collider_type) {
      case fan::collider::types_e::collider_static:
      case fan::collider::types_e::collider_dynamic: {
        // can access shape by obj0->shape
        break;
      }
      case fan::collider::types_e::collider_hidden: {
        break;
      }
      case fan::collider::types_e::collider_sensor: {
        fte_renderer_t::userdata_t userdata = *(fte_renderer_t::userdata_t*)obj1->userdata;
        if (gloco->window.key_state(userdata.key) == userdata.key_state) {
          static int x = 0;
          fan::print("press", x++);
        }

        //fan::print("sensor triggered");
        break;
      }
      }
    };
}

struct turrets_e {
  enum {
    bug_zapper, // electrocutes nearby insects
    insecticide_sprayer, // sprays poison
    sticky_trap, //slows down insects
  };
};

int turret_price_table[] = {
  100,
  200,
  400
};

struct turret_t {
  int type = turrets_e::bug_zapper;
  int price = turret_price_table[type];
  f32_t range;
  f32_t damage;
  fan::time::clock damage_time;
  std::function<void()> dt_action;
  loco_t::shape_t visual;
};

struct bug_properties_e{
  enum {
    armorless,
    armored
  };
};

struct bug_type_e {
  enum {
    ants,
    beetles,
    bees
  };
};

int bug_price_table[] = {
  10,
  20,
  40
};

f32_t bug_health_table[] = {
  10,
  50,
  100
};

f32_t bug_damage_table[] = {
  10,
  50,
  100
};

struct bug_t {
  int type = bug_type_e::ants;
  f32_t health = bug_health_table[type];
  f32_t speed;
  loco_t::shape_t visual;
  uint32_t current_path = 1;
  int price = bug_price_table[type];
  f32_t damage = bug_damage_table[type];
};

struct gui_t {

  f32_t play_speed = 1.f;
  uint32_t kills = 0;

  int money = 100;
  f32_t health = 100;

  bool playing = false;
  bool moving = false;

  int turret_type = turrets_e::bug_zapper;
  loco_t::shape_t visual_range; // shows the turret's visual range
  loco_t::shape_t visual_turret;

  std::vector<fan::vec2> path;

  std::deque<bug_t> bugs;
  std::vector<turret_t> turrets;

  inline static std::function<void()> bug_zapper_action = [&] {

  };
  inline static std::function<void()> insecticide_sprayer_action = [&] {

  };
  inline static std::function<void()> sticky_trap_action = [&] {

  };

  gui_t() {
    spawn_timer.start(1e+9);
  }

  void render() {

    ImGui::Begin("Shop");
    turret_t turret;
    loco_t::sprite_t::properties_t sp;
    sp.position = fan::vec2(fan::graphics::get_mouse_position(gloco->orthographic_camera));
    sp.position.z = 0xffa0;
    sp.size = tile_size;
    turret.damage_time.start(0.3e+9);
    if (ImGui::Button("Bug Zapper") && money >= turret_price_table[turrets_e::bug_zapper]) {
      turret.type = turrets_e::bug_zapper;
      turret.dt_action = bug_zapper_action;
      turret.range = 128;
      turret.damage = 20;
      turrets.push_back(turret);
      visual_turret = sp;
      moving = true;
      turret_type = turret.type;
    }
    if (ImGui::Button("Insecticide Sprayer") && money >= turret_price_table[turrets_e::insecticide_sprayer]) {
      turret.type = turrets_e::insecticide_sprayer;
      turret.dt_action = insecticide_sprayer_action;
      turret.range = 150;
      turret.damage = 50;
      turrets.push_back(turret);
      visual_turret = sp;
      moving = true;
      turret_type = turret.type;
    }
    if (ImGui::Button("Sticky trap") && money >= turret_price_table[turrets_e::sticky_trap]) {
      turret.type = turrets_e::sticky_trap;
      turret.dt_action = sticky_trap_action;
      turret.range = 150;
      turret.damage = 100;
      turrets.push_back(turret);
      visual_turret = sp;
      moving = true;
      turret_type = turret.type;
    }

    fan::vec3 vr_position = sp.position;
    vr_position.z -= 1;
    if (moving && visual_range.iic()) {
      visual_range = fan::graphics::circle_t{ {
        .position = vr_position,
        .radius = turret.range,
        .color = fan::color(0.8, 0.8, 0.8, 0.5),
        .blending = true
    } };
    }
    fan::vec2 previous_window_size = ImGui::GetWindowSize();
    ImGui::End();

    ImGui::SetNextWindowPos(fan::vec2(0));
    fan::vec2 window_size = gloco->window.get_size();
    ImGui::SetNextWindowSize(fan::vec2(window_size.x - previous_window_size.x, window_size.y));

    if (ImGui::Begin("Controls", 0, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoDecoration | ImGuiDockNodeFlags_AutoHideTabBar)) {
      {
        ImGui::DragFloat("Play speed", &play_speed);
        gloco->delta_time *= play_speed;
      }
      {
        auto str = ("Money:" + std::to_string(money));
        ImGui::Text(str.c_str());
      }
      {
        auto str = ("Kills:" + std::to_string(kills));
        ImGui::Text(str.c_str());
      }
      {
        auto str = ("bugs:" + std::to_string(bugs.size()));
        ImGui::Text(str.c_str());
      }

      ImGui::SetCursorPosY(ImGui::GetWindowSize().y - ImGui::GetFontSize() * 2);
      if (ImGui::Button(">", ImVec2(72, 0))) {
        playing = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("||", ImVec2(72, 0))) {
        playing = false;
      }
      ImGui::End();
    }

    if (moving) {
      fan::vec2 dst = fan::vec2(fan::graphics::get_mouse_position(gloco->orthographic_camera));
      visual_range.set_position(dst);
      visual_turret.set_position(dst);
      if (ImGui::IsMouseClicked(0)) {
        money -= turret_price_table[turret_type];
        turrets.back().visual = visual_turret;
        visual_range.erase();
        moving = false;
      }
      
    }
  }

  void play() {
    if (playing == false) {
      return;
    }

    if (spawn_timer.elapsed() >= spawn_timer.m_time / play_speed) {
      bugs.resize(bugs.size() + 1);
      bugs.back().visual = fan::graphics::sprite_t{ {
          .position = fan::vec3(path[0], 0xfffa0),
          .size = tile_size
      } };
      bugs.back().speed = 1.0f;
      spawn_timer.restart();
    }


    for (auto it = bugs.begin(); it != bugs.end(); ) {
      fan::vec2 src = it->visual.get_position();
      fan::vec2 dst = path[it->current_path];

      if (it->health <= 0) {
        money += it->price;
        ++kills;
        if (kills % fan::random::i64(5, 30) == 0 && kills) {
          spawn_timer.m_time /= fan::random::value_f32(1, 1.3);
        }
        it = bugs.erase(it);
        continue;
      }

      if (src.is_near(dst, 10)) {
        // bug dies
        if (it->current_path + 1 == path.size()) {
          health -= it->damage;
          it = bugs.erase(it);
          continue;
        }
        ++it->current_path;
        dst = path[it->current_path];
      }
      fan::vec2 offset = (dst - src).normalize() * it->speed * 100.f * gloco->delta_time;
      src += offset;
      it->visual.set_position(fan::vec3(src, 0xffa0));
      ++it;
    }

    for (auto& turret : turrets) {
      if (turret.visual.iic()) { // still placing turret
        continue;
      }
      if (turret.damage_time.finished() == false) {
        continue;
      }
      turret.damage_time.restart();
      for (auto it = bugs.begin(); it != bugs.end(); ) {
        fan::vec2 src = it->visual.get_position();
        // damage the bugs
        if (fan_2d::collision::rectangle::check_collision(turret.visual.get_position(), turret.visual.get_size() * 2, src, it->visual.get_size() * 2)) {
          it->health -= turret.damage;
          break;
        }
        ++it;
      }
    }
  }

  fan::time::clock spawn_timer;
};

int main() {

  loco_t loco;

  gui_t gui;

  loco_t::image_load_properties_t lp;
  lp.visual_output = loco_t::image_sampler_address_mode::clamp_to_border;
  lp.min_filter = GL_NEAREST;
  lp.mag_filter = GL_NEAREST;
  loco_t::texturepack_t tp;

  fan::string root_folder = "examples/games/tower_defence/";
  //tp.open_compiled("texture_packs/tilemap.ftp", lp);
  tp.open_compiled(root_folder + "texturepack.ftp", lp);

  fte_renderer_t renderer;
  renderer.open(&tp);

  //auto compiled_map = renderer.compile("tilemaps/map_game0_0.fte");
  auto compiled_map = renderer.compile(root_folder + "maps/map0.json");
  fan::vec2i render_size(16, 9);
  render_size *= 2;
  render_size += 3;
    
  fte_loader_t::properties_t p;

  p.position = fan::vec3(0, 0, 0);
  p.size = (render_size * 2) * 32;

  std::vector<std::pair<int, fan::vec2>> pairs;

  for (auto& i : compiled_map.compiled_shapes) {
    for (auto& j : i) {
      for (auto& k : j) {
        if (k.id.find("path") != std::string::npos) {
          k.position.x -= tile_size;
          k.position.y -= tile_size;
          int num = std::stoi(k.id.substr(std::string("path").length()));
          pairs.emplace_back(num, k.position);
        }
      }
    }
  }

  std::sort(pairs.begin(), pairs.end(), [](const std::pair<int, fan::vec2>& a, const std::pair<int, fan::vec2>& b) {
      return a.first < b.first;
  });
  for (const auto& pair : pairs) {
      gui.path.push_back(pair.second);
  }
  gui.path.back() += (gui.path.back() - gui.path[gui.path.size() - 2]).normalize() * 200;

  init_zoom();

  auto map_id0 = renderer.add(&compiled_map, p);


  loco.set_vsync(0);
  //loco.window.set_max_fps(3);
  f32_t total_delta = 0;

  init_bcol();

  loco.lighting.ambient = 1.0;


  loco.loop([&] {
    renderer.update(map_id0, 0);

    gui.render();

    gui.play();

  });
}