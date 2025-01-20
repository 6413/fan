#include <fan/pch.h>

#include _FAN_PATH(graphics/gui/tilemap_editor/renderer0.h)

struct player_t {
  static constexpr fan::vec2 speed{ 450, 450 };

  player_t() {
    visual = fan::graphics::sprite_t{ {
      .position = fan::vec3(0, 0, 10),
      .size = 32 / 2,
      .blending = true
    } };
    loco_t::shapes_t::light_t::properties_t lp;
    lp.position = visual.get_position();
    lp.size = 256;
    lp.color = fan::color(1, 0.4, 0.4, 1);

    lighting = lp;
  }
  void update() {

    if (ImGui::IsAnyItemActive()) {
      return;
    }

    f32_t dt = gloco->delta_time;
    f32_t multiplier = 1;
    if (gloco->window.key_pressed(fan::key_shift)) {
      multiplier = 3;
    }
    if (gloco->window.key_pressed(fan::key_d)) {
      velocity.x = speed.x * multiplier;
    }
    else if (gloco->window.key_pressed(fan::key_a)) {
      velocity.x = -speed.x * multiplier;
    }
    else {
      velocity.x = 0;
    }

    if (gloco->window.key_pressed(fan::key_w)) {
      velocity.y = -speed.y * multiplier;
    }
    else if (gloco->window.key_pressed(fan::key_s)) {
      velocity.y = speed.y * multiplier;
    }
    else {
      velocity.y = 0;
    }

    visual.set_velocity(velocity);

    visual.set_position(visual.get_collider_position());
    lighting.set_position(visual.get_position());
  }
  fan::vec2 velocity = 0;
  fan::graphics::collider_dynamic_t visual;
  loco_t::shape_t lighting;
};

f32_t zoom = 2;
void init_zoom() {
  auto& window = *gloco->get_window();
  auto update_ortho = [] {
    fan::vec2 s = gloco->window.get_size();
    gloco->default_camera->camera.set_ortho(
      fan::vec2(-s.x, s.x) / zoom,
      fan::vec2(-s.y, s.y) / zoom
    );;
    };

  update_ortho();

  window.add_buttons_callback([&](const auto& d) {

    if (ImGui::IsAnyItemHovered()) {
      return;
    }

    if (d.button == fan::mouse_scroll_up) {
      zoom *= 1.2;
    }
    else if (d.button == fan::mouse_scroll_down) {
      zoom /= 1.2;
    }

    update_ortho();
  });
}

int main() {
  loco_t loco;
  loco_t::image_t::load_properties_t lp;
  lp.visual_output = loco_t::image_t::sampler_address_mode::clamp_to_border;
  lp.min_filter = GL_NEAREST;
  lp.mag_filter = GL_NEAREST;
  loco_t::texturepack_t tp;
  tp.open_compiled("texture_packs/tilemap.ftp", lp);

  fte_renderer_t renderer;
  renderer.open(&tp);

  auto compiled_map = renderer.compile("tilemaps/map_game0_0.fte");

  fan::vec2i render_size(16, 9);
  render_size *= 2;
  render_size += 3;

  fte_loader_t::properties_t p;

  p.position = fan::vec3(0, 0, 0);
  p.size = (render_size * 2) * 32;

  init_zoom();

  auto map_id0_t = renderer.add(&compiled_map, p);

  player_t player;

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
          fan::print("sensor triggered");
          break;
        }
      }
    };

  loco.set_vsync(0);
  f32_t total_delta = 0;


  loco.lighting.ambient = 0.7;
  

  fan::string shader_code =
    R"(



#version 330

in vec2 texture_coordinate;

layout (location = 0) out vec4 o_attachment0;
uniform vec4 ccc;
uniform float m_time;

uniform sampler2D _t00;

void DrawVignette( inout vec3 color, vec2 uv )
{    
    float vignette = uv.x * uv.y * ( 1.0 - uv.x ) * ( 1.0 - uv.y );
    vignette = clamp( pow( 16.0 * vignette, 0.3 ), 0.0, 1.0 );
    color *= vignette;
}

vec2 CRTCurveUV( vec2 uv )
{
    uv = uv * 2.0 - 1.0;
    vec2 offset = abs( uv.yx ) / vec2( 6.0, 4.0 );
    uv = uv + uv * offset * offset;
    uv = uv * 0.5 + 0.5;
    return uv;
}
void DrawScanline( inout vec3 color, vec2 uv )
{
    float scanline 	= clamp( 0.95 + 0.05 * cos( 3.14 * ( uv.y + 0.008 * m_time ) * 240.0 * 1.0 ), 0.0, 1.0 );
    float grille 	= 0.85 + 0.15 * clamp( 1.5 * cos( 3.14 * uv.x * 640.0 * 1.0 ), 0.0, 1.0 );    
    color *= scanline * grille * 1.2;
}


void main() {
	vec2 tex = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);
	tex = CRTCurveUV(tex);
	vec3 actual = texture(_t00, tex).rgb;
	o_attachment0 = vec4(actual * ccc.rgb, ccc.a);
	DrawVignette(o_attachment0.rgb, tex);
	DrawScanline(o_attachment0.rgb, tex);
}

)";

  shader_code.resize(4096);

  loco_t::shader_t shader = loco.create_sprite_shader(shader_code);

  fan::color c0 = fan::colors::black, c1 = fan::colors::black;

  loco_t::shapes_t::shader_t::properties_t sp;
  static constexpr int postprocess_depth = 10;
  sp.position = fan::vec3(loco.window.get_size() / 2, postprocess_depth);
  sp.size = loco.window.get_size() / 2;
  sp.shader = &shader;
  sp.shader->get_shader().on_activate = [&](loco_t::shader_t* shader) {
    shader->set_vec4("ccc", c0);
  };
  sp.color.a = 0.25;
  sp.blending = true;

  loco_t::shape_t shader_shape = sp;
  //ImGuiInputTextFlags_ReadOnly
  
  shader_shape.set_image(&loco.color_buffers[0]);
  loco.window.add_key_callback(fan::key_r, fan::keyboard_state::press, [&](const auto&) {
    shader.set_vertex(loco.get_sprite_vertex_shader());
    shader.set_fragment(shader_code);
    shader.compile();
  });
  auto& io = ImGui::GetIO();

  loco.loop([&] {

    ImGui::ColorEdit4("##c0", c0.data());
    if (ImGui::InputTextMultiline("##TextFileContents", shader_code.data(), shader_code.size(), ImVec2(-1.0f, -1.0f), ImGuiInputTextFlags_AllowTabInput | ImGuiInputTextFlags_AutoSelectAll)) {
      fan::print(shader_code);
    }
    player.update();
    fan::vec2 dst = player.visual.get_position();
    fan::vec2 src = gloco->default_camera->camera.get_position();

    gloco->default_camera->camera.set_position(dst);
    shader_shape.set_position(src);
    shader_shape.set_size(loco.window.get_size() / zoom);
    renderer.update(map_id0_t, dst);
  });
}