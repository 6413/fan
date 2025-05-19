#include <fan/time/time.h>

import fan;

int main() {
  using namespace fan::graphics;

  engine_t loco;

  fan::vec2 initial_position = fan::vec2(loco.window.get_size() / 2);
  fan::vec2 initial_size = loco.window.get_size().y / 2;

  engine_t::image_t background;
  background = loco.image_create(fan::color(1, 0, 0, 1));

  fan::graphics::sprite_t sprite{ {
    .position = fan::vec3(initial_position, 1),
    .size = initial_size,
    .image = background,
    .blending = true
  } };

  std::string shader_code;
  fan::io::file::try_write("2.glsl", R"(#version 330

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
	tex = CRTCurveUV(tex*1.05);
	vec3 actual = texture(_t00, tex).rgb;
	o_attachment0 = vec4(actual, 1);
	DrawVignette(o_attachment0.rgb, tex);
	DrawScanline(o_attachment0.rgb, tex);
})");
  if (fan::io::file::read("2.glsl", &shader_code)) {
    return 1;
  }

  shader_code.resize(4096);

  engine_t::shader_t shader = loco.get_sprite_vertex_shader(shader_code);

  fan::color input_color = fan::colors::red / 10;
  input_color.a = 0.1;
  engine_t::image_t image = loco.image_load("images/lava_seamless.webp");

  engine_t::shader_shape_t::properties_t sp;
  sp.position = fan::vec3(fan::vec2(sprite.get_position()), 3);
  sp.size = sprite.get_size();
  sp.shader = shader;
  sp.blending = true;
  sp.image = image;
  

  engine_t::shape_t shader_shape = sp;

  bool shader_compiled = true;

  loco.window.add_key_callback(fan::key_s, fan::keyboard_state::press, [&](const auto&) {
    if (!loco.is_key_down(fan::key_left_control)) {
      return;
    }
    //auto shader_nr = loco.shader_get();
    loco.shader_set_vertex(shader, loco.shader_list[loco.shaper.GetShader(engine_t::shape_type_t::shader_shape)].svertex);
    loco.shader_set_fragment(shader, shader_code);
    shader_compiled = loco.shader_compile(shader);
  });

  fan::time::clock c;
  c.start();

  loco.loop([&] {

    loco.shader_set_value(shader, "time", c.elapsed() / 1e+9f);

    static bool toggle_color = false;
    if (gui::checkbox("toggle color", &toggle_color)) {
      loco.image_unload(background);
      background = loco.image_create(toggle_color == false ? fan::colors::black : fan::colors::white);
    }

    if (shader_compiled == false) {
      gui::text("failed to compile shader", fan::colors::green);
    }
    
    initial_position = fan::vec2(loco.window.get_size() / 2);
    initial_size = loco.window.get_size().y / 2;

    static fan::vec2 offset = 0;
    if (gui::drag_float("offset", &offset)) {
      sprite.set_position(initial_position + offset);
      shader_shape.set_position(initial_position + offset);
    }

    static fan::vec2 size_offset = 0;
    if (gui::drag_float("size offset aspect", &size_offset)) {
      size_offset.y = size_offset.x;
      sprite.set_size(initial_size + size_offset);
      shader_shape.set_size(initial_size + size_offset);
    }
    if (gui::drag_float("size_offset", &size_offset)) {
      sprite.set_size(initial_size + size_offset);
      shader_shape.set_size(initial_size + size_offset);
    }

    if (gui::color_edit4("##c0", &input_color)) {
      loco.shader_set_value(shader, "input_color", input_color);
    }
    if (gui::input_text_multiline("##TextFileContents", &shader_code, ImVec2(-1.0f, -1.0f), gui::input_text_flags_allow_tab_input | gui::input_text_flags_auto_select_all)) {
      fan::io::file::write("2.glsl", shader_code.c_str(), std::ios_base::binary);
    }

  });

  return 0;
}
