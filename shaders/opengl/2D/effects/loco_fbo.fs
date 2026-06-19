#version 330
in vec2 texture_coordinate;
layout (location = 0) out vec4 o_attachment0;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform sampler2D _t02;
uniform sampler2D _t03;
uniform sampler2D _t04;

uniform float bloom_strength = 0.04;
uniform float bloom_intensity = 1.0;
uniform vec3 bloom_tint = vec3(1.0, 1.0, 1.0);
uniform float dirt_intensity = 1.0;

uniform float blur_amount = 0.08;
uniform bool blur_focus_enabled = false;
uniform vec2 blur_focus_position = vec2(0.5, 0.5);
uniform float blur_focus_radius = 0.25;
uniform float blur_focus_falloff = 0.15;
uniform vec2 window_size = vec2(1.0, 1.0);

uniform float gamma = 1.0;
uniform float exposure = 1.0;
uniform float contrast = 1.0;
uniform float framebuffer_alpha = 1.0;
uniform int post_process_mode = 1;

const int post_process_mode_none = 0;
const int post_process_mode_bloom = 1;
const int post_process_mode_blur = 2;
const int post_process_mode_bloom_blur = 3;

float get_blur_mask() {
  if (!blur_focus_enabled) {
    return 1.0;
  }

  vec2 ws = max(window_size, vec2(1.0));
  vec2 p = texture_coordinate - vec2(blur_focus_position.x, 1.0 - blur_focus_position.y);
  p.x *= ws.x / ws.y;
  float d = length(p);
  float r0 = max(blur_focus_radius, 0.0);
  float r1 = r0 + max(blur_focus_falloff, 0.0001);
  return smoothstep(r0, r1, d);
}

void main() {
  vec3 color = texture(_t00, texture_coordinate).rgb;

  bool bloom_enabled =
    post_process_mode == post_process_mode_bloom ||
    post_process_mode == post_process_mode_bloom_blur;
  bool blur_enabled =
    post_process_mode == post_process_mode_blur ||
    post_process_mode == post_process_mode_bloom_blur;

  if (blur_enabled) {
    vec3 blur = texture(_t03, texture_coordinate).rgb;
    float amount = clamp(blur_amount * get_blur_mask(), 0.0, 1.0);
    color = mix(color, blur, amount);
  }

  vec3 light = texture(_t04, texture_coordinate).rgb;
  color *= light;

  if (bloom_enabled) {
    vec3 bloom = texture(_t01, texture_coordinate).rgb;

    bloom = bloom * bloom_tint * bloom_intensity * (bloom_strength * 10.0);

    if (dirt_intensity > 0.0) {
      vec3 dirt = texture(_t02, texture_coordinate).rgb;
      bloom += bloom * dirt * dirt_intensity;
    }

    color += bloom;
  }

  color *= exposure;
  color = (color - 0.5) * contrast + 0.5;

  o_attachment0 = vec4(color, framebuffer_alpha);
}