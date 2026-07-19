#version 420

layout(set = 0, binding = 0) uniform sampler2D _t00;
layout(set = 0, binding = 1) uniform sampler2D _t01;
layout(set = 0, binding = 2) uniform sampler2D _t02;
layout(set = 0, binding = 3) uniform sampler2D _t03;

layout(push_constant) uniform push_constants_t {
  vec4 bloom_tint_strength;
  vec4 blur_focus;
  vec4 window_frame;
  vec4 params0;
  vec4 tonemap;
} pc;

layout(location = 0) in vec2 texture_coordinate;
layout(location = 0) out vec4 o_color;

const int post_process_mode_none = 0;
const int post_process_mode_bloom = 1;
const int post_process_mode_blur = 2;
const int post_process_mode_bloom_blur = 3;

float get_blur_mask() {
  if (pc.params0.w < 0.5) {
    return 1.0;
  }

  vec2 ws = max(pc.window_frame.xy, vec2(1.0));
  vec2 p = texture_coordinate - vec2(pc.blur_focus.x, 1.0 - pc.blur_focus.y);
  p.x *= ws.x / ws.y;
  float d = length(p);
  float r0 = max(pc.blur_focus.z, 0.0);
  float r1 = r0 + max(pc.blur_focus.w, 0.0001);
  return smoothstep(r0, r1, d);
}

void main() {
  vec3 color = texture(_t00, texture_coordinate).rgb;
  int mode = int(pc.window_frame.w + 0.5);

  bool bloom_enabled = mode == post_process_mode_bloom || mode == post_process_mode_bloom_blur;
  bool blur_enabled = mode == post_process_mode_blur || mode == post_process_mode_bloom_blur;

  if (blur_enabled) {
    vec3 blur = texture(_t03, texture_coordinate).rgb;
    float amount = clamp(pc.params0.z * get_blur_mask(), 0.0, 1.0);
    color = mix(color, blur, amount);
  }

  if (bloom_enabled) {
    vec3 bloom = texture(_t01, texture_coordinate).rgb;
    bloom *= pc.bloom_tint_strength.rgb * pc.params0.x * (pc.bloom_tint_strength.w * 100.0); 

    if (pc.params0.y > 0.0) {
      vec3 dirt = texture(_t02, texture_coordinate).rgb;
      bloom += bloom * dirt * pc.params0.y;
    }

    color += bloom;
  }

  color *= pc.tonemap.y;
  color = (color - 0.5) * pc.tonemap.z + 0.5;

  o_color = vec4(color, pc.window_frame.z);
}
