#version 450
layout(location = 0) in vec2 v_uv;
layout(set = 0, binding = 0) uniform sampler2D sprite_texture;
layout(push_constant) uniform pc_t {
  vec2 c0;
  vec2 c1;
  vec2 c2;
  vec2 c3;
  vec2 uv_min;
  vec2 uv_max;
  float alpha_threshold;
} pc;
layout(location = 0) out vec4 out_color;
void main() {
  float a = texture(sprite_texture, v_uv).a;
  out_color = vec4(step(pc.alpha_threshold, a));
}
