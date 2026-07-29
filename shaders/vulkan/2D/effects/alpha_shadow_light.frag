#version 450
layout(location = 0) in vec2 v_uv;
layout(set = 0, binding = 0) uniform sampler2D shadow_texture;
layout(push_constant) uniform pc_t {
  vec2 ndc_min;
  vec2 ndc_max;
  vec4 light_color;
  float softness;
  float falloff_power;
  float angle_texel;
  float cone_angle;
  float cone_inner;
  float cone_outer;
} pc;
layout(location = 0) out vec4 out_color;
const float tau = 6.28318530718;

void main() {
  vec2 p = v_uv * 2.0 - 1.0;
  float dist = length(p);
  if (dist > 1.0) discard;
  float fu = fract((atan(p.y, p.x) / tau) + 1.0);

  float cb = texture(shadow_texture, vec2(fu, 0.5)).r;
  float center_lit = dist <= cb ? 1.0 : 0.0;

  float ratio = max(0.0, dist - cb) / max(cb, 0.05);
  float step_a = pc.angle_texel * max(min(ratio * pc.softness, 32.0), 1.0);
  float s1 = step_a;
  float s2 = step_a * 2.0;
  float s3 = step_a * 3.0;
  float s4 = step_a * 4.0;

  float lit = 0.0;
  lit += (dist <= texture(shadow_texture, vec2(fract(fu - s4), 0.5)).r ? 1.0 : 0.0) * 0.02;
  lit += (dist <= texture(shadow_texture, vec2(fract(fu - s3), 0.5)).r ? 1.0 : 0.0) * 0.06;
  lit += (dist <= texture(shadow_texture, vec2(fract(fu - s2), 0.5)).r ? 1.0 : 0.0) * 0.12;
  lit += (dist <= texture(shadow_texture, vec2(fract(fu - s1), 0.5)).r ? 1.0 : 0.0) * 0.20;
  lit += center_lit * 0.20;
  lit += (dist <= texture(shadow_texture, vec2(fract(fu + s1), 0.5)).r ? 1.0 : 0.0) * 0.20;
  lit += (dist <= texture(shadow_texture, vec2(fract(fu + s2), 0.5)).r ? 1.0 : 0.0) * 0.12;
  lit += (dist <= texture(shadow_texture, vec2(fract(fu + s3), 0.5)).r ? 1.0 : 0.0) * 0.06;
  lit += (dist <= texture(shadow_texture, vec2(fract(fu + s4), 0.5)).r ? 1.0 : 0.0) * 0.02;

  float pixel_angle = atan(p.y, p.x);
  float diff = abs(mod(pixel_angle - pc.cone_angle + 3.14159, tau) - 3.14159);
  float cone_mask = 1.0 - smoothstep(pc.cone_inner * 0.5, pc.cone_outer * 0.5, diff);
  float falloff = pow(max(1.0 - dist, 0.0), pc.falloff_power);
  float value = lit * falloff * cone_mask;
  out_color = vec4(pc.light_color.rgb * value, pc.light_color.a * value);
}