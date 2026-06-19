#version 330 core
in vec2 v_uv;
uniform sampler2D shadow_texture;
uniform vec4 light_color;
uniform float softness;
uniform float falloff_power;
uniform float angle_texel;
uniform float cone_angle;
uniform float cone_inner;
uniform float cone_outer;
out vec4 out_color;

float sample_shadow(float u, float dist) {
  float blocker = texture(shadow_texture, vec2(fract(u), 0.5)).r;
  float penumbra = max(softness * (1.0 - blocker), 0.001);
  return 1.0 - smoothstep(blocker - penumbra, blocker + penumbra, dist);
}

void main() {
  const float tau = 6.28318530718;
  vec2 p = v_uv * 2.0 - 1.0;
  float dist = length(p);
  if (dist > 1.0) { discard; }

  float u = atan(p.y, p.x) / tau;
  if (u < 0.0) { u += 1.0; }

  float w[5] = float[](0.20, 0.20, 0.12, 0.06, 0.02);
  float lit = 0.0;
  for (int i = -4; i <= 4; ++i) {
    lit += sample_shadow(u + angle_texel * float(i), dist) * w[abs(i)];
  }

  float cone_mask = 1.0;
  if (cone_outer < tau) {
    float pixel_angle = atan(p.y, p.x);
    float diff = abs(mod(pixel_angle - cone_angle + 3.14159, tau) - 3.14159);
    cone_mask = 1.0 - smoothstep(cone_inner * 0.5, cone_outer * 0.5, diff);
  }

  float falloff = pow(max(1.0 - dist, 0.0), falloff_power);
  float value = lit * falloff * cone_mask;
  out_color = vec4(light_color.rgb * value, light_color.a * value);
}