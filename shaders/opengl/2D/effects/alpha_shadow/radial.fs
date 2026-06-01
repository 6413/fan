#version 330 core
in vec2 v_uv;
uniform sampler2D occluder_texture;
uniform int radial_samples;
out vec4 out_color;
void main() {
  const float tau = 6.28318530718;
  float angle = v_uv.x * tau;
  vec2 dir = vec2(cos(angle), sin(angle));
  float step_size = 1.0 / float(radial_samples - 1);
  float blocker = 1.0;
  float accumulated = 0.0;
  for (int i = 0; i < radial_samples; ++i) {
    float r = float(i) * step_size;
    float a = texture(occluder_texture, vec2(0.5) + dir * r * 0.5).r;
    if (a > 0.5) { blocker = r; break; }
    accumulated += a;
    if (accumulated >= 2.0) { blocker = r; break; }
  }
  out_color = vec4(blocker, blocker, blocker, 1.0);
}
