#version 450
layout(location = 0) in vec2 v_uv;
layout(set = 0, binding = 0) uniform sampler2D occluder_texture;
layout(push_constant) uniform pc_t {
  int radial_samples;
} pc;
layout(location = 0) out vec4 out_color;
void main() {
  const float tau = 6.28318530718;
  float angle = v_uv.x * tau;
  vec2 dir = vec2(cos(angle), sin(angle));
  float step_size = 1.0 / float(pc.radial_samples - 1);
  float blocker = 1.0;
  float accumulated = 0.0;
  for (int i = 0; i < pc.radial_samples; ++i) {
    float r = float(i) * step_size;
    float a = texture(occluder_texture, vec2(0.5) + dir * r * 0.5).r;
    if (a > 0.5) { blocker = r; break; }
    accumulated += a;
    if (accumulated >= 2.0) { blocker = r; break; }
  }
  out_color = vec4(blocker);
}
