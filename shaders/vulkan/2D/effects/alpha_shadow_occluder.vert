#version 450
layout(location = 0) out vec2 v_uv;
layout(push_constant) uniform pc_t {
  vec2 c0;
  vec2 c1;
  vec2 c2;
  vec2 c3;
  vec2 uv_min;
  vec2 uv_max;
  float alpha_threshold;
} pc;
const int idx[6] = int[](0, 1, 2, 2, 3, 0);
const vec2 quad_uv[4] = vec2[](
  vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)
);
void main() {
  int i = idx[gl_VertexIndex];
  vec2 cv[4] = vec2[](pc.c0, pc.c1, pc.c2, pc.c3);
  v_uv = mix(pc.uv_min, pc.uv_max, quad_uv[i]);
  gl_Position = vec4(cv[i], 0.0, 1.0);
}
