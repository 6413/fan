#version 450
layout(location = 0) out vec2 v_uv;
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
vec2 verts[6] = vec2[](
  vec2(0, 0), vec2(1, 0), vec2(1, 1),
  vec2(1, 1), vec2(0, 1), vec2(0, 0)
);
void main() {
  v_uv = verts[gl_VertexIndex];
  vec2 ndc = mix(pc.ndc_min, pc.ndc_max, v_uv);
  gl_Position = vec4(ndc, 0.0, 1.0);
}
