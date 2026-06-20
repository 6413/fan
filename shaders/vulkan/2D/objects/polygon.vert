#version 450

layout(std430, set = 0, binding = 0) readonly buffer vertices_t {
  uint raw_data[];
};

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
  uint texture_id1;
  uint texture_id2;
  uint texture_id3;
} constants;

struct pv_t {
  mat4 projection;
  mat4 view;
};

layout(set = 0, binding = 1) uniform upv_t {
  pv_t pv[16];
};

uint base;
float f(uint o) { return uintBitsToFloat(raw_data[base + o / 4u]); }
vec2 v2(uint o) { return vec2(f(o), f(o + 4u)); }
vec3 v3(uint o) { return vec3(f(o), f(o + 4u), f(o + 8u)); }
vec4 v4(uint o) { return vec4(f(o), f(o + 4u), f(o + 8u), f(o + 12u)); }

vec2 rotate2(vec2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

vec4 project2(vec3 p) {
  return pv[constants.camera_id].projection * pv[constants.camera_id].view * vec4(p, 1.0);
}

layout(location = 0) out vec4 color;

void main() {
  base = uint(gl_VertexIndex) * 15u;
  vec3 position = v3(0u);
  color = v4(12u);
  vec3 offset = v3(28u);
  vec3 angle = v3(40u);
  vec2 rotation_point = v2(52u);
  vec2 rotated = rotation_point + rotate2(position.xy - rotation_point, -angle.z);
  gl_Position = project2(vec3(offset.xy + rotated, offset.z));
}
