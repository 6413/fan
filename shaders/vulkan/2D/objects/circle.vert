#version 450

layout(std430, set = 0, binding = 0) readonly buffer instances_t {
  uint raw_data[];
};

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
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
uint u(uint o) { return raw_data[base + o / 4u]; }
int i(uint o) { return int(raw_data[base + o / 4u]); }
vec2 v2(uint o) { return vec2(f(o), f(o + 4u)); }
vec3 v3(uint o) { return vec3(f(o), f(o + 4u), f(o + 8u)); }
vec4 v4(uint o) { return vec4(f(o), f(o + 4u), f(o + 8u), f(o + 12u)); }

vec2 rectangle_vertices[6] = vec2[](
  vec2(-1.0, -1.0),
  vec2(1.0, -1.0),
  vec2(1.0, 1.0),
  vec2(1.0, 1.0),
  vec2(-1.0, 1.0),
  vec2(-1.0, -1.0)
);

vec2 texture_coordinates[6] = vec2[](
  vec2(0.0, 0.0),
  vec2(1.0, 0.0),
  vec2(1.0, 1.0),
  vec2(1.0, 1.0),
  vec2(0.0, 1.0),
  vec2(0.0, 0.0)
);

vec2 rotate2(vec2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

vec2 rotate_around(vec2 p, vec2 pivot, float a) {
  return pivot + rotate2(p - pivot, a);
}

vec4 project2(vec3 p) {
  return pv[constants.camera_id].projection * pv[constants.camera_id].view * vec4(p, 1.0);
}

layout(location = 0) out vec4 instance_color;
layout(location = 1) out vec3 instance_position;
layout(location = 2) out float instance_radius;
layout(location = 3) out vec3 frag_position;
layout(location = 4) flat out uint flags;
layout(location = 5) out vec4 instance_outline_color;
layout(location = 6) out float instance_outline_width;

void main() {
  base = uint(gl_InstanceIndex) * 19u;
  uint id = uint(gl_VertexIndex) % 6u;
  vec2 rp = rectangle_vertices[id];
  vec3 position = v3(0u);
  float radius = f(12u);
  frag_position = vec3(rp * radius + position.xy, position.z);
  gl_Position = project2(frag_position);
  instance_color = v4(24u);
  instance_position = position;
  instance_radius = radius;
  flags = u(52u);
  instance_outline_color = v4(56u);
  instance_outline_width = f(72u);
}