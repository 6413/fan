#version 450

layout(std430, set = 0, binding = 0) readonly buffer instances_t {
  uint raw_data[];
};

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
  uint texture_id1;
  uint texture_id2;
  uint texture_id3;
  float time;
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

float apply_ease(float t, uint e) {
  if (e == 0u) return t;
  if (e == 1u) return (sin((t - 0.5) * 3.14159265) + 1.0) * 0.5;
  if (e == 2u) return sin(t * 3.14159265);
  if (e == 3u) return t * t;
  if (e == 4u) return 1.0 - (1.0 - t) * (1.0 - t);
  return t;
}

layout(location = 0) out vec4 instance_color;
layout(location = 1) out vec3 instance_position;
layout(location = 2) out vec2 instance_size;
layout(location = 3) out vec3 frag_position;
layout(location = 4) out vec2 uv;
layout(location = 5) flat out uint fs_flags;

void main() {
  base = uint(gl_InstanceIndex) * 28u;
  uint id = uint(gl_VertexIndex) % 6u;
  vec2 rp = rectangle_vertices[id];
  vec3 position = v3(0u);
  float seed = f(108u);
  
  vec2 size = v2(20u);
  vec2 rotation_point = v2(28u);
  float angle = v3(56u).z;
  vec2 local = rotate_around(rp * size, rotation_point, angle);
  frag_position = vec3(local + position.xy, position.z);
  gl_Position = project2(frag_position);
  
  vec4 base_color = v4(36u);
  
  vec4 target_color = v4(68u);
  float variance_speed = f(84u);
  float flicker_speed = f(88u);
  float flicker_min = f(92u);
  float flicker_max = f(96u);
  uint ease_types = u(100u);
  uint dynamic_flags = u(104u);
  
  bool enable_flicker = (dynamic_flags & 1u) != 0u;
  bool enable_variance = (dynamic_flags & 2u) != 0u;
  
  uint flicker_ease = ease_types & 0xfu;
  uint variance_ease = (ease_types >> 4u) & 0xfu;
  
  vec4 final_color = base_color;
  
  if (enable_variance) {
    float raw_t = mod((constants.time + seed) * variance_speed, 2.0);
    float half_t = raw_t > 1.0 ? 2.0 - raw_t : raw_t;
    float t = apply_ease(half_t, variance_ease);
    final_color = mix(base_color, target_color, t);
  }
  
  if (enable_flicker) {
    float raw_t = mod((constants.time + seed) * flicker_speed, 2.0);
    float half_t = raw_t > 1.0 ? 2.0 - raw_t : raw_t;
    float t = apply_ease(half_t, flicker_ease);
    float intensity = mix(flicker_min, flicker_max, t);
    final_color.a = final_color.a * intensity;
  }
  
  instance_color = final_color;
  instance_position = position;
  instance_size = size;
  uv = rp;
  fs_flags = u(52u);
}