#version 450

layout(location = 0) in vec4 vertex_color;
layout(location = 0) out vec4 o_attachment0;

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
  uint texture_id1;
  uint texture_id2;
  uint texture_id3;
  uint pad0;
  uint pad1;
  float ambient_floor;
  vec4 lighting_ambient;
} constants;

void main() {
  o_attachment0 = vertex_color;
  o_attachment0.rgb *= constants.lighting_ambient.rgb;
}