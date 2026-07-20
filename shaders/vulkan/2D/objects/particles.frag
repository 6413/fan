#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;
layout(location = 2) in float affected_by_lighting;
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

layout(set = 0, binding = 2) uniform sampler2D textures[1024];

void main() {
  o_attachment0 = texture(textures[constants.texture_id], texture_coordinate) * instance_color;
  if (affected_by_lighting > 0.5) {
    o_attachment0.rgb *= constants.lighting_ambient.rgb;
  }
}
