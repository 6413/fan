#version 450

layout(location = 0) in vec2 texture_coordinate;
layout(location = 0) out vec4 o_attachment0;

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
} constants;

layout(set = 0, binding = 2) uniform sampler2D textures[1024];

void main() {
  o_attachment0 = texture(textures[constants.texture_id], texture_coordinate);
}