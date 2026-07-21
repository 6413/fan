#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;
layout(location = 2) in flat uvec4 instance_texture_ids;
layout(location = 0) out vec4 o_attachment0;

layout(push_constant) uniform constants_t {
  uint _pad0;
  uint camera_id;
} constants;

layout(set = 0, binding = 2) uniform sampler2D textures[1024];

void main() {
  o_attachment0 = texture(textures[instance_texture_ids.x], texture_coordinate) * instance_color;
}
