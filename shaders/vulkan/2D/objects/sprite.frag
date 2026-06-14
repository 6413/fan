#version 450

layout(location = 0) out vec4 o_color;

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;

layout(set = 0, binding = 2) uniform sampler2D _t[1024];

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
} constants;

void main() {
  o_color = texture(_t[constants.texture_id], texture_coordinate) * instance_color;
}
