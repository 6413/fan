#version 450

layout(location = 0) out vec4 o_color;

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;

layout(binding = 5) uniform sampler2D _t00;

void main() {
  o_color = texture(_t00, texture_coordinate) * instance_color;
  if (o_color.a < 0.9) {
    discard;
  }
}