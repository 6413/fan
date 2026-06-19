#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 0) out vec4 o_attachment0;

void main() {
  o_attachment0 = instance_color;
}