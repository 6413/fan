#version 450

layout(location = 0) out vec4 color;
layout(location = 1) out vec4 rcolor;

//layout(location = 0) in vec4 instance_color;

void main() {
  rcolor = vec4(0);
  color = vec4(0, 1, 0, 1);
}