
#version 330

layout (location = 0) out vec4 o_attachment0;

in vec4 instance_color;

void main() {
  o_attachment0 = instance_color;
}
