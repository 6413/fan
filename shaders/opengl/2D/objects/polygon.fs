
#version 330

layout (location = 0) out vec4 o_attachment0;

in vec4 color;

void main() {
  o_attachment0 = color;
}
