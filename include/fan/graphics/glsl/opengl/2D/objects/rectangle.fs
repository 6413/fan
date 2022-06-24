R"(
#version 140

out vec4 color;

in vec4 instance_color;

void main() {
  color = instance_color;
}
)"