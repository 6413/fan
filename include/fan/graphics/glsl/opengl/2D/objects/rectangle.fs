R"(
#version 140

out vec4 color;

in vec4 instance_color;

void main() {
  if (instance_color.a < 0.9) {
    discard;
  }
  color = instance_color;
}
)"