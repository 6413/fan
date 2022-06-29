R"(
#version 330

in vec2 texture_coordinate;

in vec4 instance_color;

out vec4 o_color;

uniform sampler2D texture_sampler;

void main() {
  o_color = texture(texture_sampler, texture_coordinate) * instance_color;
}
)"