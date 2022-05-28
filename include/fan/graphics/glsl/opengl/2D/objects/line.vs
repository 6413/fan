R"(
#version 130

in vec4 input0;
in vec2 input1;

uniform mat4 projection;
uniform mat4 view;

out vec4 instance_color;

vec2 line_vertices[] = vec2[](
	vec2(0, 0),
  vec2(300, 300)
);


void main() {
  vec4 color = vec4(input0[0], input0[1], input0[2], input0[3]);
  vec2 position = vec2(input1[0], input1[1]);

  gl_Position = projection * view * vec4(position, 0, 1);
  instance_color = color;
}

)"