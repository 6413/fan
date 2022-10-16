R"(
#version 330

out vec4 color;

in vec4 instance_color;

float rand(vec2 co){
return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  //if (instance_color.a < 0.9) {
  //  discard;
  //}
  color = instance_color;
}
)"