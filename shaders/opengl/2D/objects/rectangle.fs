#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 instance_position;
in vec4 instance_color;
in vec4 instance_outline_color;
in vec2 texture_coordinate;
in vec2 instance_size;

uniform float outline_pixels = 2.0; 
uniform vec2 window_size;

float roundedBoxSDF(vec2 CenterPosition, vec2 Size, float Radius) {
  return length(max(abs(CenterPosition)-Size+Radius,0.0))-Radius;
}

void main() {
  vec2 smallest = vec2(1) / instance_size;
  vec2 tex_center = instance_size;
  vec2 tex_diff = abs(tex_center - (texture_coordinate * instance_size * 2 + vec2(0.5, 0.0)));
  vec2 size = instance_size - vec2(2);
  if (tex_diff.x > size.x || tex_diff.y > size.y) {
    o_attachment0 = instance_outline_color;
  } 
  else {
    o_attachment0 = instance_color;
  }
}
