R"(
#version 330

layout (location = 2) out vec4 o_attachment2;

in vec4 instance_color;
in vec3 instance_position;
in vec2 instance_size;
in vec3 frag_position;
in vec2 texture_coordinate;

void main() {
  o_attachment2 = instance_color;
  float distance = length(frag_position - instance_position);
  float radius = instance_size.x;
  float smooth_edge = radius;
  float intensity = 1.0 - smoothstep(radius / 3 -smooth_edge, radius, distance);
  //float intensity2 = intensity;
  //float salsa = frag_position.y / 20;
  //salsa = max(salsa, 1);
  //intensity /= salsa;
  //intensity += 1.0 - smoothstep(radius / 3 -smooth_edge, radius, distance) * 2;
  o_attachment2 *= intensity;
}
)"