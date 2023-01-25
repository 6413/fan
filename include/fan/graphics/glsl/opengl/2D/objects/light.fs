R"(
#version 330

layout (location = 0) out vec4 o_attachment0;
layout (location = 1) out uint o_attachment1;
layout (location = 2) out vec4 o_attachment2;

in vec4 instance_color;
in vec3 instance_position;
in vec2 instance_size;
in vec3 frag_position;

void main() {
  o_attachment2 += instance_color;

  vec3 lightDir = normalize(instance_position - frag_position);
  float distance = length(frag_position - instance_position);
  float radius = instance_size.x;
  float smooth_edge = 0.1;
  float intensity = 1.0 - smoothstep(radius / 3 -smooth_edge, radius, distance);
  o_attachment2 *= intensity;
  //diffuse  *= intensity;
  //specular *= intensity;

}
)"