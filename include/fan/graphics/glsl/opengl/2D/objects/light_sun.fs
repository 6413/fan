R"(
#version 330

layout (location = 2) out vec4 o_attachment1;

in vec4 instance_color;
in vec3 instance_position;
in vec2 instance_size;
in vec3 frag_position;

void main() {
  o_attachment1 = instance_color;
  float distance = length(frag_position - instance_position);
  float radius = instance_size.x;
  float smooth_edge = radius;
  float intensity = clamp(1.0 - (distance - radius/3 + smooth_edge) / (2*smooth_edge), 0.0, 1.0);
  float inv_salsa = 1.0 / max(frag_position.y / 20, 1.0);
  intensity *= inv_salsa;
  intensity += 1.0 - max((distance - radius/3 + smooth_edge) / smooth_edge, 0.0) * 2;
  o_attachment1.rgb = max(vec3(0), min(o_attachment1.rgb * intensity, vec3(1)));
  o_attachment1.a = 1.0;
}

)"