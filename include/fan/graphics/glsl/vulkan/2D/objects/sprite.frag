#version 450

layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 o2_color;

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;

layout(binding = 2) uniform sampler2D _t[16];

layout( push_constant ) uniform constants_t {
	uint texture_id;
	uint matrices_id;
}constants;

void main() {
  o_color = texture(_t[constants.texture_id], texture_coordinate) * instance_color;
  o2_color.r = 1;
  o2_color.a = 1.0f;
  if (o_color.a < 0.9) {
    discard;
  }
}