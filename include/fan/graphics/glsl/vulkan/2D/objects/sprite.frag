#version 450

layout(location = 0) out vec4 o_color;

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;
layout(location = 2) in float instance_id;

layout(binding = 2) uniform sampler2D _t[8];

layout(binding = 3) uniform texture_id_t {
	uint tm[8];
};

layout( push_constant ) uniform constants
{
	uint id;
} PushConstants;

void main() {
  o_color = texture(_t[uint(PushConstants.id)], texture_coordinate) * instance_color;
  //if (o_color.a < 0.9) {
  //  discard;
  //}
}