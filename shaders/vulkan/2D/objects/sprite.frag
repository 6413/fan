#version 420

// bindless
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) out vec4 o_color;

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;
layout(location = 2) flat in uint flag;

layout(binding = 2) uniform sampler2D _t[];

layout( push_constant ) uniform constants_t {
	uint texture_id;
	uint camera_id;
}constants;

void main() {
  o_color = texture(_t[constants.texture_id], texture_coordinate) * instance_color;
  if (o_color.a < 0.9) {
    discard;
  }
}