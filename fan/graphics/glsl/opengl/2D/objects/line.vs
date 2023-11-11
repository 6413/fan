
#version 140

#define get_instance() instance[gl_VertexID / 2]

out vec4 instance_color;

uniform mat4 view;
uniform mat4 projection;

struct block_instance_t {
	vec4 color;
	vec3 src;
	vec2 dst;
};

layout (std140) uniform instance_t {
	block_instance_t instance[256];
};

void main() {
	uint id = uint(gl_VertexID);

  gl_Position = projection * view * vec4(((id & 1u) == 0u) ? get_instance().src.xy : get_instance().dst.xy, get_instance().src.z, 1);
	instance_color = get_instance().color;
}
