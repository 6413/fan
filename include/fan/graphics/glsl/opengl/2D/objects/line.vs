R"(
#version 140

#define get_instance() instance.st[gl_VertexID / 2]

out vec4 instance_color;

uniform mat4 view;
uniform mat4 projection;

layout (std140) uniform instance_t {
	struct{
		vec4 color;
		vec2 src;
		vec2 dst;
	}st[256];
}instance;

void main() {
	uint id = uint(gl_VertexID);

  gl_Position = view * projection * vec4(((id & 1u) == 0u) ? get_instance().src : get_instance().dst, 0, 1);
	instance_color = get_instance().color;
}
)"