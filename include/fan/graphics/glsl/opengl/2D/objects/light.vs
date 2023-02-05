R"(
#version 330

#define get_instance() instance[gl_VertexID / 6]

layout (location = 1) in vec2 aTexCoord;

out vec4 instance_color;
out vec3 instance_position;
out vec2 instance_size;
out vec3 frag_position;

out vec2 texture_coordinate;

flat out uint instance_light_type;

uniform mat4 view;
uniform mat4 projection;

struct block_instance_t{
	vec3 position;
  uint light_type;
	vec2 size;
	vec2 rotation_point;
	vec4 color;
	vec3 rotation_vector;
	float angle;
};

layout (std140) uniform instance_t {
	block_instance_t instance[256];
};

vec2 rectangle_vertices[] = vec2[](
	vec2(-1.0, -1.0),
	vec2(1.0, -1.0),
	vec2(1.0, 1.0),

	vec2(1.0, 1.0),
	vec2(-1.0, 1.0),
	vec2(-1.0, -1.0)
);

void main() {
	uint id = uint(gl_VertexID % 6);

	vec2 ratio_size = get_instance().size;

	vec2 rp = rectangle_vertices[id];
	
	float c = cos(-get_instance().angle);
	float s = sin(-get_instance().angle);

	float x = rp.x * c - rp.y * s;
	float y = rp.x * s + rp.y * c;

  texture_coordinate = aTexCoord;

  instance_position = get_instance().position;
  instance_size = get_instance().size;
  frag_position = vec4(vec4(vec2(x, y) * get_instance().size + get_instance().position.xy, get_instance().position.z, 1)).xyz;
  instance_light_type = get_instance().light_type;

  gl_Position = projection * view * vec4(vec2(x, y) * get_instance().size + get_instance().position.xy, get_instance().position.z, 1);

	instance_color = get_instance().color;
}
)"