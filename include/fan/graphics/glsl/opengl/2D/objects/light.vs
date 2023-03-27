R"(
#version 330

#define get_instance() instance[gl_VertexID / 6]

out vec4 instance_color;
out vec3 instance_position;
out vec2 instance_size;
out vec3 frag_position;

out vec2 texture_coordinate;


uniform mat4 view;
uniform mat4 projection;

struct block_instance_t{
	vec3 position;
  float parallax_factor;
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

  instance_position = get_instance().position;
  instance_size = get_instance().size;

  mat4 view_mat = view;

  view_mat[3].xy *= 1 - get_instance().parallax_factor;

  vec2 p = get_instance().position.xy * (1 - get_instance().parallax_factor);

  vec4 fs = vec4(vec4(vec2(x, y) * get_instance().size + get_instance().position.xy, get_instance().position.z, 1));
  vec4 fs2 = vec4(vec4(vec2(x, y) * get_instance().size + p, get_instance().position.z, 1));

  frag_position = fs.xyz;

  gl_Position = projection * view_mat * fs2;

	instance_color = get_instance().color;
}
)"