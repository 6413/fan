#version 330

layout (location = 0) in vec3 in_position;
layout (location = 1) in float in_parallax_factor;
layout (location = 2) in vec2 in_size;
layout (location = 3) in vec2 in_rotation_point;
layout (location = 4) in vec4 in_color;
layout (location = 5) in uint in_flags;
layout (location = 6) in vec3 in_angle;


out vec4 instance_color;
out vec3 instance_position;
out vec2 instance_size;
out vec3 frag_position;
out vec2 uv;

out vec2 texture_coordinate;
flat out uint fs_flags;

uniform mat4 view;
uniform mat4 projection;

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

	vec2 ratio_size = in_size;

	vec2 rp = rectangle_vertices[id];
	
	float c = cos(0/*-in_angle*/);
	float s = sin(0/*-in_angle*/);

	float x = rp.x * c - rp.y * s;
	float y = rp.x * s + rp.y * c;

	instance_position = in_position;
	instance_size = in_size;

	mat4 view_mat = view;

	view_mat[3].xy *= 1 - in_parallax_factor;

	vec2 p = in_position.xy * (1 - in_parallax_factor);

	vec4 fs = vec4(vec4(vec2(x, y) * in_size + in_position.xy, in_position.z, 1));
	vec4 fs2 = vec4(vec4(vec2(x, y) * in_size + p, in_position.z, 1));

	frag_position = fs.xyz;
  
  uv = rp;
	gl_Position = projection * view_mat * fs2;

	instance_color = in_color;
	fs_flags = in_flags;
}
