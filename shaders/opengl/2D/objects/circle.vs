#version 330
layout (location = 0) in vec3 in_position;
layout (location = 1) in float in_radius;
layout (location = 2) in vec2 in_rotation_point;
layout (location = 3) in vec4 in_color;
layout (location = 4) in vec3 in_angle;
layout (location = 5) in uint in_flags;

out vec4 instance_color;
out vec3 instance_position;
out float instance_radius;
out vec3 frag_position;
out vec2 texture_coordinate;
flat out uint flags;

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

vec2 tc[] = vec2[](
	vec2(0, 0), // top left
	vec2(1, 0), // top right
	vec2(1, 1), // bottom right
	vec2(1, 1), // bottom right
	vec2(0, 1), // bottom left
	vec2(0, 0) // top left
);

void main() {
	vec2 rp = rectangle_vertices[gl_VertexID % 6];

  float x = rp.x;
  float y = rp.y;

  texture_coordinate = tc[gl_VertexID % 6];

  instance_position = in_position;
  instance_color = in_color;
  instance_radius = in_radius;
  frag_position = vec4(vec2(x, y) * vec2(instance_radius) + instance_position.xy, instance_position.z, 1).xyz;
  flags = in_flags;

  gl_Position = projection * view * vec4(vec2(x, y) * vec2(instance_radius) + instance_position.xy, instance_position.z, 1);
}
