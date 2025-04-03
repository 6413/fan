
#version 140

#define plane_count 8
#define vertex_count 6

#define total_vertex (vertex_count * plane_count)

#define get_instance() instance[0]

out vec4 instance_color;
out vec2 texture_coordinate;

uniform mat4 view;
uniform mat4 projection;

uniform vec2 window_size;

uniform float _time;

struct block_instance_t{
	vec3 position;
	vec2 size;
	vec4 color;
	vec2 tc_position;
	vec2 tc_size;
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

vec2 tc[] = vec2[](
	vec2(0, 0), // top left
	vec2(1, 0), // top right
	vec2(1, 1), // bottom right
	vec2(1, 1), // bottom right
	vec2(0, 1), // bottom left
	vec2(0, 0) // top left
);

mat4 translate(mat4 m, vec3 v) {
	mat4 matrix = m;

	matrix[3][0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0];
	matrix[3][1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1];
	matrix[3][2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2];
	matrix[3][3] = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3];

	return matrix;
}

mat4 rotate(mat4 m, float angle, vec3 v) {
	float a = angle;
	float c = cos(a);
	float s = sin(a);
	vec3 axis = vec3(normalize(v));
	vec3 temp = vec3(axis * (1.0f - c));

	mat4 rotation;
	rotation[0][0] = c + temp[0] * axis[0];
	rotation[0][1] = temp[0] * axis[1] + s * axis[2];
	rotation[0][2] = temp[0] * axis[2] - s * axis[1];

	rotation[1][0] = temp[1] * axis[0] - s * axis[2];
	rotation[1][1] = c + temp[1] * axis[1];
	rotation[1][2] = temp[1] * axis[2] + s * axis[0];

	rotation[2][0] = temp[2] * axis[0] + s * axis[1];
	rotation[2][1] = temp[2] * axis[1] - s * axis[0];
	rotation[2][2] = c + temp[2] * axis[2];

	mat4 matrix;
	matrix[0][0] = (m[0][0] * rotation[0][0]) + (m[1][0] * rotation[0][1]) + (m[2][0] * rotation[0][2]);
	matrix[1][0] = (m[0][1] * rotation[0][0]) + (m[1][1] * rotation[0][1]) + (m[2][1] * rotation[0][2]);
	matrix[2][0] = (m[0][2] * rotation[0][0]) + (m[1][2] * rotation[0][1]) + (m[2][2] * rotation[0][2]);

	matrix[0][1] = (m[0][0] * rotation[1][0]) + (m[1][0] * rotation[1][1]) + (m[2][0] * rotation[1][2]);
	matrix[1][1] = (m[0][1] * rotation[1][0]) + (m[1][1] * rotation[1][1]) + (m[2][1] * rotation[1][2]);
	matrix[2][1] = (m[0][2] * rotation[1][0]) + (m[1][2] * rotation[1][1]) + (m[2][2] * rotation[1][2]);

	matrix[0][2] = (m[0][0] * rotation[2][0]) + (m[1][0] * rotation[2][1]) + (m[2][0] * rotation[2][2]);
	matrix[1][2] = (m[0][1] * rotation[2][0]) + (m[1][1] * rotation[2][1]) + (m[2][1] * rotation[2][2]);
	matrix[2][2] = (m[0][2] * rotation[2][0]) + (m[1][2] * rotation[2][1]) + (m[2][2] * rotation[2][2]);

	matrix[3] = m[3];

	return matrix;
}

float rotation_speed = 1;
float rotation_angle = 1;

float calculate_vertex_angle(int plane) {
  float time = _time;
  float id = gl_VertexID / total_vertex;
  time -= id;
  float wind = sin(time * rotation_speed);
  wind -= sin((time * rotation_speed) / 2);
  wind -= sin((time * rotation_speed) / 4); 
  wind -= sin((time * rotation_speed) / 8);
  float result = (float(plane)) * rotation_angle * -wind;
  return result;
}

void main() {
	uint id = uint(gl_VertexID % vertex_count);
  vec2 rp = rectangle_vertices[id];

  mat4 view_mat = view;

  mat4 m = mat4(1);

  vec2 rotation_point = vec2(0, -get_instance().size.y);

  float angle = calculate_vertex_angle(((gl_VertexID / vertex_count) % plane_count) + 1);

  m = translate(m, -vec3(rotation_point, 0));
  m = rotate(m, angle, vec3(0, 0, 1)); 
  m = translate(m, vec3(rotation_point, 0));

  vec2 size = get_instance().size;
  size.x /= ((gl_VertexID / vertex_count  + 1) % plane_count) * 2;

  vec2 rotated = vec4(m * vec4(rp * size, 0, 1)).xy;
  
  vec2 offset = vec2(0);
  // fix optimize
  for (int i = 0; i < (gl_VertexID / vertex_count) % plane_count; i += 1) {
    float prev_angle = calculate_vertex_angle(i + 1);
    offset += vec2(sin(prev_angle) * 2, cos(prev_angle) * 2) * get_instance().size.y;
  }
  rotated -= offset;

  gl_Position = projection * view_mat * vec4(rotated + get_instance().position.xy, get_instance().position.z, 1);
	instance_color = get_instance().color;
	texture_coordinate = tc[id] * get_instance().tc_size + get_instance().tc_position;
}
