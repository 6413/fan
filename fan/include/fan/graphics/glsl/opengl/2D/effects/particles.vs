R"(
#version 330

out vec4 i_color;

uniform uint vertex_count;
uniform uint count;
uniform vec2 position;
uniform vec2 size;
uniform vec2 position_velocity;
uniform vec2 angle_velocity;
uniform vec3 rotation_vector;

uniform float time;

uniform mat4 projection;
uniform mat4 view;

uint RAND(uint seed) {
  for(uint i = 0u; i < 2u; i++){
    seed += (seed << 10u);
    seed ^= (seed >> 6u);
    seed += (seed << 3u);
    seed ^= (seed >> 11u);
    seed += (seed << 15u);
  }
  return seed;
}

float floatConstruct(uint m) {
  const uint ieeeMantissa = 0x007FFFFFu;
  const uint ieeeOne = 0x3F800000u;

  m &= ieeeMantissa;
  m |= ieeeOne;

  float f = uintBitsToFloat(m);
  return f - 1.0;
}

mat4 translate(mat4 m, vec3 v) {
	mat4 matrix = m;

	matrix[3][0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0];
	matrix[3][1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1];
	matrix[3][2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2];
	matrix[3][3] = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3];

	return matrix;
}

mat4 scale(mat4 m, vec3 v) {
	mat4 matrix = mat4(1);

	matrix[0][0] = m[0][0] * v[0];
	matrix[0][1] = m[0][1] * v[0];
	matrix[0][2] = m[0][2] * v[0];

	matrix[1][0] = m[1][0] * v[1];
	matrix[1][1] = m[1][1] * v[1];
	matrix[1][2] = m[1][2] * v[1];

	matrix[2][0] = m[2][0] * v[2];
	matrix[2][1] = m[2][1] * v[2];
	matrix[2][2] = m[2][2] * v[2];

	matrix[3][0] = m[3][0];
	matrix[3][1] = m[3][1];
	matrix[3][2] = m[3][2];

	matrix[3] = m[3];

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

const float triangle_height = 0.86602540378443864676372317075294;
const float offset = sqrt(3) / 6;

vec2 triangle_vertices[] = vec2[](
	vec2(0, 0)
);

out vec2 texture_coordinate;

vec2 vec2_direction(uint r, uint r2, float min, float max) {
  min = -min;
  max = -max;

  float rr = mod(2 * 3.141 * floatConstruct(r), max) + min;
	float rr2 = mod(2 * 3.141 * floatConstruct(r2), max) + min;
  return vec2(cos(rr), sin(rr2));
}

void main() {

	vec2 resolution = vec2(view[3][0], view[3][1]) * 2;

	mat4 m = mat4(1);

	uint id = uint(gl_VertexID) / vertex_count + 1u;

	uint seed;
	seed = id * count;
	float new_time = time + floatConstruct(RAND(seed));
	seed = uint(new_time / 10.0);
	seed *= id * count;
	seed *= 4u;
	float time_mod = mod(new_time, 10.0);

	vec2 pos;
	pos.x = (floatConstruct(RAND(seed + 0u))) * 300 + 1920 / 2 - 150;
	pos.y = (floatConstruct(RAND(seed + 1u))) * 300 + 800;
	vec2 velocity = vec2_direction(RAND(seed + 2u), RAND(seed + 3u), 3.141 / 4, 3.141 - 3.141 / 2) * (floatConstruct(RAND(seed + 4u)) * 10000);

	pos.x += velocity.x * time_mod / 10.0;
	pos.y += velocity.y * time_mod / 10.0;

	vec2 middle = vec2(0, 0);

	m = translate(m, vec3(vec2(pos) + middle, 0));

	m = rotate(m, time_mod * 10 * 3.141 * 2, vec3(0, 0, 1));

	m = translate(m, vec3(-middle, 0));

	gl_PointSize = 5;

	gl_Position = projection * view * m * vec4(triangle_vertices[gl_VertexID % 3], 0, 1);
	float r = floatConstruct(RAND(id * 3u + 0u));
	float g = floatConstruct(RAND(id * 3u + 1u));
	float b = floatConstruct(RAND(id * 3u + 2u));
	i_color = vec4(r, g, b, 1);
}
)"
