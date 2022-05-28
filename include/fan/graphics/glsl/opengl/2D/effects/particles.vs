R"(
#version 330

out vec4 i_color;

uniform uint vertex_count;
uniform uint count;
uniform float radius;
uniform vec2 position;
uniform vec2 size;
uniform vec2 position_velocity;
uniform float angle_velocity;
uniform vec3 rotation_vector;
uniform float alive_time;
uniform float respawn_time;
uniform float begin_angle;
uniform float end_angle;
uniform sampler2D s;

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

vec2 triangle_vertices[] = vec2[](
	vec2(0, (2 * sqrt(3)) / 6),
	vec2(-1.0 / 2, -sqrt(3) / 6),
	vec2(1.0 / 2, -sqrt(3) / 6)
);

out vec2 texture_coordinate;

vec2 vec2_direction(uint r, uint r2, float min, float max) {
  min = -min;
  max = -max;

  float rr = mod(2 * 3.141 * floatConstruct(r), max) + min;
	float rr2 = mod(2 * 3.141 * floatConstruct(r2), max) + min;
  return vec2(cos(rr), sin(rr2));
}

uint rgb(float ratio)
{
    //we want to normalize ratio so that it fits in to 6 regions
    //where each region is 256 units long
    uint normalized = uint(ratio * 256 * 6);

    //find the region for this position
    uint region = normalized / 256u;

    //find the distance to the start of the closest region
    uint x = normalized % 256u;

    uint r = 0u, g = 0u, b = 0u;
    switch (region)
    {
    case 0u: r = 255u; g = 0u;   b = 0u;   g += x; break;
    case 1u: r = 255u; g = 255u; b = 0u;   r -= x; break;
    case 2u: r = 0u;   g = 255u; b = 0u;   b += x; break;
    case 3u: r = 0u;   g = 255u; b = 255u; g -= x; break;
    case 4u: r = 0u;   g = 0u;   b = 255u; r += x; break;
    case 5u: r = 255u; g = 0u;   b = 255u; b -= x; break;
    }
    return r + (g << 8u) + (b << 16u);
}

float normalize(float value, float min, float max) {
	return (value - min) / (max - min);
}

void main() {

	vec2 resolution = vec2(view[3][0], view[3][1]) * 2;

	mat4 m = mat4(1);

	uint id = uint(gl_VertexID) / vertex_count + 1u;

	uint seed;
	seed = id * count;
	float new_time = time + floatConstruct(RAND(seed)) * 10000.0;
	seed = uint(new_time / (alive_time + respawn_time));
	seed *= id * count;
	seed *= 4u;

	vec2 pos;
	pos.x = position.x;
	pos.y = position.y;
	vec2 velocity = vec2_direction(RAND(seed + 2u), RAND(seed + 3u), begin_angle, end_angle);

	float time_mod = mod(new_time, alive_time + respawn_time);
	time_mod -= respawn_time;
	if(time_mod < 0){
		i_color = vec4(0, 0, 0, 0);
		return;
	}
	float length = sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1]);
	if (length != 0) {
		velocity /= length;
	}
	velocity *= position_velocity;

	pos.x += velocity.x * time_mod;
	pos.y += velocity.y * time_mod;

	m = translate(m, vec3(pos, 0));
	m = rotate(m, time_mod * angle_velocity * 3.141 * 2, rotation_vector);
	m = scale(m, vec3(size, 0));

	gl_Position = projection * view * m * vec4(triangle_vertices[gl_VertexID % 3], 0, 1);

	float r = 1.0;
	float g = 0.1;
	float b = 0.1;

	i_color = vec4(r, g, b, 1);
}
)"
