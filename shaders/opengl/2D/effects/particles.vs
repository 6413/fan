#version 330

out vec4 i_color;

uniform uint vertex_count;
uniform uint count;
uniform vec2 position;
uniform vec2 size;
uniform vec2 position_velocity;
uniform vec3 angle_velocity;
uniform float alive_time;
uniform float respawn_time;
uniform float begin_angle;
uniform float end_angle;
uniform vec3 angle;
uniform vec4 color;
uniform vec2 gap_size;
uniform vec2 max_spread_size;
uniform vec2 size_velocity;
uniform int shape;

uniform float time;

uniform mat4 projection;
uniform mat4 view;

vec2 tc[] = vec2[](
	vec2(0, 0), // top left
	vec2(1, 0), // top right
	vec2(1, 1), // bottom right
	vec2(1, 1), // bottom right
	vec2(0, 1), // bottom left
	vec2(0, 0) // top left
);

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

mat4 rotate(mat4 m, vec3 angles) {
    float cx = cos(angles.x);
    float sx = sin(angles.x);
    float cy = cos(angles.y);
    float sy = sin(angles.y);
    float cz = cos(angles.z);
    float sz = sin(angles.z);

    mat4 rotationX = mat4(1.0, 0.0, 0.0, 0.0,
                          0.0, cx, -sx, 0.0,
                          0.0, sx, cx, 0.0,
                          0.0, 0.0, 0.0, 1.0);

    mat4 rotationY = mat4(cy, 0.0, sy, 0.0,
                          0.0, 1.0, 0.0, 0.0,
                          -sy, 0.0, cy, 0.0,
                          0.0, 0.0, 0.0, 1.0);

    mat4 rotationZ = mat4(cz, -sz, 0.0, 0.0,
                          sz, cz, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 1.0);

    mat4 matrix = rotationX * rotationY * rotationZ * m;
    return matrix;
}

vec2 triangle_vertices[] = vec2[](
	vec2(0.0, (2.0 * sqrt(3.0)) / 6.0),
	vec2(-1.0 / 2.0, -sqrt(3.0) / 6.0),
	vec2(1.0 / 2.0, -sqrt(3.0) / 6.0)
);

vec2 rectangle_vertices[] = vec2[](
	vec2(-1.0, -1.0),
	vec2(1.0, -1.0),
	vec2(1.0, 1.0),

	vec2(1.0, 1.0),
	vec2(-1.0, 1.0),
	vec2(-1.0, -1.0)
);


out vec2 texture_coordinate;

vec2 vec2_direction(uint r, uint r2, float min, float max) {
  min = -min;
  max = -max;

  float rr = mod(2 * 3.141 * floatConstruct(r), max) + min;
	float rr2 = mod(2 * 3.141 * floatConstruct(r2), max) + min;
  return vec2(cos(rr), sin(rr2));
}

float normalize(float value, float min, float max) {
	return (value - min) / (max - min);
}

void main() {
	mat4 m = mat4(1);

  int modded_index = gl_VertexID % (int(count) * 6);

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
  if (shape == 1) {
    pos.x += mod(float(id) * gap_size.x, max_spread_size.x);
    pos.y += mod(float(id) * gap_size.y, max_spread_size.y);
  }
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

  vec2 size_factor = vec2(1.0) + size_velocity * time_mod;

	m = translate(m, vec3(pos, 0));
	m = rotate(m, angle + time_mod * angle_velocity);
	m = scale(m, vec3(size * size_factor, 0));

	gl_Position = projection * view * m * vec4(rectangle_vertices[gl_VertexID % 6], 0, 1);
  gl_Position.z = 1.f - (float(modded_index) / 6.f) / count;
  i_color= vec4(
    color.r,//float(RAND(seed + 2u)) / 5000000000.f, 
    color.g,//float(RAND(seed + 3u)) / 5000000000.f, 
    color.b,//float(RAND(seed + 4u)) / 5000000000.f, 
    color.a);
  float alpha = 1.0 - time_mod / alive_time;
  i_color.a = alpha;
	texture_coordinate = tc[gl_VertexID % 6];
}

