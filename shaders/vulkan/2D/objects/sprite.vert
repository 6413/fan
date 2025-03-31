#version 430

#define get_instance() instances[gl_InstanceIndex]

layout(location = 0) out vec4 instance_color;
layout(location = 1) out vec2 texture_coordinate;
layout(location = 2) out flat uint flag;



struct block_instance_t{
	vec3 position;
  float parallax_factor;
  vec2 size;
  vec2 rotation_point;
  vec4 color;
  vec3 angle;
  uint flags;
  vec2 tc_position;
  vec2 tc_size;
  float seed;
};

layout(std140, binding = 0) readonly buffer instances_t{
	block_instance_t instances[];
};

layout(push_constant) uniform constants_t {
	uint texture_id;
	uint camera_id;
}constants;

struct pv_t {
	mat4 projection;
	mat4 view;
};

layout(binding = 1) uniform upv_t {
	pv_t pv[16];
};

vec2 tc[] = vec2[](
	vec2(0, 0), // top left
	vec2(1, 0), // top right
	vec2(1, 1), // bottom right
	vec2(1, 1), // bottom right
	vec2(0, 1), // bottom left
	vec2(0, 0) // top left
);

vec2 rectangle_vertices[] = vec2[](
	vec2(-1.0, -1.0),
	vec2(1.0, -1.0),
	vec2(1.0, 1.0),

	vec2(1.0, 1.0),
	vec2(-1.0, 1.0),
	vec2(-1.0, -1.0)
);

mat4 translate(mat4 m, vec3 v) {
	mat4 matrix = m;

	matrix[3][0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0];
	matrix[3][1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1];
	matrix[3][2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2];
	matrix[3][3] = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3];

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


void main() {
	uint id = uint(gl_VertexIndex % 6);

	vec2 rp = rectangle_vertices[id];

	mat4 view = pv[constants.camera_id].view;
	mat4 view_mat = view;

  mat4 m = mat4(1);
  m = translate(m, -vec3(get_instance().rotation_point, 0));
  m = rotate(m, get_instance().angle); 
  m = translate(m, vec3(get_instance().rotation_point, 0));

  vec2 rotated = vec4(m * vec4(rp * get_instance().size, 0, 1)).xy;

  vec4 view_position = view * vec4(rotated + get_instance().position.xy, get_instance().position.z, 1);

  gl_Position = pv[constants.camera_id].projection * view_position;
	instance_color = get_instance().color;
	texture_coordinate = tc[id] * get_instance().tc_size + get_instance().tc_position;
}