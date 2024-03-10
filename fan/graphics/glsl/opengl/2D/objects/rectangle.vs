
#version 140

#define get_instance() instance[gl_VertexID / 6]

out vec4 instance_color;

uniform mat4 view;
uniform mat4 projection;

struct block_instance_t{
	vec3 position;
	vec2 size;
	vec2 rotation_point;
	vec4 color;
	vec3 angle;
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
	uint id = uint(gl_VertexID % 6);

	vec2 ratio_size = get_instance().size;

	vec2 rp = rectangle_vertices[id];

  mat4 m = mat4(1);
  mat4 t1 = translate(mat4(1), -vec3(get_instance().rotation_point, 0));
  mat4 t2 = translate(mat4(1), vec3(get_instance().rotation_point, 0));
  mat4 r = rotate(mat4(1), get_instance().angle); 
  m = t2 * r * t1;
  vec2 rotated = vec4(m * vec4(rp* get_instance().size, 0, 1)).xy;

  gl_Position = projection * view * vec4(rotated + get_instance().position.xy, get_instance().position.z, 1);

	instance_color = get_instance().color;
}
