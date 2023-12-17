
#version 140

#define get_instance() instance[gl_VertexID / 6]

out vec4 text_color;
out vec4 outline_color;
out vec2 texture_coordinate;
out float render_size;
out float outline_size;

uniform mat4 view;
uniform mat4 projection;

struct block_instance_t{	
	vec3 position;
  float outline_size;
	vec2 size;
	vec2 tc_position;
  vec4 color;
  vec4 outline_color;
	vec2 tc_size;
  vec3 angle;
};


layout (std140) uniform instance_t {
	block_instance_t instance[204];
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
  vec3 rotation_vector = vec3(0, 0, 1);
  mat4 m = mat4(1);
  vec2 rotation_point = vec2(0, 0);
  m = translate(m, -vec3(rotation_point, 0));
  m = rotate(m, get_instance().angle); 
  m = translate(m, vec3(rotation_point, 0));
  vec2 rotated = vec4(m * vec4(rp * get_instance().size, 0, 1)).xy;

  gl_Position = projection * view * vec4((rotated + get_instance().position.xy), get_instance().position.z, 1);

	text_color = get_instance().color;
  outline_color = get_instance().outline_color;
	texture_coordinate = tc[id] * get_instance().tc_size + get_instance().tc_position;
	render_size = get_instance().size.y;
  outline_size = get_instance().outline_size;
}
