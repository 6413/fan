#version 130

in vec4 in_color;
in vec2 position;
in vec2 size;
in float angle;

out vec4 color;

uniform mat4 projection;
uniform mat4 view;

mat4 translate(mat4 m, vec3 v) {
	mat4 matrix = m;

	matrix[3][0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0];
	matrix[3][1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1];
	matrix[3][2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2];
	matrix[3][3] = m[0][3] * v[0] + m[1][3] * v[1] + m[2][3] * v[2] + m[3][3];

	return matrix;
}

mat4 scale(mat4 m, vec3 v) {
	mat4 matrix;

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

vec2 rectangle_vertices[] = vec2[](
	vec2(0, 0),
	vec2(1, 0),
	vec2(1, 1),

	vec2(0, 0),
	vec2(0, 1),
	vec2(1, 1)
);

void main() {

	mat4 m = mat4(1);

	m = translate(mat4(1), vec3(position.x + size.x / 2, position.y + size.y / 2, 0));

	m = rotate(m, angle, vec3(0, 0, 1));

	m = translate(m, vec3(-size.x / 2, -size.y / 2, 0));

	m = scale(m, vec3(size.x, size.y, 0));

	vec4 m2 = projection * view * m * vec4(rectangle_vertices[gl_VertexID % 6].x, rectangle_vertices[gl_VertexID % 6].y, 0, 1);

	gl_Position = m2;

	color = in_color;
}