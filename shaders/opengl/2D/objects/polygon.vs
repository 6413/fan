#version 330

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec4 in_color;
layout (location = 2) in vec3 in_offset;
layout (location = 3) in vec3 in_angle;
layout (location = 4) in vec2 in_rotation_point;

out vec4 color;

uniform mat4 view;
uniform mat4 projection;

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

  mat4 m = mat4(1);
  mat4 t1 = translate(mat4(1), -vec3(in_rotation_point, 0));
  mat4 t2 = translate(mat4(1), vec3(in_rotation_point, 0));
  mat4 r = rotate(mat4(1), vec3(0, 0, -in_angle)); 
  m = t1 * r * t2;
  vec2 rotated = vec4(m * vec4(in_position, 1)).xy;
  // z is assigned to in_position because of depth ordering
	gl_Position = projection * view * vec4(in_offset.xy + rotated, in_offset.z, 1);
  color = in_color;
}