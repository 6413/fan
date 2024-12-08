
#version 120
#extension GL_EXT_gpu_shader4 : require

varying vec4 instance_color;

uniform mat4 view;
uniform mat4 projection;

attribute vec3 in_position;
attribute vec3 in_size;
attribute vec4 in_color;
attribute vec3 in_angle;


// Define 3D cube vertices
vec3 rectangle_vertices[36] = vec3[](
    // Front face
    vec3(-1.0, -1.0, 1.0),
    vec3(1.0, -1.0, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(-1.0, 1.0, 1.0),
    vec3(-1.0, -1.0, 1.0),

    // Back face
    vec3(-1.0, -1.0, -1.0),
    vec3(1.0, -1.0, -1.0),
    vec3(1.0, 1.0, -1.0),
    vec3(1.0, 1.0, -1.0),
    vec3(-1.0, 1.0, -1.0),
    vec3(-1.0, -1.0, -1.0),

    // Left face
    vec3(-1.0, -1.0, -1.0),
    vec3(-1.0, -1.0, 1.0),
    vec3(-1.0, 1.0, 1.0),
    vec3(-1.0, 1.0, 1.0),
    vec3(-1.0, 1.0, -1.0),
    vec3(-1.0, -1.0, -1.0),

    // Right face
    vec3(1.0, -1.0, -1.0),
    vec3(1.0, -1.0, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(1.0, 1.0, -1.0),
    vec3(1.0, -1.0, -1.0),

    // Top face
    vec3(-1.0, 1.0, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(1.0, 1.0, -1.0),
    vec3(1.0, 1.0, -1.0),
    vec3(-1.0, 1.0, -1.0),
    vec3(-1.0, 1.0, 1.0),

    // Bottom face
    vec3(-1.0, -1.0, 1.0),
    vec3(1.0, -1.0, 1.0),
    vec3(1.0, -1.0, -1.0),
    vec3(1.0, -1.0, -1.0),
    vec3(-1.0, -1.0, -1.0),
    vec3(-1.0, -1.0, 1.0)
);
void main() {
	uint id = uint(gl_VertexID % 36);

	vec3 rp = rectangle_vertices[id];

  mat4 m = mat4(1);
  vec3 rotated = vec4(m * vec4(rp * in_size + in_position, 1)).xyz;

  gl_Position = projection * view * vec4(rotated, 1);

	instance_color = in_color;
}
