#version 450

// Draws a full-screen triangle.
// This is used for full-screen passes over images,
// such as during the resolve step.
// We could also do this using a compute shader instead.

vec2 rectangle_vertices[] = vec2[](
	vec2(1.0, -1.0),
	
	vec2(1.0, 1.0),
	vec2(-1.0, -1.0),
	
	vec2(-1.0, -1.0),
	vec2(1.0, 1.0),
	vec2(-1.0, 1.0)
	
);

void main() {

  gl_Position = vec4(rectangle_vertices[gl_VertexIndex], 0, 1);
}