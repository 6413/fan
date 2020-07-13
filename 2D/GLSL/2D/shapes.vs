#version 430 core

out vec4 color; 

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform vec4 shape_color;
uniform int shape_type;

//for line
uniform vec2 begin;
uniform vec2 end;

// shape_type = 1
const vec2 square_vertices[] = {
	vec2(1, 0),
	vec2(0, 0),
	vec2(0, 1),
	vec2(1, 0),
	vec2(1, 1),
	vec2(0, 1)
};

void main() {
    color = shape_color;
	switch(shape_type) {
		case 0: {
			gl_Position = projection * view * vec4((gl_VertexID % 2 == 0 ? begin : end), 0, 1);
			break;
		}
		case 1: {
			gl_Position = projection * view * model * vec4(square_vertices[gl_VertexID % square_vertices.length()], 0, 1);
			break;
		}
	}
}