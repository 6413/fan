#version 430 core

// same locations as in 2D version to use the same class for base class
layout (location = 0) in vec4 in_color;
layout (location = 1) in vec3 src; 
layout (location = 2) in vec3 dst;

const vec3 square_vertices[] = {

	vec3(1,  1, 0), // down
	vec3(1, 0, 0),
	vec3(0, 0, 0),

	vec3(0, 0, 0),
	vec3(0,  1, 0),
	vec3( 1,  1, 0),

	vec3(0, 0, 1), // up
	vec3(1, 0, 1),
	vec3( 1, 1, 1),

	vec3(1, 1, 1),
	vec3(0, 1, 1),
	vec3(0, 0, 1),

	vec3(0, 1, 1),
	vec3(0, 1, 0),
	vec3(0, 0, 0), // front

	vec3(0, 0, 0),
	vec3(0, 0, 1),
	vec3(0, 1, 1),

	vec3(1, 0, 1),
	vec3(1, 0, 0),
	vec3(1, 1, 0), // back

	vec3(1, 1, 0),
	vec3(1, 1, 1),
	vec3(1, 0, 1),

	vec3(0, 0, 1), // right
	vec3(0, 0, 0),
	vec3(1, 0, 0),

	vec3(1, 0, 0),
	vec3(1, 0, 1),
	vec3(0, 0, 1),

	vec3(1, 1, 1), // left
	vec3(1, 1, 0),
	vec3(0, 1, 0),

	vec3(0, 1, 0),
	vec3(0, 1, 1),
	vec3(1, 1, 1)
};

uniform mat4 projection;
uniform mat4 view;
uniform int shape_type;

//const vec2 texture_coordinates[] = { // for single sided
//	vec2(0.0f, 0.0f),
//	vec2(1.0f, 0.0f),
//	vec2(1.0f, 1.0f),
//	vec2(1.0f, 1.0f),
//	vec2(0.0f, 1.0f),
//	vec2(0.0f, 0.0f),
//
//	vec2(0.0f, 0.0f),
//	vec2(1.0f, 0.0f),
//	vec2(1.0f, 1.0f),
//	vec2(1.0f, 1.0f),
//	vec2(0.0f, 1.0f),
//	vec2(0.0f, 0.0f),
//
//	vec2(1.0f, 0.0f),
//	vec2(1.0f, 1.0f),
//	vec2(0.0f, 1.0f),
//	vec2(0.0f, 1.0f),
//	vec2(0.0f, 0.0f),
//	vec2(1.0f, 0.0f),
//
//	vec2(1.0f, 0.0f),
//	vec2(1.0f, 1.0f),
//	vec2(0.0f, 1.0f),
//	vec2(0.0f, 1.0f),
//	vec2(0.0f, 0.0f),
//	vec2(1.0f, 0.0f),
//
//	vec2(0.0f, 1.0f),
//	vec2(1.0f, 1.0f),
//	vec2(1.0f, 0.0f),
//	vec2(1.0f, 0.0f),
//	vec2(0.0f, 0.0f),
//	vec2(0.0f, 1.0f),
//
//	vec2(0.0f, 1.0f),
//	vec2(1.0f, 1.0f),
//	vec2(1.0f, 0.0f),
//	vec2(1.0f, 0.0f),
//	vec2(0.0f, 0.0f),
//	vec2(0.0f, 1.0f)
//
//};

layout(std430, binding = 0) buffer texture_coordinate_layout 
{
    vec2 texture_coordinates[];
};

layout(std430, binding = 1) buffer textures_layout
{
    int textures[];
};

out vec2 texture_coordinate;
out vec4 color;

flat out int mode;

void main() {
	int index = gl_VertexID + textures[gl_InstanceID] * 36;
	texture_coordinate = texture_coordinates[index];
	color = in_color;
	mode = shape_type;
	vec3 size = dst - src;

	switch (shape_type) {
		case 1: { // line
			if (gl_VertexID % 2 == 0) {
				gl_Position = projection * view * vec4(src, 1);        
			}
			else {
				gl_Position = projection * view * vec4(dst, 1);        
			}
			break;
		}
		case 2: { // square
			vec3 vertice = square_vertices[gl_VertexID % square_vertices.length()];
			gl_Position = projection * view * vec4(vertice * size + src, 1); 
			break;
		}
	}
}