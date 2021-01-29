#version 430 core
layout (location = 0) in vec4 in_color;
layout (location = 1) in vec2 position;
layout (location = 2) in vec2 size;

out vec4 color; 

uniform mat4 projection;
uniform mat4 view;
uniform int shape_type;

const vec2 square_vertices[] = {
	vec2(0, 0),
	vec2(0, 1),
	vec2(1, 1),
	vec2(1, 1),
	vec2(1, 0),
	vec2(0, 0)
};

void main() {
    color = in_color;
    switch (shape_type) {
        case 0: {
            gl_Position = projection * view * vec4(position, 0, 1);
            break;
        }
        case 1: { // line
            if (gl_VertexID % 2 == 0) {
                gl_Position = projection * view * vec4(position, 0, 1);        
            }
            else {
                gl_Position = projection * view * vec4(size, 0, 1);        
            }
            break;
        }
        case 2: { // square
             vec2 vertice = square_vertices[gl_VertexID % square_vertices.length()];
             gl_Position = projection * view * vec4(vec2(vertice.x * size.x + position.x, vertice.y * size.y + position.y), 0, 1);
             break;
        }
    }
}