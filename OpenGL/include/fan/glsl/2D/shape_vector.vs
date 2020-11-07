#version 430 core
layout (location = 0) in vec4 in_color;
layout (location = 1) in vec2 position;
layout (location = 2) in vec2 size;
layout (location = 3) in vec2 triangle_left;
layout (location = 4) in vec2 triangle_middle;
layout (location = 5) in vec2 triangle_right;

out vec4 color; 

uniform mat4 projection;
uniform mat4 view;
uniform int shape_type;

const vec2 line_vertices[] = {
    vec2(1, 1),
    vec2(1, 1)
};

const vec2 triangle_vertices[] = {
    vec2(50, 0),
    vec2(-50, 100),
    vec2(50, 100)
};

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
        case 0: { // line
            vec2 vertice = line_vertices[gl_VertexID % line_vertices.length()];
            if (gl_VertexID % 2 == 0) {
                gl_Position = projection * view * vec4(vec2(vertice.x + position.x, vertice.y + position.y), 0, 1);        
            }
            else {
                gl_Position = projection * view * vec4(vec2(vertice.x + size.x, vertice.y + size.y), 0, 1);        
            }
            break;
        }
        case 1: { // square
             vec2 vertice = square_vertices[gl_VertexID % square_vertices.length()];
             gl_Position = projection * view * vec4(vec2(vertice.x * size.x + position.x, vertice.y * size.y + position.y), 0, 1);
             break;
        }
        case 2: {
             if (gl_VertexID == 0) {
                gl_Position = projection * view * vec4(vec2(triangle_left.x, triangle_left.y), 0, 1);
             }
             if (gl_VertexID == 1) {
                gl_Position = projection * view * vec4(vec2(triangle_middle.x, triangle_middle.y), 0, 1);
             }
             if (gl_VertexID == 2) {
                gl_Position = projection * view * vec4(vec2(triangle_right.x, triangle_right.y), 0, 1);
             }
             break;
        }
    }
}