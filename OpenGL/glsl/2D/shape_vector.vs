#version 130

precision highp float;

attribute vec4 in_color;
attribute vec2 position;

out vec4 color; 

uniform mat4 projection;
uniform mat4 view;
uniform int shape_type;

out vec2 texture_coordinate;

const vec2 texture_coordinates[] = vec2[](
	vec2(0, 0),
	vec2(0, 1),
	vec2(1, 1),
	vec2(0, 0),
	vec2(1, 0),
	vec2(1, 1)
);

void main() {
    color = in_color;
    texture_coordinate = texture_coordinates[gl_VertexID % 6];
    switch (shape_type) {
        case 0: {
            gl_Position = projection * view * vec4(position, 0, 1);
            break;
        }
    }
}