#version 130

precision highp float;

attribute vec4 in_color;
attribute vec2 position;

out vec4 color; 

uniform mat4 projection;
uniform mat4 view;
uniform int shape_type;

void main() {
    color = in_color;
    switch (shape_type) {
        case 0: {
            gl_Position = projection * view * vec4(position, 0, 1);
            break;
        }
    }
}