#version 120

//
//precision highp float;

attribute vec4 in_color;
attribute vec2 position;
attribute vec2 texture_coordinates;

varying vec4 color; 

uniform mat4 projection;
uniform mat4 view;

varying vec2 texture_coordinate;

void main() {
    color = in_color;
    texture_coordinate = texture_coordinates;
	gl_Position = projection * view * vec4(position, 0, 1);
}