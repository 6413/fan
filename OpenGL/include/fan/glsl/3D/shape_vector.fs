#version 330 core

flat in int mode;
in vec2 texture_coordinate;
in vec4 color;

out vec4 shape_texture;

uniform sampler2D texture_sampler;

void main() {
    switch(mode) {
        case 1: {
            shape_texture = color;
            break;
		}
        case 2: {
            shape_texture = texture2D(texture_sampler, texture_coordinate);
            break;
        }
	}

} 