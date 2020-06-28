#version 330 core

in vec2 texture_coordinate;
out vec4 shape_color;

uniform sampler2D texture_sampler;

uniform bool horizontal;

float weight[11] = float[11](0.000003, 0.000229, 0.005977, 0.060598, 0.24173, 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003);

in vec2 blur_texture_coordinates[11];

void main() {
    //shape_color = vec4(0);
    //for (int i = 0; i < 11; i++) {
    //    shape_color += texture(texture_sampler, blur_texture_coordinates[i]) * weight[i]; 
	//}
    shape_color = texture(texture_sampler, texture_coordinate);
} 