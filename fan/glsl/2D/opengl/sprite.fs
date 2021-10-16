#version 130

in vec2 texture_coordinate;
in float transparency;

out vec4 color;

uniform sampler2D texture_sampler;

void main() {

	vec2 flipped_y = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);
	
	color = texture(texture_sampler, flipped_y);
	color.a -= mod(transparency, 1);

}