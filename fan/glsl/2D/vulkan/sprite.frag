#version 450

layout(location = 0) in vec2 fragment_texture_coordinate;

layout(location = 5) in float layout_rotation_vector;

layout(location = 0) out vec4 color;

layout(binding = 1) uniform sampler2D texture_sampler;

void main() {
	vec2 flipped_y = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);
	
	color = texture(texture_sampler, flipped_y);

	//color.a -= 1.0 - transparency;
}