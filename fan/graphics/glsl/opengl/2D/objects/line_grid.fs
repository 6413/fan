#version 330

layout (location = 0) out vec4 o_attachment0;

in vec4 instance_color;
in vec3 instance_position;
in vec3 frag_position;

uniform vec2 window_size;
uniform vec2 scaler;

void main() {
	vec2 grid_size = scaler;
	float gridX = mod(gl_FragCoord.x, grid_size.x);
	float gridY = mod(gl_FragCoord.y, grid_size.y);
	if ((gridX) < 1.0 || (gridY) < 1.0) {
		o_attachment0 = instance_color;
	}
	else {
		discard;
	}
}