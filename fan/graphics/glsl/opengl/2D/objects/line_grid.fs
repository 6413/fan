#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;
in vec3 instance_position;
in vec3 frag_position;

uniform vec2 window_size;
uniform vec2 scaler;

void main() {
	vec2 grid_size = vec2(scaler.x, scaler.y); // user input
	vec2 grid_thickness = vec2(0.0025, 0.0025); // user input
	vec2 nl = grid_size * grid_thickness; // NeedLess, calculate inside vertex shader

	vec2 rc = texture_coordinate * (grid_thickness + 1.0); // relative coordinate
	rc *= grid_size;

	float m0 = mod(rc.x, 1.0); // modulo0
	float m1 = mod(rc.y, 1.0); // modulo1
	if(m0 < nl.x || m1 < nl.y){
		o_attachment0 = instance_color;
	}
	else{
		discard;
	}
}
