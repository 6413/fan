#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec2 instance_position;
in vec2 instance_size;
in vec4 instance_color;
in vec3 frag_position;

uniform vec2 matrix_size;
uniform vec2 viewport_size;
uniform vec2 scaler;

void main() {
	vec2 grid_size = vec2(10, 10); // user input
	vec2 grid_thickness = vec2(2.); // user input
	vec2 d = instance_size / grid_size;

	vec2 rpos = gl_FragCoord.xy - (instance_position - instance_size); // relative position

	float m0 = mod(rpos.x, d.x); // modulo0
	float m1 = mod(rpos.y, d.y); // modulo1
	if(m0 < grid_thickness.x || m1 < grid_thickness.y){
		o_attachment0 = instance_color;
	}
	else{
		discard;
	}
}
