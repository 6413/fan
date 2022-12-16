R"(
	#version 330

	in vec2 texture_coordinate;

	out vec4 o_color;

	uniform sampler2D _t00;
	uniform usampler2D _t01;

	void main() {
		vec3 actual_image = texture(_t00, texture_coordinate).rgb;
		uvec4 bit_map = texture(_t01, texture_coordinate);
		o_color = vec4(float(bit_map.r) / 255, 0, 0, 1);
	}
)"