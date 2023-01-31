R"(
	#version 330

	in vec2 texture_coordinate;

	out vec4 o_color;

	uniform sampler2D _t00;
	uniform usampler2D _t01;
  uniform sampler2D _t02;

	void main() {
		vec3 actual_image = texture(_t00, texture_coordinate).rgb;
		//uint bit_map = texture(_t01, texture_coordinate).r;
    vec3 light_map = texture(_t02, texture_coordinate).rgb;
// add + light_map
    o_color = vec4(actual_image + light_map, 1);
	}
)"