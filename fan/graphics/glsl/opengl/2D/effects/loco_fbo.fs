
	#version 330

	in vec2 texture_coordinate;

	out vec4 o_color;

	uniform sampler2D _t00;
	uniform usampler2D _t01;
  uniform sampler2D _t02;

	void main() {
    vec3 actual = texture(_t00, texture_coordinate.xy).rgb;
//    FragColor = vec4(col, 1.0);
    o_color = vec4(actual, 1);
	}
