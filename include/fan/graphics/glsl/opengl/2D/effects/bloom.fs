R"(
	#version 330

	in vec2 texture_coordinate;

	in vec4 instance_color;

	out vec4 o_color;

	//uniform vec2 resolution;
	//uniform float filter_radius;

	uniform sampler2D _t00;
	uniform sampler2D _t01;

	void main() {
    vec3 result = vec3(0.0);
		vec3 hdrColor = texture(_t00, texture_coordinate).rgb;
		vec3 bloomColor = texture(_t01, texture_coordinate).rgb;
		//result = mix(bloomColor, hdrColor, 0.5); // linear interpolation
		o_color.rgb = hdrColor;
		o_color.a = texture(_t01, texture_coordinate).a;
		//result = vec3(1.0) - exp(-result * 0.001);
		//// also gamma correct while we're at it
		//const float gamma = 2.2;
  //  result = pow(result, vec3(1.0 / gamma));
		//o_color = vec4(result, 1);
	}
)"