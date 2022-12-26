R"(
	#version 330

	in vec2 texture_coordinate;

	in vec4 instance_color;

	layout (location = 0) out vec4 o_color;

	uniform sampler2D _t00;
	uniform sampler2D _t01;

	uniform float bloom;

	float luma(vec3 color) {
		return dot(color, vec3(0.299, 0.587, 0.114));
	}

	void main() {
    vec3 result = vec3(0.0);
		vec4 hdrColor = texture(_t00, texture_coordinate);
		//vec3 bloomColor = texture(_t01, texture_coordinate).rgb;
		float b = bloom;
		result = mix(hdrColor, bloomColor, b); // linear interpolation
		o_color = result;
	}
)"