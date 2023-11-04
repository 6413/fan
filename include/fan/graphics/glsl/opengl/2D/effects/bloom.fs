
	#version 330

	in vec2 texture_coordinate;

	in vec4 instance_color;

  layout (location = 0) out vec4 o_attachment0;
  layout (location = 1) out vec4 o_attachment1;

	uniform sampler2D _t00;
	uniform sampler2D _t01;

	uniform float bloom;

	float luma(vec3 color) {
		return dot(color, vec3(0.299, 0.587, 0.114));
	}

	void main() {
    vec3 result = vec3(0.0);
		vec3 hdrColor = texture(_t00, texture_coordinate).rgb;
		vec3 bloomColor = texture(_t01, texture_coordinate).rgb;
		//if (luma(bloomColor) < 0.5) {
		//	bloomColor *= luma(bloomColor);
		//}
		//result = bloomColor;
		float b = bloom;
		//if (hdrColor.r < 0.8 && hdrColor.g < 0.8 && hdrColor.b < 0.8) {
		//	bloomColor /= 5;
		//}
		result = mix(hdrColor, bloomColor, b); // linear interpolation
		//result = vec3(1.0) - exp(-result * 1.0);
		//// also gamma correct while we're at it
		//const float gamma = 2.2;
    //result = pow(result, vec3(1.0 / gamma));
		o_attachment0 = vec4(result, 1);
	}
